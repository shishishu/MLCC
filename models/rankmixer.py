import torch
import torch.nn as nn
from typing import List, Optional


class TokenMix(nn.Module):
    """
    Token Mixing operation with optional Add & Norm.

    Rearranges embedding dimensions across multiple heads for feature interaction,
    optionally followed by residual connection and layer normalization.
    LayerNorm is applied on the last dimension after reshaping to [B, T, D].
    """

    def __init__(self,
                 num_features: int,
                 dim_multiplier: int,
                 num_heads: Optional[int] = None,
                 use_add_norm: bool = True,
                 layer_idx: int = 0):
        super().__init__()
        self.num_features = num_features  # T
        self.dim_multiplier = dim_multiplier
        self.D = num_features * dim_multiplier
        self.H = num_heads if num_heads is not None else num_features
        self.head_dim = self.D // self.H
        self.use_add_norm = use_add_norm

        if use_add_norm:
            # LayerNorm without learnable parameters (scale=False, center=False)
            self.layer_norm = nn.LayerNorm(self.D, elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, T*D)
        Returns:
            Output tensor of shape (batch_size, T*D)
        """
        batch_size = x.size(0)
        T = self.num_features
        D = self.D
        H = self.H
        head_dim = self.head_dim

        # Reshape to [batch_size, T, D] for easier head splitting
        reshaped_emb = x.view(batch_size, T, D)

        # Split into H heads along the D dimension
        heads = []
        for h in range(H):
            start_idx = h * head_dim
            end_idx = (h + 1) * head_dim
            head_h = reshaped_emb[:, :, start_idx:end_idx]  # [batch_size, T, head_dim]
            # Reshape to [batch_size, 1, T*head_dim]
            head_h_reshaped = head_h.reshape(batch_size, 1, T * head_dim)
            heads.append(head_h_reshaped)

        # Concatenate all heads along axis=1 to get [batch_size, H, T*head_dim]
        heads_concat = torch.cat(heads, dim=1)
        # Reshape to [batch_size, T*D]
        mixed_emb = heads_concat.reshape(batch_size, T * D)

        # Optional Add & Norm: Residual connection + Layer Normalization
        if self.use_add_norm:
            residual_output = mixed_emb + x  # Add residual connection
            # Apply LayerNorm on last dimension: [B, T*D] -> [B, T, D] -> LN -> [B, T*D]
            reshaped = residual_output.view(batch_size, T, D)   # [B, T, D]
            normed = self.layer_norm(reshaped)                   # LN over last dim (D)
            output = normed.reshape(batch_size, T * D)           # [B, T*D]
        else:
            output = mixed_emb

        return output


class PerTokenFFN(nn.Module):
    """
    Per-token Feed Forward Network: Apply separate FFN to each token.

    This module applies a separate two-layer FFN (D -> k*D -> D) to each of the T tokens independently.
    """

    def __init__(self,
                 num_features: int,
                 dim_multiplier: int,
                 ffn_multiplier: int = 4,
                 layer_idx: int = 0):
        super().__init__()
        self.num_features = num_features  # T
        self.dim_multiplier = dim_multiplier
        self.D = num_features * dim_multiplier
        self.ffn_hidden_dim = ffn_multiplier * self.D

        # Create FFN layers for each token
        self.ffn_layers = nn.ModuleList()
        for t in range(num_features):
            ffn = nn.Sequential(
                nn.Linear(self.D, self.ffn_hidden_dim),
                nn.GELU(),
                nn.Linear(self.ffn_hidden_dim, self.D)
            )
            self.ffn_layers.append(ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, T*D)
        Returns:
            Output tensor of shape (batch_size, T*D)
        """
        batch_size = x.size(0)
        T = self.num_features
        D = self.D

        # Reshape to [batch_size, T, D] to process each token separately
        reshaped_emb = x.view(batch_size, T, D)

        # Apply FFN to each token separately
        token_outputs = []
        for t in range(T):
            token_t = reshaped_emb[:, t, :]  # [batch_size, D]
            ffn_output = self.ffn_layers[t](token_t)  # [batch_size, D]
            token_outputs.append(ffn_output)

        # Stack tokens back to [batch_size, T, D]
        processed_tokens = torch.stack(token_outputs, dim=1)

        # Reshape back to [batch_size, T*D]
        output = processed_tokens.reshape(batch_size, T * D)

        return output


class RankMixerLayer(nn.Module):
    """
    A single RankMixer layer consisting of Token Mix and Per-token FFN.
    """

    def __init__(self,
                 num_features: int,
                 dim_multiplier: int,
                 num_heads: Optional[int] = None,
                 use_add_norm: bool = True,
                 ffn_multiplier: int = 4,
                 layer_idx: int = 0):
        super().__init__()

        self.token_mix = TokenMix(
            num_features=num_features,
            dim_multiplier=dim_multiplier,
            num_heads=num_heads,
            use_add_norm=use_add_norm,
            layer_idx=layer_idx
        )

        self.per_token_ffn = PerTokenFFN(
            num_features=num_features,
            dim_multiplier=dim_multiplier,
            ffn_multiplier=ffn_multiplier,
            layer_idx=layer_idx
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, T*D)
        Returns:
            Output tensor of shape (batch_size, T*D)
        """
        # Token Mix with optional Add & Norm
        mixed_emb = self.token_mix(x)

        # Per-token FFN
        output = self.per_token_ffn(mixed_emb)

        return output


class RankMixer(nn.Module):
    """
    RankMixer: A CTR prediction model using token mixing and per-token FFN.

    Refactored to use unified embedding approach consistent with DCNv2 and EmbMLP.
    All features (numerical and categorical) are processed through embedding layers.
    """

    def __init__(self,
                 num_features: int = 39,
                 embedding_dim: int = 8,
                 dim_multiplier: int = 10,
                 num_layers: int = 1,
                 num_heads: Optional[int] = None,
                 use_add_norm: bool = True,
                 ffn_multiplier: int = 4,
                 mix_hidden_dims: List[int] = [256, 128, 64, 32],
                 dropout: float = 0.2,
                 hash_size: int = 1000000):
        """
        Args:
            num_features: Total number of features (39 for Criteo)
            embedding_dim: Dimension of embeddings for all features
            dim_multiplier: Multiplier for token dimension (D = T * dim_multiplier)
            num_layers: Number of RankMixer layers to apply
            num_heads: Number of heads for token mixing, if None uses num_features
            use_add_norm: Whether to apply Add & Norm in token mixing
            ffn_multiplier: Multiplier for FFN hidden dimension
            mix_hidden_dims: Hidden layer dimensions for final MLP
            dropout: Dropout rate
            hash_size: Hash size for all features
        """
        super().__init__()

        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.dim_multiplier = dim_multiplier
        self.num_layers = num_layers

        # Unified embedding processing: all features go through embedding layers
        # Consistent with DCNv2 and EmbMLP architecture
        self.embeddings = nn.ModuleList([
            nn.Embedding(hash_size, embedding_dim)
            for _ in range(num_features)
        ])

        # Tokenization: map each feature embedding to output_dim
        self.output_dim = num_features * dim_multiplier
        self.tokenization_layers = nn.ModuleList([
            nn.Linear(embedding_dim, self.output_dim)
            for _ in range(num_features)
        ])

        # RankMixer Layers
        self.rankmixer_layers = nn.ModuleList([
            RankMixerLayer(
                num_features=num_features,
                dim_multiplier=dim_multiplier,
                num_heads=num_heads,
                use_add_norm=use_add_norm,
                ffn_multiplier=ffn_multiplier,
                layer_idx=i
            )
            for i in range(num_layers)
        ])

        # Final MLP for prediction
        # Input: num_features * output_dim (T*D)
        mix_input_dim = num_features * self.output_dim

        # Build final MLP
        mix_layers = []
        prev_dim = mix_input_dim

        for hidden_dim in mix_hidden_dims:
            mix_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Final output layer
        mix_layers.append(nn.Linear(prev_dim, 1))

        self.mix_layer = nn.Sequential(*mix_layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with unified embedding features.

        Args:
            x: Input features tensor of shape (batch_size, num_features)
               Contains hash indices for all features (numerical and categorical)

        Returns:
            logits: Output logits of shape (batch_size, 1)
        """
        batch_size = x.size(0)

        # All features go through embedding processing - consistent with DCNv2
        embeddings = []
        for i in range(x.size(1)):
            if i < len(self.embeddings):
                feature_idx = x[:, i].long()  # Ensure long type indices
                embedded = self.embeddings[i](feature_idx)  # [batch_size, embedding_dim]
                embeddings.append(embedded)

        # Tokenization: map each feature embedding to output_dim
        mapped_embeddings = []
        for i, emb in enumerate(embeddings):
            mapped_emb = self.tokenization_layers[i](emb)  # [batch_size, output_dim]
            mapped_embeddings.append(mapped_emb)

        # Concatenate all mapped embeddings [batch_size, T*D]
        concat_emb = torch.cat(mapped_embeddings, dim=1)

        # Apply RankMixer layers
        current_emb = concat_emb
        for layer in self.rankmixer_layers:
            current_emb = layer(current_emb)

        # Final MLP for prediction
        logits = self.mix_layer(current_emb)

        return logits
