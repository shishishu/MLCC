import torch
import torch.nn as nn
from typing import List


class LinearCompressBlock(nn.Module):
    """Linear Compression Block - compresses number of embeddings."""

    def __init__(self, num_emb_in: int, num_emb_out: int):
        """
        Args:
            num_emb_in: Number of input embeddings
            num_emb_out: Number of output embeddings
        """
        super().__init__()
        self.num_emb_in = num_emb_in
        self.num_emb_out = num_emb_out

        # Linear transformation weight (num_emb_in, num_emb_out)
        self.weight = nn.Parameter(torch.randn(num_emb_in, num_emb_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, num_emb_in, dim_emb)

        Returns:
            Output tensor of shape (batch_size, num_emb_out, dim_emb)
        """
        # (bs, num_emb_in, dim_emb) -> (bs, dim_emb, num_emb_in)
        x = x.transpose(1, 2)

        # (bs, dim_emb, num_emb_in) @ (num_emb_in, num_emb_out) -> (bs, dim_emb, num_emb_out)
        x = torch.matmul(x, self.weight)

        # (bs, dim_emb, num_emb_out) -> (bs, num_emb_out, dim_emb)
        x = x.transpose(1, 2)

        return x


class FactorizationMachineBlock(nn.Module):
    """Factorization Machine Block for feature interaction."""

    def __init__(self, num_emb_in: int, num_emb_out: int, dim_emb: int, rank: int):
        """
        Args:
            num_emb_in: Number of input embeddings
            num_emb_out: Number of output embeddings
            dim_emb: Embedding dimension
            rank: Rank for factorization
        """
        super().__init__()
        self.num_emb_in = num_emb_in
        self.num_emb_out = num_emb_out
        self.dim_emb = dim_emb
        self.rank = rank

        # Layer normalization
        self.norm = nn.LayerNorm(num_emb_in * rank)

        # Rank weight (num_emb_in, rank)
        self.rank_weight = nn.Parameter(torch.randn(num_emb_in, rank))

        # MLP for transformation
        self.mlp = nn.Sequential(
            nn.Linear(num_emb_in * rank, num_emb_out * dim_emb),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, num_emb_in, dim_emb)

        Returns:
            Output tensor of shape (batch_size, num_emb_out, dim_emb)
        """
        # (bs, num_emb_in, dim_emb) -> (bs, dim_emb, num_emb_in)
        x_t = x.transpose(1, 2)

        # (bs, dim_emb, num_emb_in) @ (num_emb_in, rank) -> (bs, dim_emb, rank)
        x_rank = torch.matmul(x_t, self.rank_weight)

        # (bs, num_emb_in, dim_emb) @ (bs, dim_emb, rank) -> (bs, num_emb_in, rank)
        x_fm = torch.matmul(x, x_rank)

        # (bs, num_emb_in, rank) -> (bs, num_emb_in * rank)
        x_flat = x_fm.reshape(-1, self.num_emb_in * self.rank)

        # (bs, num_emb_in * rank) -> (bs, num_emb_out * dim_emb)
        x_mlp = self.mlp(self.norm(x_flat))

        # (bs, num_emb_out * dim_emb) -> (bs, num_emb_out, dim_emb)
        output = x_mlp.reshape(-1, self.num_emb_out, self.dim_emb)

        return output


class ResidualProjection(nn.Module):
    """Residual projection for dimension matching."""

    def __init__(self, num_emb_in: int, num_emb_out: int):
        """
        Args:
            num_emb_in: Number of input embeddings
            num_emb_out: Number of output embeddings
        """
        super().__init__()
        self.num_emb_in = num_emb_in
        self.num_emb_out = num_emb_out

        # Projection weight (num_emb_in, num_emb_out)
        self.weight = nn.Parameter(torch.randn(num_emb_in, num_emb_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, num_emb_in, dim_emb)

        Returns:
            Output tensor of shape (batch_size, num_emb_out, dim_emb)
        """
        # (bs, num_emb_in, dim_emb) -> (bs, dim_emb, num_emb_in)
        x = x.transpose(1, 2)

        # (bs, dim_emb, num_emb_in) @ (num_emb_in, num_emb_out) -> (bs, dim_emb, num_emb_out)
        x = torch.matmul(x, self.weight)

        # (bs, dim_emb, num_emb_out) -> (bs, num_emb_out, dim_emb)
        x = x.transpose(1, 2)

        return x


class WukongLayer(nn.Module):
    """Single Wukong layer combining LCB and FMB."""

    def __init__(self, num_emb_in: int, dim_emb: int, num_emb_lcb: int,
                 num_emb_fmb: int, rank_fmb: int):
        """
        Args:
            num_emb_in: Number of input embeddings
            dim_emb: Embedding dimension
            num_emb_lcb: Number of output embeddings from LCB
            num_emb_fmb: Number of output embeddings from FMB
            rank_fmb: Rank for FMB factorization
        """
        super().__init__()
        self.num_emb_in = num_emb_in
        self.dim_emb = dim_emb
        self.num_emb_lcb = num_emb_lcb
        self.num_emb_fmb = num_emb_fmb
        self.num_emb_out = num_emb_lcb + num_emb_fmb

        # Layer normalization
        self.norm = nn.LayerNorm(dim_emb)

        # Linear Compression Block
        self.lcb = LinearCompressBlock(num_emb_in, num_emb_lcb)

        # Factorization Machine Block
        self.fmb = FactorizationMachineBlock(num_emb_in, num_emb_fmb, dim_emb, rank_fmb)

        # Residual projection if dimensions don't match
        if num_emb_in != self.num_emb_out:
            self.residual_projection = ResidualProjection(num_emb_in, self.num_emb_out)
        else:
            self.residual_projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, num_emb_in, dim_emb)

        Returns:
            Output tensor of shape (batch_size, num_emb_lcb + num_emb_fmb, dim_emb)
        """
        # (bs, num_emb_in, dim_emb) -> (bs, num_emb_lcb, dim_emb)
        lcb_out = self.lcb(x)

        # (bs, num_emb_in, dim_emb) -> (bs, num_emb_fmb, dim_emb)
        fmb_out = self.fmb(x)

        # Concatenate FMB and LCB outputs
        # (bs, num_emb_fmb, dim_emb), (bs, num_emb_lcb, dim_emb) -> (bs, num_emb_out, dim_emb)
        output = torch.cat([fmb_out, lcb_out], dim=1)

        # Residual connection with layer normalization
        # (bs, num_emb_out, dim_emb) -> (bs, num_emb_out, dim_emb)
        output = self.norm(output + self.residual_projection(x))

        return output


class WukongInteraction(nn.Module):
    """
    Wukong feature interaction module.
    Stacks multiple WukongLayer for deep feature interaction.
    """

    def __init__(self, num_emb_in: int, dim_emb: int, num_layers: int,
                 num_emb_lcb: int, num_emb_fmb: int, rank_fmb: int):
        """
        Args:
            num_emb_in: Number of input embeddings (number of features)
            dim_emb: Embedding dimension
            num_layers: Number of Wukong layers
            num_emb_lcb: Number of output embeddings from LCB in each layer
            num_emb_fmb: Number of output embeddings from FMB in each layer
            rank_fmb: Rank for FMB factorization
        """
        super().__init__()
        self.num_emb_in = num_emb_in
        self.dim_emb = dim_emb
        self.num_layers = num_layers
        self.num_emb_lcb = num_emb_lcb
        self.num_emb_fmb = num_emb_fmb
        self.num_emb_out = num_emb_lcb + num_emb_fmb

        # Stack multiple Wukong layers
        self.interaction_layers = nn.ModuleList()
        for i in range(num_layers):
            # First layer takes num_emb_in, subsequent layers take num_emb_out
            layer_input_dim = num_emb_in if i == 0 else self.num_emb_out
            self.interaction_layers.append(
                WukongLayer(
                    layer_input_dim,
                    dim_emb,
                    num_emb_lcb,
                    num_emb_fmb,
                    rank_fmb
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, num_emb_in * dim_emb)

        Returns:
            Output tensor of shape (batch_size, num_emb_out * dim_emb)
        """
        # Reshape to 3D: (batch_size, num_emb_in * dim_emb) -> (batch_size, num_emb_in, dim_emb)
        batch_size = x.size(0)
        x = x.reshape(batch_size, self.num_emb_in, self.dim_emb)

        # Apply Wukong layers sequentially
        for layer in self.interaction_layers:
            x = layer(x)

        # Flatten back: (batch_size, num_emb_out, dim_emb) -> (batch_size, num_emb_out * dim_emb)
        output = x.reshape(batch_size, -1)

        return output


class Wukong(nn.Module):
    """
    Wukong: Deep CTR prediction model with efficient feature interaction.

    Reference: https://github.com/clabrugere/wukong-recommendation

    Refactored to use unified embedding approach consistent with EmbMLP and DCNv2.
    All features (numerical and categorical) are processed through embedding layers.
    """

    def __init__(self,
                 num_features: int = 39,
                 embedding_dim: int = 8,
                 num_layers: int = 2,
                 num_emb_lcb: int = 16,
                 num_emb_fmb: int = 16,
                 rank_fmb: int = 32,
                 mlp_hidden_dims: List[int] = [256, 128, 64, 32],
                 dropout: float = 0.2,
                 hash_size: int = 1000000):
        """
        Args:
            num_features: Total number of features (39 for Criteo)
            embedding_dim: Dimension of embeddings for all features
            num_layers: Number of Wukong interaction layers
            num_emb_lcb: Number of output embeddings from LCB in each layer
            num_emb_fmb: Number of output embeddings from FMB in each layer
            rank_fmb: Rank for FMB factorization
            mlp_hidden_dims: Hidden layer dimensions for final MLP
            dropout: Dropout rate
            hash_size: Hash size for all features
        """
        super().__init__()

        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Unified embedding processing: all features go through embedding layers
        # Consistent with EmbMLP and DCNv2 architecture
        self.embeddings = nn.ModuleList([
            nn.Embedding(hash_size, embedding_dim)
            for _ in range(num_features)
        ])

        # Input dimension: num_features * embedding_dim
        self.input_dim = num_features * embedding_dim

        # Wukong interaction module
        self.wukong_interaction = WukongInteraction(
            num_emb_in=num_features,
            dim_emb=embedding_dim,
            num_layers=num_layers,
            num_emb_lcb=num_emb_lcb,
            num_emb_fmb=num_emb_fmb,
            rank_fmb=rank_fmb
        )

        # Output dimension from Wukong interaction
        self.wukong_output_dim = (num_emb_lcb + num_emb_fmb) * embedding_dim

        # Final MLP layers
        mlp_layers = []
        prev_dim = self.wukong_output_dim

        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        mlp_layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*mlp_layers)

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
            elif isinstance(module, nn.Parameter):
                nn.init.xavier_uniform_(module)

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

        # All features go through embedding processing - consistent with EmbMLP
        embeddings = []
        for i in range(x.size(1)):
            if i < len(self.embeddings):
                feature_idx = x[:, i].long()  # Ensure long type for indexing
                embedded = self.embeddings[i](feature_idx)
                embeddings.append(embedded)

        # Concatenate all embeddings
        if embeddings:
            combined_features = torch.cat(embeddings, dim=1)
        else:
            combined_features = torch.empty(batch_size, 0, device=x.device)

        # Wukong interaction
        interaction_output = self.wukong_interaction(combined_features)

        # Final MLP
        logits = self.mlp(interaction_output)

        return logits
