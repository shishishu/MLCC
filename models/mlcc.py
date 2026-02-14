import torch
import torch.nn as nn
from typing import List, Optional


class DeepNetwork(nn.Module):
    """Deep Network component for MLCC."""

    def __init__(self, input_dim: int, hidden_dims: List[int], activation: str = "relu", dropout: float = 0.2):
        super().__init__()

        # 激活函数映射
        activation_map = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU()
        }

        if activation not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_map[activation],
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.deep_network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Deep network output
        """
        return self.deep_network(x)


class SENet(nn.Module):
    """Squeeze-and-Excitation Network for feature recalibration."""

    def __init__(self, hidden_sizes: tuple, name: str = 'SENet'):
        """
        Args:
            hidden_sizes: Tuple of layer sizes, e.g., (cross_slot_num, cross_slot_num // 2, cross_slot_num)
            name: Module name
        """
        super().__init__()
        self.name = name
        self.hidden_sizes = hidden_sizes

        # LayerNorm without learnable parameters (scale=False, center=False in TF)
        self.ln = nn.LayerNorm(hidden_sizes[1], elementwise_affine=False)

        # Build linear layers
        self.layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot normal initialization."""
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, S, E]

        Returns:
            Output tensor of shape [B, S, E] with attention applied
        """
        # [B, S, E] -> [B, S] (squeeze)
        z = torch.mean(x, dim=2)

        # First layer: [B, S] -> [B, S//2]
        z = self.layers[0](z)
        z = torch.relu(z)
        z = self.ln(z)

        # Second layer: [B, S//2] -> [B, S]
        a = self.layers[1](z)
        a = torch.sigmoid(a)

        # [B, S] -> [B, S, 1]
        a = a.unsqueeze(-1)

        # Excitation: [B, S, 1] * [B, S, E] -> [B, S, E]
        return a * x


class MLCC(nn.Module):
    """
    Multi-Level Cross Compression (MLCC) for feature interaction learning.

    This module implements a sophisticated feature crossing mechanism using:
    1. Dynamic MLP weights generated from compressed features
    2. Multi-head feature crossing
    3. Squeeze-and-Excitation for feature recalibration
    """

    def __init__(self,
                 cross_slot_num: int,
                 emb_size: int = 16,
                 head_num: int = 2,
                 dmlp_dim_list: List[int] = [1],
                 dmlp_dim_concat: List[int] = [],
                 emb_size2: Optional[int] = None,
                 num_inputs: int = 1,
                 emb_raw_dim: Optional[int] = None,
                 name: str = 'MLCC'):
        """
        Args:
            cross_slot_num: Number of feature slots (e.g., number of features)
            emb_size: Embedding dimension for input features
            head_num: Number of attention heads
            dmlp_dim_list: Dimension list for dynamic MLP layers
            dmlp_dim_concat: Indices of layers to concatenate (e.g., [0, 1])
                            - If 0 is in dmlp_dim_concat, raw embeddings will be concatenated
            emb_size2: Output embedding dimension (defaults to emb_size)
            num_inputs: Number of input embeddings to process
            emb_raw_dim: Dimension of raw embeddings when 0 is in dmlp_dim_concat (defaults to emb_size)
            name: Module name
        """
        super().__init__()

        self.cross_slot_num = cross_slot_num
        self.emb_size = emb_size
        self.head_num = head_num
        self.num_inputs = num_inputs
        self.name_prefix = name

        # Calculate dynamic MLP parameters
        self.dmlp_param_dict = self._calc_dmlp_param(self.emb_size, dmlp_dim_list)
        self.dmlp_cnt_sum = sum(self.dmlp_param_dict['cnt_list'])
        self.dmlp_dim_last = self.dmlp_param_dict['dim_list'][-1]
        self.dmlp_dim_concat = dmlp_dim_concat
        self.dmlp_concat_dim_top = 0

        # Calculate concatenation dimensions
        if self.dmlp_dim_concat:
            assert len(dmlp_dim_concat) < len(self.dmlp_param_dict['dim_list']), \
                "dmlp_dim_concat length must be less than number of MLP layers"
            for _dim in dmlp_dim_concat:
                concat_dim = self.dmlp_param_dict['dim_list'][_dim]
                if _dim > 0:
                    self.dmlp_concat_dim_top += concat_dim

        self.dmlp_dim_ext = self.dmlp_dim_last + self.dmlp_concat_dim_top
        self.dmlp_out_dim = self.head_num * self.dmlp_dim_ext
        self.emb_size2 = emb_size2 if emb_size2 else self.emb_size

        # Determine raw embedding dimension for concatenation
        self.emb_raw_dim = emb_raw_dim if emb_raw_dim is not None else self.emb_size

        # Calculate output dimension sum (including raw concat if needed)
        self.dmlp_out_dim_sum = self.dmlp_out_dim * num_inputs
        if 0 in self.dmlp_dim_concat:
            self.dmlp_out_dim_sum += self.emb_raw_dim

        # First level compression matrices: [S*E, H*M] for each input
        self.compress_l1_list = nn.ParameterList([
            nn.Parameter(torch.empty(self.emb_size * self.cross_slot_num,
                                    self.head_num * self.dmlp_cnt_sum))
            for _ in range(num_inputs)
        ])

        # Second level compression matrix: [S, H*(Mo+Ct)+Cb, E2]
        self.compress_l2 = nn.Parameter(
            torch.empty(self.cross_slot_num, self.dmlp_out_dim_sum, self.emb_size2)
        )

        # Squeeze-and-Excitation Network
        self.se_dense = SENet((self.cross_slot_num,
                              self.cross_slot_num // 2,
                              self.cross_slot_num),
                             name=name + '_SENet')

        self._init_weights()

    def _calc_dmlp_param(self, input_size: int, dim_list: List[int]) -> dict:
        """
        Calculate parameter dimensions for dynamic MLP.

        Args:
            input_size: Input dimension
            dim_list: List of output dimensions for each layer

        Returns:
            Dictionary with 'dim_list' and 'cnt_list' (parameter counts)
        """
        param_dim_list = [input_size] + dim_list
        param_cnt_list = [x * y for x, y in zip(param_dim_list, param_dim_list[1:])]
        param_dict = {
            'dim_list': param_dim_list,
            'cnt_list': param_cnt_list,
        }
        return param_dict

    def _init_weights(self):
        """Initialize model weights using Xavier/Glorot normal initialization."""
        for param in self.compress_l1_list:
            nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.compress_l2)

    def _forward_compress_l1_cross(self,
                                   emb_input: torch.Tensor,
                                   compress_l1_mat: torch.Tensor,
                                   head_num: int) -> torch.Tensor:
        """
        Forward pass with L1 compression and dynamic cross network.

        Args:
            emb_input: Input embeddings [B, S*E]
            compress_l1_mat: First level compression matrix [S*E, H*M]
            head_num: Number of attention heads

        Returns:
            Crossed features [B, S, H*(Mo+Ct)]
        """
        batch_size = emb_input.size(0)

        # Reshape to 3D: [B, S*E] -> [B, S, E]
        emb_raw = emb_input.view(batch_size, self.cross_slot_num, self.emb_size)

        # Compress L1: [B, S*E] × [S*E, H*M] -> [B, H*M]
        emb_comp = torch.matmul(emb_input, compress_l1_mat)

        # Reshape to multi-head: [B, H*M] -> [B, H, M]
        emb_comp_r = emb_comp.view(batch_size, head_num, self.dmlp_cnt_sum)

        # Split into weight matrices for each dynamic MLP layer
        dmlp_weight_list = torch.split(emb_comp_r, self.dmlp_param_dict['cnt_list'], dim=-1)

        hidden = emb_raw  # [B, S, E]
        hidden_concat_list = []

        # Apply dynamic MLP layers with optional concatenation
        for idx, _dmlp_weight in enumerate(dmlp_weight_list):
            # Save intermediate outputs for concatenation if needed
            if idx > 0 and idx in self.dmlp_dim_concat:
                hidden_concat_list.append(hidden)

            i_dim = self.dmlp_param_dict['dim_list'][idx]
            o_dim = self.dmlp_param_dict['dim_list'][idx + 1]

            # Reshape weight: [B, H, Mx] -> [B, H, I, O]
            dmlp_weight = _dmlp_weight.view(batch_size, head_num, i_dim, o_dim)

            if idx == 0:
                # First layer: [B, S, I] × [B, H, I, O] -> [B, H, S, O]
                hidden = torch.einsum('bsi,bhio->bhso', hidden, dmlp_weight)
            else:
                # Subsequent layers: [B, H, S, I] × [B, H, I, O] -> [B, H, S, O]
                hidden = torch.matmul(hidden, dmlp_weight)

        # Concatenate intermediate outputs if specified
        if len(hidden_concat_list) > 0:
            hidden_concat_list.append(hidden)
            emb_cross = torch.cat(hidden_concat_list, dim=-1)  # [B, H, S, Mo+Ct]
        else:
            emb_cross = hidden  # [B, H, S, Mo]

        # Reshape: [B, H, S, Mo+Ct] -> [B, S, H, Mo+Ct] -> [B, S, H*(Mo+Ct)]
        emb_cross_t = emb_cross.permute(0, 2, 1, 3)
        emb_cross_r = emb_cross_t.reshape(batch_size, self.cross_slot_num,
                                         head_num * self.dmlp_dim_ext)

        return emb_cross_r

    def forward(self,
                emb_input_list: List[torch.Tensor],
                emb_raw_dim3: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of MLCC module.

        Args:
            emb_input_list: List of input embeddings, each of shape [B, S*E]
            emb_raw_dim3: Optional raw embeddings of shape [B, S, E_raw] for concatenation
                         Required when 0 is in dmlp_dim_concat

        Returns:
            Refined embeddings of shape [B, S*E2]
        """
        assert len(emb_input_list) == self.num_inputs, \
            f"Expected {self.num_inputs} inputs, got {len(emb_input_list)}"

        emb_cross_list = []

        # Process each input through L1 compression and cross network
        for idx, emb_input in enumerate(emb_input_list):
            _emb_cross = self._forward_compress_l1_cross(
                emb_input, self.compress_l1_list[idx], self.head_num
            )
            emb_cross_list.append(_emb_cross)

        # Concatenate raw input if specified
        if 0 in self.dmlp_dim_concat:
            assert emb_raw_dim3 is not None, \
                "emb_raw_dim3 must be provided when 0 is in dmlp_dim_concat"
            # Validate dimension matches initialization parameter
            assert emb_raw_dim3.size(-1) == self.emb_raw_dim, \
                f"emb_raw_dim3 last dimension {emb_raw_dim3.size(-1)} must match emb_raw_dim {self.emb_raw_dim}"
            emb_cross_list.append(emb_raw_dim3)

        # Concatenate all crossed features: [B, S, H*(Mo+Ct)+Cb]
        emb_cross_res = torch.cat(emb_cross_list, dim=-1)

        # Compress L2: [B, S, H*(Mo+Ct)+Cb] × [S, H*(Mo+Ct)+Cb, E2] -> [B, S, E2]
        emb_refined = torch.einsum('bch,che->bce', emb_cross_res, self.compress_l2)

        # Flatten: [B, S, E2] -> [B, S*E2]
        emb_refined_flat = emb_refined.reshape(-1, self.cross_slot_num * self.emb_size2)

        # Apply Squeeze-and-Excitation
        emb_concat = emb_refined_flat
        emb_se = self.se_dense(emb_concat.reshape(-1, self.cross_slot_num, self.emb_size2))

        # Flatten output: [B, S, E2] -> [B, S*E2]
        res = emb_se.reshape(-1, self.cross_slot_num * self.emb_size2)

        return res


class MLCCModel(nn.Module):
    """
    Complete MLCC model for CTR prediction.

    Architecture: Embeddings → MLCC → DNN → Output
    Similar to EmbeddingMLP but with MLCC for feature interaction.
    """

    def __init__(self,
                 num_features: int = 39,
                 embedding_dim_total: int = 8,
                 embedding_dim_block: int = 8,
                 hidden_dims: List[int] = [128, 64],
                 head_num: int = 2,
                 dmlp_dim_list: List[int] = [1],
                 dmlp_dim_concat: List[int] = [],
                 emb_size2: int = 8,
                 deep_activation: str = "relu",
                 dropout: float = 0.2,
                 hash_size: int = 1000000):
        """
        Args:
            num_features: Total number of features (39 for Criteo)
            embedding_dim_total: Total embedding dimension for each feature
            embedding_dim_block: Block size for embedding split (MLCC emb_size)
            hidden_dims: Hidden layer dimensions for DNN
            head_num: Number of attention heads in MLCC
            dmlp_dim_list: Dimension list for dynamic MLP layers in MLCC
            dmlp_dim_concat: Indices of layers to concatenate in MLCC
            emb_size2: Output embedding dimension for MLCC
            deep_activation: Activation function for DNN
            dropout: Dropout rate
            hash_size: Hash size for all features
        """
        super().__init__()

        self.num_features = num_features
        self.embedding_dim_total = embedding_dim_total
        self.embedding_dim_block = embedding_dim_block
        self.emb_size2 = emb_size2

        # Validate dimensions
        assert self.embedding_dim_total % self.embedding_dim_block == 0, \
            f"embedding_dim_total ({self.embedding_dim_total}) must be divisible by embedding_dim_block ({self.embedding_dim_block})"

        # Calculate number of blocks
        self.num_blocks = self.embedding_dim_total // self.embedding_dim_block

        # Embedding layers for all features (using total dimension)
        self.embeddings = nn.ModuleList([
            nn.Embedding(hash_size, self.embedding_dim_total)
            for _ in range(num_features)
        ])

        # Input dimension: num_features * embedding_dim_total
        self.input_dim = num_features * self.embedding_dim_total

        # MLCC output dimension
        self.mlcc_output_dim = num_features * emb_size2

        # MLCC module for feature crossing
        self.mlcc = MLCC(
            cross_slot_num=num_features,
            emb_size=self.embedding_dim_block,
            head_num=head_num,
            dmlp_dim_list=dmlp_dim_list,
            dmlp_dim_concat=dmlp_dim_concat,
            emb_size2=emb_size2,
            num_inputs=self.num_blocks,
            emb_raw_dim=self.embedding_dim_total,
            name='MLCC'
        )

        # DNN on top of MLCC output (using MLCC output dimension)
        self.dnn = DeepNetwork(self.mlcc_output_dim, hidden_dims, deep_activation, dropout)

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        # Initialize weights
        self._init_weights()

    def _emb_split_block(self, emb_input: torch.Tensor) -> List[torch.Tensor]:
        """
        Split embeddings into blocks.

        Args:
            emb_input: Combined embeddings of shape [B, S*E_total]

        Returns:
            List of embedding blocks, each of shape [B, S*E_block]
        """
        batch_size = emb_input.size(0)
        slots = self.num_features

        # Reshape to [B, S, E_total]
        emb_t = emb_input.view(batch_size, slots, self.embedding_dim_total)

        # Split along embedding dimension: [B, S, E_total] -> n x [B, S, E_block]
        emb_split_list = torch.split(emb_t, self.embedding_dim_block, dim=2)

        # Flatten each block: [B, S, E_block] -> [B, S*E_block]
        emb_list = [x.reshape(batch_size, slots * self.embedding_dim_block) for x in emb_split_list]

        return emb_list

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
        Forward pass.

        Args:
            x: Input features tensor of shape (batch_size, num_features)
               Contains hash indices for all features

        Returns:
            logits: Output logits of shape (batch_size, 1)
        """
        batch_size = x.size(0)

        # All features through embedding layers
        embeddings = []
        for i in range(x.size(1)):
            if i < len(self.embeddings):
                feature_idx = x[:, i].long()
                embedded = self.embeddings[i](feature_idx)
                embeddings.append(embedded)

        # Concatenate all embeddings: [B, S*E_total]
        if embeddings:
            combined_features = torch.cat(embeddings, dim=1)
        else:
            combined_features = torch.empty(batch_size, 0, device=x.device)

        # Generate emb_list by splitting embeddings into blocks
        emb_list = self._emb_split_block(combined_features)  # List of [B, S*E_block]

        # Generate emb_dim3: reshape to [B, S, E_total]
        emb_dim3 = combined_features.view(batch_size, self.num_features, self.embedding_dim_total)

        # MLCC forward with emb_list and emb_dim3: -> [B, S*E2]
        mlcc_output = self.mlcc(emb_list, emb_dim3)

        # DNN forward: [B, S*E2] -> [B, hidden_dims[-1]]
        dnn_output = self.dnn(mlcc_output)

        # Output layer: [B, hidden_dims[-1]] -> [B, 1]
        logits = self.output_layer(dnn_output)

        return logits
