import torch
import torch.nn as nn
from typing import List


class CrossNetwork(nn.Module):
    """Cross Network component of DCNv2 with vector/matrix parameterization."""

    def __init__(self, input_dim: int, num_layers: int, parameterization: str = "vector"):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.parameterization = parameterization

        if parameterization == "vector":
            # Vector parameterization: W_l is a vector of size input_dim
            self.cross_weights = nn.ParameterList([
                nn.Parameter(torch.randn(input_dim))
                for _ in range(num_layers)
            ])
        elif parameterization == "matrix":
            # Matrix parameterization: W_l is a matrix of size (input_dim, input_dim)
            self.cross_weights = nn.ParameterList([
                nn.Parameter(torch.randn(input_dim, input_dim))
                for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"Unknown parameterization: {parameterization}")

        # Bias terms
        self.cross_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Cross network output of shape (batch_size, input_dim)
        """
        x0 = x  # Keep the original input
        xi = x

        for i in range(self.num_layers):
            if self.parameterization == "vector":
                # Vector parameterization: x_{l+1} = x_0 ⊙ (W_l^T x_l) + b_l + x_l
                xl = torch.sum(xi * self.cross_weights[i], dim=1, keepdim=True)  # (B,1)
                xi = x0 * xl + xi + self.cross_biases[i]  # bias shape (d,)
            else:
                # Matrix parameterization: x_{l+1} = x_0 ⊙ (W_l x_l) + b_l + x_l
                xl = torch.matmul(xi, self.cross_weights[i])  # (B,d)
                xi = x0 * xl + xi + self.cross_biases[i]

        return xi


class DeepNetwork(nn.Module):
    """Deep Network component of DCNv2."""

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


class DCNv2(nn.Module):
    """
    DCN-V2: Improved Deep & Cross Network for Feature Cross Learning.

    Refactored to use unified embedding approach consistent with EmbMLP.
    All features (numerical and categorical) are processed through embedding layers.
    """

    def __init__(self,
                 num_features: int = 39,
                 embedding_dim: int = 8,
                 mix_hidden_dims: List[int] = [256, 128, 64, 32],
                 deep_hidden_dims: List[int] = [128, 64],
                 cross_layers: int = 3,
                 cross_parameterization: str = "vector",
                 deep_activation: str = "relu",
                 dropout: float = 0.2,
                 hash_size: int = 1000000):
        """
        Args:
            num_features: Total number of features (39 for Criteo)
            embedding_dim: Dimension of embeddings for all features
            mix_hidden_dims: Hidden layer dimensions for mix layer (final MLP)
            deep_hidden_dims: Hidden layer dimensions for deep network
            cross_layers: Number of cross layers
            cross_parameterization: "vector" or "matrix" for cross network
            deep_activation: Activation function for deep network
            dropout: Dropout rate
            hash_size: Hash size for all features
        """
        super().__init__()

        self.num_features = num_features
        self.embedding_dim = embedding_dim

        # 统一embedding处理：所有39个特征都通过embedding层
        # 与EmbMLP保持完全一致的架构
        self.embeddings = nn.ModuleList([
            nn.Embedding(hash_size, embedding_dim)
            for _ in range(num_features)
        ])

        # 输入维度：num_features * embedding_dim
        self.input_dim = num_features * embedding_dim

        # Cross Network - 保持原始输入维度
        self.cross_network = CrossNetwork(
            self.input_dim,
            cross_layers,
            cross_parameterization
        )

        # Deep Network - 使用deep_hidden_dims和deep_activation
        self.deep_network = DeepNetwork(self.input_dim, deep_hidden_dims, deep_activation, dropout)

        # Mix layer - 使用mix_hidden_dims构建MLP，与EmbeddingMLP保持一致
        # Input: concatenation of cross output (input_dim) and deep output (deep_hidden_dims[-1])
        mix_input_dim = self.input_dim + deep_hidden_dims[-1]

        # 构建Mix layer MLP
        mix_layers = []
        prev_dim = mix_input_dim

        for hidden_dim in mix_hidden_dims:
            mix_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # 最后的输出层
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

        # Initialize cross network biases
        for bias in self.cross_network.cross_biases:
            nn.init.zeros_(bias)

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

        # 所有特征都通过embedding处理 - 与EmbMLP完全一致
        embeddings = []
        for i in range(x.size(1)):
            if i < len(self.embeddings):
                feature_idx = x[:, i].long()  # 确保是long类型的索引
                embedded = self.embeddings[i](feature_idx)
                embeddings.append(embedded)

        # 拼接所有embedding
        if embeddings:
            combined_features = torch.cat(embeddings, dim=1)
        else:
            combined_features = torch.empty(batch_size, 0, device=x.device)

        # Cross Network forward - 保持原始维度
        cross_output = self.cross_network(combined_features)

        # Deep Network forward - 降维到deep_dims[-1]
        deep_output = self.deep_network(combined_features)

        # 拼接Cross和Deep输出
        mix_input = torch.cat([cross_output, deep_output], dim=1)

        # Mix layer forward - 通过MLP处理
        logits = self.mix_layer(mix_input)

        return logits