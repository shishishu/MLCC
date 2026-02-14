import torch
import torch.nn as nn
from typing import List


class EmbeddingMLP(nn.Module):
    """Simple Embedding + MLP baseline model for CTR prediction."""

    def __init__(self,
                 num_features: int,
                 categorical_features: List[int],
                 numerical_features: List[int],
                 embedding_dim: int = 16,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout: float = 0.8,
                 hash_size: int = 1000000):
        """
        Args:
            num_features: Total number of features
            categorical_features: Indices of categorical features
            numerical_features: Indices of numerical features
            embedding_dim: Dimension of embeddings for categorical features
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            hash_size: Hash size for categorical features
        """
        super().__init__()

        self.num_features = num_features
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.embedding_dim = embedding_dim

        # Embedding layers for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(hash_size, embedding_dim)
            for _ in categorical_features
        ])

        # Calculate input dimension for MLP
        mlp_input_dim = len(numerical_features) + len(categorical_features) * embedding_dim

        # MLP layers
        layers = []
        prev_dim = mlp_input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)

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

        # 所有特征都通过embedding处理
        # x contains hash indices, need to convert to embeddings
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

        # Forward through MLP
        logits = self.mlp(combined_features)

        return logits