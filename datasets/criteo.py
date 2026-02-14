import torch
import numpy as np
from typing import Tuple, List
from .common import BaseCTRDataset
from .feature_processor import FeatureIDProcessor


class CriteoDataset(BaseCTRDataset):
    """Criteo Click Logs Dataset (Kaggle/1TB version)."""

    def __init__(self,
                 data_path: str,
                 hash_size: int = 1000000,
                 min_threshold: int = 10,
                 delimiter: str = '\t'):
        """
        Args:
            data_path: Path to the Criteo dataset file
            hash_size: Size of hash space for all features
            min_threshold: Minimum threshold for numerical feature clipping
            delimiter: Field delimiter (tab for Criteo)
        """
        # 使用统一ID化处理，所有特征都是embedding特征
        # Criteo总共39个特征：13个数值 + 26个类别
        all_features = list(range(0, 39))  # 39个特征，全部通过embedding处理

        super().__init__(
            data_path=data_path,
            num_features=39,
            categorical_features=all_features,  # 所有特征都作为类别特征处理
            numerical_features=[],             # 不再有数值特征
            hash_size=hash_size,
            delimiter=delimiter
        )
        self.min_threshold = min_threshold
        self.feature_processor = FeatureIDProcessor(hash_size)

    def parse_line(self, line: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parse a line from Criteo dataset using unified feature ID processing.

        Format: label\tI1\tI2\t...\tI13\tC1\tC2\t...\tC26
        All features are ID-ized and hashed to embedding indices.
        """
        # 优化：预分配字段分割，避免重复创建对象
        fields = line.split(self.delimiter, 39)  # 限制分割数量

        # 使用统一的特征处理器进行ID化和hash
        feature_hashes, label = self.feature_processor.process_criteo_line(
            fields, self.min_threshold
        )

        # 优化：直接传入数组，减少tensor创建开销
        features = torch.from_numpy(np.array(feature_hashes, dtype=np.int64))
        label = torch.tensor(label, dtype=torch.float32)

        return features, label