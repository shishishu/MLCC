import torch
import numpy as np
from typing import Tuple
from .common import BaseCTRDataset
from .feature_processor import FeatureIDProcessor


class AvazuDataset(BaseCTRDataset):
    """Avazu Click-Through Rate Prediction Dataset."""

    def __init__(self,
                 data_path: str,
                 hash_size: int = 1000000,
                 delimiter: str = ','):
        """
        Args:
            data_path: Path to the Avazu dataset file
            hash_size: Size of hash space for categorical features
            delimiter: Field delimiter (comma for Avazu)
        """
        # 使用统一ID化处理，所有特征都是embedding特征
        # Avazu总共22个特征，全部为类别特征
        all_features = list(range(0, 22))  # 22个特征，全部通过embedding处理

        super().__init__(
            data_path=data_path,
            num_features=22,
            categorical_features=all_features,  # 所有特征都作为类别特征处理
            numerical_features=[],             # 不再有数值特征
            hash_size=hash_size,
            delimiter=delimiter,
            skip_header=False                  # 数据文件已移除header行
        )
        self.feature_processor = FeatureIDProcessor(hash_size)

    def parse_line(self, line: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parse a line from Avazu dataset using unified feature ID processing.

        Format: id,click,hour,C1,banner_pos,site_id,site_domain,site_category,app_id,...
        All features are ID-ized and hashed to embedding indices.
        """
        # 优化：预分配字段分割，避免重复创建对象
        fields = line.split(self.delimiter, 24)  # 限制分割数量

        # 使用统一的特征处理器进行ID化和hash
        feature_hashes, label = self.feature_processor.process_avazu_line(fields)

        # 优化：直接传入数组，减少tensor创建开销
        features = torch.from_numpy(np.array(feature_hashes, dtype=np.int64))
        label = torch.tensor(label, dtype=torch.float32)

        return features, label