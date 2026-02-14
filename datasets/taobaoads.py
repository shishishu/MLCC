import torch
import numpy as np
from typing import Tuple
from .common import BaseCTRDataset
from .feature_processor import FeatureIDProcessor


class TaobaoAdsDataset(BaseCTRDataset):
    """Taobao Ads Click-Through Rate Prediction Dataset."""

    def __init__(self,
                 data_path: str,
                 hash_size: int = 1000000,
                 delimiter: str = ','):
        """
        Args:
            data_path: Path to the Taobao Ads dataset file
            hash_size: Size of hash space for categorical features
            delimiter: Field delimiter (comma for Taobao Ads)
        """
        # 使用统一ID化处理，所有特征都是embedding特征
        # TaobaoAds总共17个特征，全部为类别特征
        # 格式: label, userid, time_stamp, adgroup_id, pid, cate_id, campaign_id,
        #      customer, brand, price, cms_segid, cms_group_id, final_gender_code,
        #      age_level, pvalue_level, shopping_level, occupation, new_user_class_level
        all_features = list(range(0, 17))  # 17个特征，全部通过embedding处理

        super().__init__(
            data_path=data_path,
            num_features=17,
            categorical_features=all_features,  # 所有特征都作为类别特征处理
            numerical_features=[],             # 不再有数值特征
            hash_size=hash_size,
            delimiter=delimiter,
            skip_header=False                  # 数据文件已移除header行
        )
        self.feature_processor = FeatureIDProcessor(hash_size)

    def parse_line(self, line: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parse a line from Taobao Ads dataset using unified feature ID processing.

        Format: label, userid, time_stamp, adgroup_id, pid, cate_id, campaign_id,
                customer, brand, price, cms_segid, cms_group_id, final_gender_code,
                age_level, pvalue_level, shopping_level, occupation, new_user_class_level
        All features are ID-ized and hashed to embedding indices.
        """
        # 优化：预分配字段分割，避免重复创建对象
        fields = line.split(self.delimiter, 18)  # 限制分割数量 (1 label + 17 features)

        # Parse label
        label = float(fields[0]) if len(fields) > 0 else 0.0

        # 处理17个类别特征 (field_id: 0-16, 对应 fields[1] - fields[17])
        feature_hashes = []
        for i in range(17):
            field_id = i
            field_idx = i + 1  # 跳过label字段
            value = fields[field_idx] if field_idx < len(fields) else ''
            feature_id = self.feature_processor.categorical_to_id(field_id, value)
            hash_val = self.feature_processor.id_to_hash(feature_id)
            feature_hashes.append(hash_val)

        # 优化：直接传入数组，减少tensor创建开销
        features = torch.from_numpy(np.array(feature_hashes, dtype=np.int64))
        label = torch.tensor(label, dtype=torch.float32)

        return features, label
