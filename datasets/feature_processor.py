"""
统一的特征ID化处理器
所有特征（数值+类别）都转换为field_id:value格式，然后统一hash处理
"""

import torch
import hashlib
from typing import List, Dict, Any, Tuple


class FeatureIDProcessor:
    """统一特征ID化处理器"""

    def __init__(self, hash_size: int = 1000000):
        """
        Args:
            hash_size: 哈希空间大小
        """
        self.hash_size = hash_size

    def numerical_to_id(self, field_id: int, value: str, min_threshold: int = 10) -> str:
        """
        数值特征ID化: field_id:value -> "field_value"

        Args:
            field_id: 特征字段ID (如1表示I1)
            value: 原始数值字符串
            min_threshold: 最小阈值，小于此值设为0

        Returns:
            ID化字符串，如 "1_100" 或 "1_0"
        """
        if not value or value.strip() == '':
            return f"{field_id}_0"

        try:
            val = int(value)
            # 应用阈值处理，但保留原始值用于ID化
            if val < min_threshold:
                return f"{field_id}_0"
            else:
                return f"{field_id}_{val}"
        except ValueError:
            return f"{field_id}_0"

    def categorical_to_id(self, field_id: int, value: str) -> str:
        """
        类别特征ID化: field_id:category -> "field_category"

        Args:
            field_id: 特征字段ID (如14表示C1)
            value: 原始类别字符串

        Returns:
            ID化字符串，如 "14_abc123" 或 "14_"
        """
        if not value or value.strip() == '':
            return f"{field_id}_"
        return f"{field_id}_{value}"

    def id_to_hash(self, feature_id: str) -> int:
        """
        ID字符串转换为hash值

        Args:
            feature_id: ID化的特征字符串

        Returns:
            哈希值 [0, hash_size)
        """
        return int(hashlib.md5(feature_id.encode()).hexdigest(), 16) % self.hash_size

    def process_criteo_line(self, fields: List[str], min_threshold: int = 10) -> Tuple[List[int], float]:
        """
        处理Criteo数据行，统一ID化所有特征

        Args:
            fields: 分割后的字段列表 [label, I1, I2, ..., I13, C1, C2, ..., C26]
            min_threshold: 数值特征最小阈值

        Returns:
            feature_hashes: 所有特征的hash值列表 (39个)
            label: 标签值
        """
        label = float(fields[0])
        feature_hashes = []

        # 处理数值特征 I1-I13 (field_id: 1-13)
        for i in range(1, 14):
            field_id = i
            value = fields[i] if i < len(fields) else ''
            feature_id = self.numerical_to_id(field_id, value, min_threshold)
            hash_val = self.id_to_hash(feature_id)
            feature_hashes.append(hash_val)

        # 处理类别特征 C1-C26 (field_id: 14-39)
        for i in range(14, 40):
            field_id = i
            value = fields[i] if i < len(fields) else ''
            feature_id = self.categorical_to_id(field_id, value)
            hash_val = self.id_to_hash(feature_id)
            feature_hashes.append(hash_val)

        return feature_hashes, label

    def process_avazu_line(self, fields: List[str]) -> Tuple[List[int], float]:
        """
        处理Avazu数据行，统一ID化所有特征

        Args:
            fields: 分割后的字段列表 [id, click, hour, C1, banner_pos, site_id, ...]

        Returns:
            feature_hashes: 所有特征的hash值列表 (22个)
            label: 标签值
        """
        # Parse label (click)
        label = float(fields[1]) if len(fields) > 1 else 0.0
        feature_hashes = []

        # 处理22个类别特征 (field_id: 0-21, 对应 fields[2] - fields[23])
        # Avazu所有特征都是类别特征
        for i in range(22):
            field_id = i
            field_idx = i + 2  # 跳过id和click字段
            value = fields[field_idx] if field_idx < len(fields) else ''
            feature_id = self.categorical_to_id(field_id, value)
            hash_val = self.id_to_hash(feature_id)
            feature_hashes.append(hash_val)

        return feature_hashes, label

    def process_taobaoads_line(self, fields: List[str]) -> Tuple[List[int], float]:
        """
        处理TaobaoAds数据行，统一ID化所有特征

        Args:
            fields: 分割后的字段列表 [clk, userid, time_stamp, adgroup_id, pid, ...]

        Returns:
            feature_hashes: 所有特征的hash值列表 (17个)
            label: 标签值
        """
        # Parse label (clk)
        label = float(fields[0]) if len(fields) > 0 else 0.0
        feature_hashes = []

        # 处理17个类别特征 (field_id: 0-16, 对应 fields[1] - fields[17])
        # TaobaoAds所有特征都作为类别特征处理
        for i in range(17):
            field_id = i
            field_idx = i + 1  # 跳过label字段
            value = fields[field_idx] if field_idx < len(fields) else ''
            feature_id = self.categorical_to_id(field_id, value)
            hash_val = self.id_to_hash(feature_id)
            feature_hashes.append(hash_val)

        return feature_hashes, label