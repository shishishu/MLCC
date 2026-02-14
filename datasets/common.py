import torch
from torch.utils.data import IterableDataset
import gzip
import csv
from typing import Iterator, Tuple, List, Optional
from abc import ABC, abstractmethod


class BaseCTRDataset(IterableDataset, ABC):
    """Base class for CTR prediction datasets."""

    def __init__(self,
                 data_path: str,
                 num_features: int,
                 categorical_features: List[int],
                 numerical_features: List[int],
                 hash_size: int = 1000000,
                 delimiter: str = '\t',
                 skip_header: bool = False):
        """
        Args:
            data_path: Path to the dataset file
            num_features: Total number of features
            categorical_features: Indices of categorical features
            numerical_features: Indices of numerical features
            hash_size: Size of hash space for categorical features
            delimiter: Field delimiter in the data file
            skip_header: Whether to skip the first line (header)
        """
        self.data_path = data_path
        self.num_features = num_features
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.hash_size = hash_size
        self.delimiter = delimiter
        self.skip_header = skip_header

    @abstractmethod
    def parse_line(self, line: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parse a line from the dataset file.

        Returns:
            features: Feature tensor
            label: Label tensor
        """
        pass

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over the dataset (单进程优化版本)."""
        if self.data_path.endswith('.gz'):
            file_obj = gzip.open(self.data_path, 'rt', buffering=8192)
        else:
            file_obj = open(self.data_path, 'r', buffering=8192)

        try:
            # 跳过header行（如果需要）
            if self.skip_header:
                next(file_obj, None)

            # 优化的单进程数据加载
            for line in file_obj:
                line = line.strip()
                if line:
                    yield self.parse_line(line)
        finally:
            file_obj.close()

    def hash_feature(self, feature_value: str, feature_index: int) -> int:
        """Hash categorical feature to fixed size space."""
        return hash(f"{feature_index}_{feature_value}") % self.hash_size