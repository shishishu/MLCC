import torch
import numpy as np
from torch.utils.data import IterableDataset
from typing import Iterator, Tuple, List
import gzip
import os


class ShardedCriteoDataset(IterableDataset):
    """基于文件分片的Criteo数据集，解决多进程数据重复问题"""

    def __init__(self,
                 data_path: str,
                 hash_size: int = 1000000,
                 min_threshold: int = 10,
                 delimiter: str = '\t'):
        self.data_path = data_path
        self.hash_size = hash_size
        self.min_threshold = min_threshold
        self.delimiter = delimiter

        # 导入特征处理器
        from .feature_processor import FeatureIDProcessor
        self.feature_processor = FeatureIDProcessor(hash_size)

        # 预计算文件信息
        self._file_info = self._get_file_info()

    def _get_file_info(self):
        """获取文件信息：行数和字节偏移量"""
        line_offsets = [0]  # 每行的字节偏移量

        if self.data_path.endswith('.gz'):
            # 对于gzip文件，需要解压后计算
            with gzip.open(self.data_path, 'rt') as f:
                while f.readline():
                    # gzip不支持精确的字节偏移，所以用行号
                    line_offsets.append(len(line_offsets))
        else:
            with open(self.data_path, 'rb') as f:
                while f.readline():
                    line_offsets.append(f.tell())

        return {
            'total_lines': len(line_offsets) - 1,
            'line_offsets': line_offsets[:-1]  # 去掉最后一个偏移量
        }

    def _get_worker_slice(self, worker_id: int, num_workers: int):
        """为每个worker分配不重复的数据片段"""
        total_lines = self._file_info['total_lines']
        lines_per_worker = total_lines // num_workers

        start_line = worker_id * lines_per_worker
        if worker_id == num_workers - 1:
            # 最后一个worker处理剩余的所有行
            end_line = total_lines
        else:
            end_line = start_line + lines_per_worker

        return start_line, end_line

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """多进程安全的迭代器"""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # 单进程模式：处理所有数据
            start_line, end_line = 0, self._file_info['total_lines']
            worker_id = 0
        else:
            # 多进程模式：每个worker处理自己的分片
            start_line, end_line = self._get_worker_slice(
                worker_info.id, worker_info.num_workers
            )
            worker_id = worker_info.id

        # 根据文件类型选择读取方式
        if self.data_path.endswith('.gz'):
            yield from self._iter_gzip_lines(start_line, end_line, worker_id)
        else:
            yield from self._iter_text_lines(start_line, end_line, worker_id)

    def _iter_text_lines(self, start_line: int, end_line: int, worker_id: int):
        """处理普通文本文件"""
        with open(self.data_path, 'r', buffering=8192) as f:
            # 跳到起始位置
            if start_line > 0:
                for _ in range(start_line):
                    f.readline()

            # 读取分配的行数
            for line_idx in range(start_line, end_line):
                line = f.readline()
                if not line:
                    break

                line = line.strip()
                if line:
                    try:
                        features, label = self.parse_line(line)
                        yield features, label
                    except Exception as e:
                        print(f"Worker {worker_id} parse error at line {line_idx}: {e}")
                        continue

    def _iter_gzip_lines(self, start_line: int, end_line: int, worker_id: int):
        """处理gzip压缩文件"""
        with gzip.open(self.data_path, 'rt', buffering=8192) as f:
            # 跳到起始位置
            current_line = 0
            while current_line < start_line:
                f.readline()
                current_line += 1

            # 读取分配的行数
            while current_line < end_line:
                line = f.readline()
                if not line:
                    break

                line = line.strip()
                if line:
                    try:
                        features, label = self.parse_line(line)
                        yield features, label
                    except Exception as e:
                        print(f"Worker {worker_id} parse error at line {current_line}: {e}")

                current_line += 1

    def parse_line(self, line: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """解析Criteo数据行"""
        # 优化：限制分割数量
        fields = line.split(self.delimiter, 39)

        # 使用特征处理器
        feature_hashes, label = self.feature_processor.process_criteo_line(
            fields, self.min_threshold
        )

        # 优化：直接从numpy数组创建tensor
        features = torch.from_numpy(np.array(feature_hashes, dtype=np.int64))
        label = torch.tensor(label, dtype=torch.float32)

        return features, label

    def get_data_info(self):
        """返回数据集信息"""
        return {
            'total_lines': self._file_info['total_lines'],
            'data_path': self.data_path
        }


def test_sharded_dataset():
    """测试分片数据集的正确性"""
    import tempfile
    import os

    # 创建测试数据
    test_data = [
        "1\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12\t13\ta\tb\tc\td\te\tf\tg\th\ti\tj\tk\tl\tm\tn\to\tp\tq\tr\ts\tt\tu\tv\tw\tx\ty\tz",
        "0\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12\t13\t14\tb\tc\td\te\tf\tg\th\ti\tj\tk\tl\tm\tn\to\tp\tq\tr\ts\tt\tu\tv\tw\tx\ty\tz\ta",
        "1\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12\t13\t14\t15\tc\td\te\tf\tg\th\ti\tj\tk\tl\tm\tn\to\tp\tq\tr\ts\tt\tu\tv\tw\tx\ty\tz\ta\tb",
        "0\t4\t5\t6\t7\t8\t9\t10\t11\t12\t13\t14\t15\t16\td\te\tf\tg\th\ti\tj\tk\tl\tm\tn\to\tp\tq\tr\ts\tt\tu\tv\tw\tx\ty\tz\ta\tb\tc"
    ]

    # 写入临时文件
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        for line in test_data:
            f.write(line + '\n')
        temp_path = f.name

    try:
        # 测试数据集
        dataset = ShardedCriteoDataset(temp_path)

        print(f"Dataset info: {dataset.get_data_info()}")

        # 测试单进程迭代
        print("Testing single process iteration:")
        count = 0
        for features, label in dataset:
            print(f"Sample {count}: features shape={features.shape}, label={label}")
            count += 1
            if count >= 2:  # 只测试前两个样本
                break

        print("Test completed successfully!")

    finally:
        # 清理临时文件
        os.unlink(temp_path)


class ShardedAvazuDataset(IterableDataset):
    """基于文件分片的Avazu数据集，解决多进程数据重复问题"""

    def __init__(self,
                 data_path: str,
                 hash_size: int = 1000000,
                 delimiter: str = ','):
        self.data_path = data_path
        self.hash_size = hash_size
        self.delimiter = delimiter

        # 导入特征处理器
        from .feature_processor import FeatureIDProcessor
        self.feature_processor = FeatureIDProcessor(hash_size)

        # 预计算文件信息
        self._file_info = self._get_file_info()

    def _get_file_info(self):
        """获取文件信息：行数和字节偏移量"""
        line_offsets = [0]  # 每行的字节偏移量

        if self.data_path.endswith('.gz'):
            # 对于gzip文件，需要解压后计算
            with gzip.open(self.data_path, 'rt') as f:
                while f.readline():
                    # gzip不支持精确的字节偏移，所以用行号
                    line_offsets.append(len(line_offsets))
        else:
            with open(self.data_path, 'rb') as f:
                while f.readline():
                    line_offsets.append(f.tell())

        return {
            'total_lines': len(line_offsets) - 1,
            'line_offsets': line_offsets[:-1]  # 去掉最后一个偏移量
        }

    def _get_worker_slice(self, worker_id: int, num_workers: int):
        """为每个worker分配不重复的数据片段"""
        total_lines = self._file_info['total_lines']
        lines_per_worker = total_lines // num_workers

        start_line = worker_id * lines_per_worker
        if worker_id == num_workers - 1:
            # 最后一个worker处理剩余的所有行
            end_line = total_lines
        else:
            end_line = start_line + lines_per_worker

        return start_line, end_line

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """多进程安全的迭代器"""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # 单进程模式：处理所有数据
            start_line, end_line = 0, self._file_info['total_lines']
            worker_id = 0
        else:
            # 多进程模式：每个worker处理自己的分片
            start_line, end_line = self._get_worker_slice(
                worker_info.id, worker_info.num_workers
            )
            worker_id = worker_info.id

        # 根据文件类型选择读取方式
        if self.data_path.endswith('.gz'):
            yield from self._iter_gzip_lines(start_line, end_line, worker_id)
        else:
            yield from self._iter_text_lines(start_line, end_line, worker_id)

    def _iter_text_lines(self, start_line: int, end_line: int, worker_id: int):
        """处理普通文本文件"""
        with open(self.data_path, 'r', buffering=8192) as f:
            # 跳到起始位置
            if start_line > 0:
                for _ in range(start_line):
                    f.readline()

            # 读取分配的行数
            for line_idx in range(start_line, end_line):
                line = f.readline()
                if not line:
                    break

                line = line.strip()
                if line:
                    try:
                        features, label = self.parse_line(line)
                        yield features, label
                    except Exception as e:
                        print(f"Worker {worker_id} parse error at line {line_idx}: {e}")
                        continue

    def _iter_gzip_lines(self, start_line: int, end_line: int, worker_id: int):
        """处理gzip压缩文件"""
        with gzip.open(self.data_path, 'rt', buffering=8192) as f:
            # 跳到起始位置
            current_line = 0
            while current_line < start_line:
                f.readline()
                current_line += 1

            # 读取分配的行数
            while current_line < end_line:
                line = f.readline()
                if not line:
                    break

                line = line.strip()
                if line:
                    try:
                        features, label = self.parse_line(line)
                        yield features, label
                    except Exception as e:
                        print(f"Worker {worker_id} parse error at line {current_line}: {e}")

                current_line += 1

    def parse_line(self, line: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """解析Avazu数据行"""
        # 优化：限制分割数量
        fields = line.split(self.delimiter, 24)

        # 使用特征处理器
        feature_hashes, label = self.feature_processor.process_avazu_line(fields)

        # 优化：直接从numpy数组创建tensor
        features = torch.from_numpy(np.array(feature_hashes, dtype=np.int64))
        label = torch.tensor(label, dtype=torch.float32)

        return features, label

    def get_data_info(self):
        """返回数据集信息"""
        return {
            'total_lines': self._file_info['total_lines'],
            'data_path': self.data_path
        }


class ShardedTaobaoAdsDataset(IterableDataset):
    """基于文件分片的TaobaoAds数据集，解决多进程数据重复问题"""

    def __init__(self,
                 data_path: str,
                 hash_size: int = 1000000,
                 delimiter: str = ','):
        self.data_path = data_path
        self.hash_size = hash_size
        self.delimiter = delimiter

        # 导入特征处理器
        from .feature_processor import FeatureIDProcessor
        self.feature_processor = FeatureIDProcessor(hash_size)

        # 预计算文件信息
        self._file_info = self._get_file_info()

    def _get_file_info(self):
        """获取文件信息：行数和字节偏移量"""
        line_offsets = [0]  # 每行的字节偏移量

        if self.data_path.endswith('.gz'):
            # 对于gzip文件，需要解压后计算
            with gzip.open(self.data_path, 'rt') as f:
                while f.readline():
                    # gzip不支持精确的字节偏移，所以用行号
                    line_offsets.append(len(line_offsets))
        else:
            with open(self.data_path, 'rb') as f:
                while f.readline():
                    line_offsets.append(f.tell())

        return {
            'total_lines': len(line_offsets) - 1,
            'line_offsets': line_offsets[:-1]  # 去掉最后一个偏移量
        }

    def _get_worker_slice(self, worker_id: int, num_workers: int):
        """为每个worker分配不重复的数据片段"""
        total_lines = self._file_info['total_lines']
        lines_per_worker = total_lines // num_workers

        start_line = worker_id * lines_per_worker
        if worker_id == num_workers - 1:
            # 最后一个worker处理剩余的所有行
            end_line = total_lines
        else:
            end_line = start_line + lines_per_worker

        return start_line, end_line

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """多进程安全的迭代器"""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # 单进程模式：处理所有数据
            start_line, end_line = 0, self._file_info['total_lines']
            worker_id = 0
        else:
            # 多进程模式：每个worker处理自己的分片
            start_line, end_line = self._get_worker_slice(
                worker_info.id, worker_info.num_workers
            )
            worker_id = worker_info.id

        # 根据文件类型选择读取方式
        if self.data_path.endswith('.gz'):
            yield from self._iter_gzip_lines(start_line, end_line, worker_id)
        else:
            yield from self._iter_text_lines(start_line, end_line, worker_id)

    def _iter_text_lines(self, start_line: int, end_line: int, worker_id: int):
        """处理普通文本文件"""
        with open(self.data_path, 'r', buffering=8192) as f:
            # 跳到起始位置
            if start_line > 0:
                for _ in range(start_line):
                    f.readline()

            # 读取分配的行数
            for line_idx in range(start_line, end_line):
                line = f.readline()
                if not line:
                    break

                line = line.strip()
                if line:
                    try:
                        features, label = self.parse_line(line)
                        yield features, label
                    except Exception as e:
                        print(f"Worker {worker_id} parse error at line {line_idx}: {e}")
                        continue

    def _iter_gzip_lines(self, start_line: int, end_line: int, worker_id: int):
        """处理gzip压缩文件"""
        with gzip.open(self.data_path, 'rt', buffering=8192) as f:
            # 跳到起始位置
            current_line = 0
            while current_line < start_line:
                f.readline()
                current_line += 1

            # 读取分配的行数
            while current_line < end_line:
                line = f.readline()
                if not line:
                    break

                line = line.strip()
                if line:
                    try:
                        features, label = self.parse_line(line)
                        yield features, label
                    except Exception as e:
                        print(f"Worker {worker_id} parse error at line {current_line}: {e}")

                current_line += 1

    def parse_line(self, line: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """解析TaobaoAds数据行"""
        # 优化：限制分割数量
        fields = line.split(self.delimiter, 18)  # 1 label + 17 features

        # 使用特征处理器
        feature_hashes, label = self.feature_processor.process_taobaoads_line(fields)

        # 优化：直接从numpy数组创建tensor
        features = torch.from_numpy(np.array(feature_hashes, dtype=np.int64))
        label = torch.tensor(label, dtype=torch.float32)

        return features, label

    def get_data_info(self):
        """返回数据集信息"""
        return {
            'total_lines': self._file_info['total_lines'],
            'data_path': self.data_path
        }


if __name__ == "__main__":
    test_sharded_dataset()