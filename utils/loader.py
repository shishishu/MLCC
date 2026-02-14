import yaml
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple

from datasets import CriteoDataset, AvazuDataset, AliCCPDataset, TaobaoAdsDataset
from datasets.sharded_dataset import ShardedCriteoDataset, ShardedAvazuDataset, ShardedTaobaoAdsDataset


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dataset(dataset_name: str, data_path: str, hash_size: int, use_sharding: bool = True):
    """Create dataset based on name."""
    if dataset_name.lower() == 'criteo':
        if use_sharding:
            return ShardedCriteoDataset(data_path, hash_size=hash_size)
        else:
            return CriteoDataset(data_path, hash_size=hash_size)
    elif dataset_name.lower() == 'avazu':
        if use_sharding:
            return ShardedAvazuDataset(data_path, hash_size=hash_size)
        else:
            return AvazuDataset(data_path, hash_size=hash_size)
    elif dataset_name.lower() == 'taobaoads':
        if use_sharding:
            return ShardedTaobaoAdsDataset(data_path, hash_size=hash_size)
        else:
            return TaobaoAdsDataset(data_path, hash_size=hash_size)
    elif dataset_name.lower() == 'ali_ccp':
        return AliCCPDataset(data_path, hash_size=hash_size)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def create_data_loaders(config: Dict) -> Tuple[DataLoader, Optional[DataLoader], Optional[Dict[str, DataLoader]]]:
    """Create data loaders from configuration."""
    data_config = config['data']
    dataset_name = data_config['dataset']
    hash_size = config['model'].get('hash_size', 1000000)
    train_batch_size = data_config['batch_size']
    eval_batch_size = data_config.get('eval_batch_size', train_batch_size)  # 默认使用训练batch_size

    # 多进程配置
    use_sharding = data_config.get('use_sharding', True)
    num_workers = data_config.get('num_workers', 4)

    # Create train loader (多进程分片模式)
    train_dataset = create_dataset(dataset_name, data_config['train_path'], hash_size,
                                 use_sharding=use_sharding)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,  # IterableDataset doesn't support shuffle
        num_workers=num_workers if use_sharding else 0
    )

    # Create validation loader (多进程分片模式)
    val_loader = None
    if 'val_path' in data_config:
        val_dataset = create_dataset(dataset_name, data_config['val_path'], hash_size,
                                   use_sharding=use_sharding)
        val_loader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers if use_sharding else 0
        )

    # Create test loaders (多进程分片模式)
    test_loaders = None
    if 'test_paths' in data_config:
        test_loaders = {}
        for i, test_path in enumerate(data_config['test_paths']):
            test_dataset = create_dataset(dataset_name, test_path, hash_size,
                                        use_sharding=use_sharding)
            test_loader = DataLoader(
                test_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=num_workers if use_sharding else 0
            )
            test_loaders[f'test_{i}'] = test_loader

    return train_loader, val_loader, test_loaders