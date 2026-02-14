#!/usr/bin/env python3
"""
Main training script for CTR prediction models.

Usage:
    python scripts/train.py --config configs/dcnv2/dcnv2_16d_3l.yaml
    python scripts/train.py --config configs/baseline/embedding_mlp.yaml
"""

import argparse
import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DCNv2, EmbeddingMLP, Wukong, RankMixer
from models.mlcc import MLCCModel
from trainers import CTRTrainer
from utils import set_seed, load_config, create_data_loaders
from utils.model_analysis import print_model_analysis


def create_model(config: dict, device: torch.device) -> torch.nn.Module:
    """Create model based on configuration."""
    model_config = config['model']
    model_name = model_config['name'].lower()

    # Get dataset info for feature configuration
    if config['data']['dataset'].lower() == 'criteo':
        # Criteo: 39个特征全部统一ID化处理，都通过embedding
        num_features = 39
        categorical_features = list(range(0, 39))  # 所有特征都是embedding特征
        numerical_features = []                    # 不再有数值特征
    elif config['data']['dataset'].lower() == 'avazu':
        # Avazu: 22个特征全部统一ID化处理，都通过embedding
        num_features = 22
        categorical_features = list(range(0, 22))
        numerical_features = []
    elif config['data']['dataset'].lower() == 'taobaoads':
        # TaobaoAds: 17个特征全部统一ID化处理，都通过embedding
        num_features = 17
        categorical_features = list(range(0, 17))
        numerical_features = []
    else:
        raise ValueError(f"Unsupported dataset: {config['data']['dataset']}")

    # Create model based on type
    if model_name == 'dcnv2':
        model = DCNv2(
            num_features=num_features,
            embedding_dim=model_config['embedding_dim'],
            mix_hidden_dims=model_config['mix_hidden_dims'],
            deep_hidden_dims=model_config['deep_hidden_dims'],
            cross_layers=model_config['cross_layers'],
            cross_parameterization=model_config.get('cross_parameterization', 'vector'),
            deep_activation=model_config.get('deep_activation', 'relu'),
            dropout=model_config['dropout'],
            hash_size=model_config['hash_size']
        )
    elif model_name == 'embmlp':
        model = EmbeddingMLP(
            num_features=num_features,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            embedding_dim=model_config['embedding_dim'],
            hidden_dims=model_config['hidden_dims'],
            dropout=model_config['dropout'],
            hash_size=model_config['hash_size']
        )
    elif model_name == 'mlcc':
        model = MLCCModel(
            num_features=num_features,
            embedding_dim_total=model_config['embedding_dim_total'],
            embedding_dim_block=model_config['embedding_dim_block'],
            hidden_dims=model_config['deep_hidden_dims'],
            head_num=model_config['head_num'],
            dmlp_dim_list=model_config['dmlp_dim_list'],
            dmlp_dim_concat=model_config['dmlp_dim_concat'],
            emb_size2=model_config['emb_size2'],
            deep_activation=model_config.get('deep_activation', 'relu'),
            dropout=model_config['dropout'],
            hash_size=model_config['hash_size']
        )
    elif model_name == 'wukong':
        model = Wukong(
            num_features=num_features,
            embedding_dim=model_config['embedding_dim'],
            num_layers=model_config['num_layers'],
            num_emb_lcb=model_config['num_emb_lcb'],
            num_emb_fmb=model_config['num_emb_fmb'],
            rank_fmb=model_config['rank_fmb'],
            mlp_hidden_dims=model_config['hidden_dims'],
            dropout=model_config['dropout'],
            hash_size=model_config['hash_size']
        )
    elif model_name == 'rankmixer':
        model = RankMixer(
            num_features=num_features,
            embedding_dim=model_config['embedding_dim'],
            dim_multiplier=model_config['dim_multiplier'],
            num_layers=model_config['num_layers'],
            num_heads=model_config.get('num_heads'),  # None if not specified
            use_add_norm=model_config.get('use_add_norm', True),
            ffn_multiplier=model_config.get('ffn_multiplier', 4),
            mix_hidden_dims=model_config['mix_hidden_dims'],
            dropout=model_config['dropout'],
            hash_size=model_config['hash_size']
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description='Train CTR prediction models')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, auto)')

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")

    # Use output directories from config file
    print(f"Output directories:")
    print(f"  Model: {config['output']['model_dir']}")
    print(f"  Logs: {config['output']['log_dir']}")

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loaders = create_data_loaders(config)
    print(f"Train loader created")
    if val_loader is not None:
        print(f"Validation loader created")
    if test_loaders is not None:
        print(f"Test loaders created: {list(test_loaders.keys())}")

    # Create model
    print("Creating model...")
    model = create_model(config, device)
    print(f"Model created: {config['model']['name']}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Model analysis before training
    print("\n" + "="*50)
    print("模型结构分析")
    print("="*50)
    print_model_analysis(model, input_shape=(1, config['model'].get('num_features', 39)))

    # Create trainer
    trainer = CTRTrainer(model, device, config)

    # Start training
    print("Starting training...")
    trainer.train(train_loader, val_loader, test_loaders)

    # Model analysis after training (with checkpoint file size)
    print("\n" + "="*50)
    print("训练完成后模型分析")
    print("="*50)
    best_checkpoint = os.path.join(config['output']['model_dir'], 'best_checkpoint.pth')
    print_model_analysis(
        model,
        checkpoint_path=best_checkpoint if os.path.exists(best_checkpoint) else None,
        input_shape=(1, config['model'].get('num_features', 39))
    )

    print("Training completed successfully!")


if __name__ == '__main__':
    main()