#!/usr/bin/env python3
"""
Evaluation script for CTR prediction models.

Usage:
    python scripts/evaluate.py --config configs/dcnv2/dcnv2_16d_3l.yaml --checkpoint outputs/checkpoints/dcnv2_16d_3l/best_checkpoint.pth
"""

import argparse
import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DCNv2, EmbeddingMLP
from trainers import CTRTrainer
from utils import set_seed, load_config, create_data_loaders


def load_model_from_checkpoint(config: dict, checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    # Create model (same as in train.py)
    model_config = config['model']
    model_name = model_config['name'].lower()

    if config['data']['dataset'].lower() == 'criteo':
        # Criteo: 39个特征全部统一ID化处理，都通过embedding
        num_features = 39
        categorical_features = list(range(0, 39))  # 所有特征都是embedding特征
        numerical_features = []                    # 不再有数值特征
    elif config['data']['dataset'].lower() == 'avazu':
        num_features = 22
        categorical_features = list(range(0, 22))
        numerical_features = []
    else:
        raise ValueError(f"Unsupported dataset: {config['data']['dataset']}")

    if model_name == 'dcnv2':
        model = DCNv2(
            num_features=num_features,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            embedding_dim=model_config['embedding_dim'],
            cross_layers=model_config['cross_layers'],
            deep_layers=model_config['deep_layers'],
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
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Filter out unnecessary keys from state_dict (like thop profiling keys)
    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items()
                          if not k.endswith('total_ops') and not k.endswith('total_params')}

    model.load_state_dict(filtered_state_dict)
    model.to(device)

    return model


def main():
    parser = argparse.ArgumentParser(description='Evaluate CTR prediction models')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
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

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loaders = create_data_loaders(config)

    # Load model from checkpoint
    print(f"Loading model from {args.checkpoint}")
    model = load_model_from_checkpoint(config, args.checkpoint, device)

    # Create trainer for evaluation
    trainer = CTRTrainer(model, device, config)

    # Evaluate on validation set
    if val_loader is not None:
        print("\nEvaluating on validation set...")
        val_metrics = trainer.evaluate(val_loader, 'val')
        print("Validation Results:")
        print(f"  AUC: {val_metrics.get('auc', 0):.4f}")
        print(f"  LOGLOSS: {val_metrics.get('logloss', 0):.4f}")

    # Evaluate on test sets
    if test_loaders is not None:
        print("\nEvaluating on test sets...")
        for test_name, test_loader in test_loaders.items():
            test_metrics = trainer.evaluate(test_loader, test_name)
            print(f"\n{test_name.capitalize()} Results:")
            print(f"  AUC: {test_metrics.get('auc', 0):.4f}")
            print(f"  LOGLOSS: {test_metrics.get('logloss', 0):.4f}")

    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()