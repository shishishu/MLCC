import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
from typing import Dict, List, Optional
from tqdm import tqdm

from .metrics import compute_metrics


class CTRTrainer:
    """Trainer class for CTR prediction models."""

    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 config: Dict):
        """
        Args:
            model: PyTorch model to train
            device: Device to run training on
            config: Training configuration dictionary
        """
        self.model = model.to(device)
        self.device = device
        self.config = config

        # Training parameters
        self.epochs = config['training']['epochs']
        self.learning_rate = config['training']['learning_rate']
        self.weight_decay = config['training'].get('weight_decay', 0)

        # Setup optimizer
        optimizer_name = config['training'].get('optimizer', 'adam').lower()
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Setup scheduler
        scheduler_name = config['training'].get('scheduler', 'none').lower()
        if scheduler_name == 'step':
            step_size = config['training'].get('step_size', 3)
            gamma = config['training'].get('gamma', 0.8)
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs
            )
        else:
            self.scheduler = None

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Early stopping
        self.early_stopping = config['training'].get('early_stopping', False)
        self.patience = config['training'].get('patience', 5)
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Evaluation
        self.eval_metrics = config['evaluation']['metrics']
        self.eval_interval = config['evaluation'].get('eval_interval', 1000)
        self.train_metric_interval = config['training'].get('train_metric_interval', 100)

        # Output directories
        self.model_dir = config['output']['model_dir']
        self.log_dir = config['output']['log_dir']
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Tensorboard writer
        self.writer = SummaryWriter(self.log_dir)

        # Training state
        self.global_step = 0
        self.best_metrics = {}

        # Training log for step-level metrics
        self.training_log = []

        # Test results log
        self.test_results = []

    def _estimate_batches_per_epoch(self, train_loader: DataLoader) -> int:
        """Estimate number of batches per epoch by counting lines in data file."""
        try:
            # Get dataset from loader
            dataset = train_loader.dataset
            if hasattr(dataset, 'data_path'):
                # Count lines in data file
                with open(dataset.data_path, 'r') as f:
                    line_count = sum(1 for _ in f)

                batch_size = train_loader.batch_size
                estimated_batches = (line_count + batch_size - 1) // batch_size
                print(f"Estimated batches per epoch: {estimated_batches} (lines: {line_count}, batch_size: {batch_size})")
                return estimated_batches
            else:
                print("Warning: Cannot estimate batches per epoch, using default 100")
                return 100
        except Exception as e:
            print(f"Warning: Error estimating batches per epoch: {e}, using default 100")
            return 100

    def train_epoch(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # For IterableDataset, estimate batches per epoch and use islice to limit
        estimated_batches = self._estimate_batches_per_epoch(train_loader)

        from itertools import islice
        limited_loader = islice(train_loader, estimated_batches)
        progress_bar = tqdm(limited_loader, desc="Training", total=estimated_batches)

        for batch_idx, (features, labels) in enumerate(progress_bar):
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.criterion(logits.squeeze(), labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

            # Step-level evaluation and logging
            if val_loader is not None and self.global_step % self.train_metric_interval == 0:
                # Save model state for restoration
                self.model.eval()

                val_metrics = self.evaluate(val_loader, 'val')
                val_loss = val_metrics.get('loss', 0)
                val_auc = val_metrics.get('auc', 0)

                # Log to tensorboard
                self.writer.add_scalar('train_step/val_loss', val_loss, self.global_step)
                self.writer.add_scalar('train_step/val_auc', val_auc, self.global_step)

                # Print step-level metrics with sample counts
                val_batches = val_metrics.get('num_batches', 0)
                val_samples = val_metrics.get('num_samples', 0)
                print(f"\nStep {self.global_step}: train_logloss={loss.item():.4f}, val_logloss={val_loss:.4f}, val_auc={val_auc:.4f} (val: {val_batches} batches, {val_samples} samples)")

                # Save to training log
                log_entry = {
                    'step': self.global_step,
                    'train_logloss': loss.item(),
                    'val_logloss': val_loss,
                    'val_auc': val_auc
                }
                self.training_log.append(log_entry)

                # Save metrics immediately after each validation
                self.save_training_metrics()

                # Restore training mode
                self.model.train()

            # Log to tensorboard
            if self.global_step % 100 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)

        avg_loss = total_loss / num_batches

        # Check if the last step needs validation evaluation
        if val_loader is not None and self.global_step % self.train_metric_interval != 0:
            val_metrics = self.evaluate(val_loader, 'val')
            val_loss = val_metrics.get('loss', 0)
            val_auc = val_metrics.get('auc', 0)
            val_batches = val_metrics.get('num_batches', 0)
            val_samples = val_metrics.get('num_samples', 0)
            print(f"\nStep {self.global_step}: train_logloss={avg_loss:.4f}, val_logloss={val_loss:.4f}, val_auc={val_auc:.4f} (val: {val_batches} batches, {val_samples} samples)")

            # Save to training log
            log_entry = {
                'step': self.global_step,
                'train_logloss': avg_loss,
                'val_logloss': val_loss,
                'val_auc': val_auc
            }
            self.training_log.append(log_entry)

            # Save metrics immediately after each validation
            self.save_training_metrics()

            # Restore training mode
            self.model.train()
        return {'loss': avg_loss}

    def evaluate(self, eval_loader: DataLoader, prefix: str = 'val') -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        num_samples = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for features, labels in tqdm(eval_loader, desc=f"Evaluating {prefix}", leave=False):
                features = features.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(features)
                loss = self.criterion(logits.squeeze(), labels)

                total_loss += loss.item()
                num_batches += 1
                num_samples += len(labels)

                # Convert logits to probabilities
                probs = torch.sigmoid(logits.squeeze())
                all_labels.append(labels)
                all_preds.append(probs)

        # Concatenate all predictions and labels
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)

        # Compute metrics
        avg_loss = total_loss / num_batches
        metrics = compute_metrics(all_labels, all_preds, self.eval_metrics)
        metrics['loss'] = avg_loss
        metrics['num_batches'] = num_batches
        metrics['num_samples'] = num_samples

        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'global_step': self.global_step
        }

        # Save latest checkpoint
        if self.config['output'].get('save_last', True):
            torch.save(checkpoint, os.path.join(self.model_dir, 'last_checkpoint.pth'))

        # Save best checkpoint
        if is_best and self.config['output'].get('save_best', True):
            torch.save(checkpoint, os.path.join(self.model_dir, 'best_checkpoint.pth'))

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              test_loaders: Optional[Dict[str, DataLoader]] = None):
        """Main training loop."""
        print(f"Starting training for {self.epochs} epochs...")

        # Check if we should evaluate test set per epoch
        eval_per_epoch = self.config['training'].get('eval_per_epoch', False)

        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader, val_loader)
            print(f"Train LOGLOSS: {train_metrics['loss']:.4f}")

            # Log training metrics
            for metric, value in train_metrics.items():
                self.writer.add_scalar(f'train/{metric}', value, epoch)

            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, 'val')
                # Log validation metrics
                for metric, value in val_metrics.items():
                    self.writer.add_scalar(f'val/{metric}', value, epoch)

                # Only print AUC and LOGLOSS with sample counts
                val_batches = val_metrics.get('num_batches', 0)
                val_samples = val_metrics.get('num_samples', 0)
                print(f"Val AUC: {val_metrics.get('auc', 0):.4f}")
                print(f"Val LOGLOSS: {val_metrics.get('logloss', 0):.4f}")
                print(f"Val Data: {val_batches} batches, {val_samples} samples")

                # Check for best model
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                    self.best_metrics = val_metrics.copy()
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                # Save checkpoint
                self.save_checkpoint(epoch, val_metrics, is_best)

                # Early stopping
                if self.early_stopping and self.patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break

            # Evaluate on test sets per epoch if enabled
            if eval_per_epoch and test_loaders is not None:
                print(f"\nEvaluating on test sets (Epoch {epoch})...")
                for test_name, test_loader in test_loaders.items():
                    test_metrics = self.evaluate(test_loader, f'test_{test_name}')
                    eval_logloss = test_metrics.get('loss', 0)
                    eval_auc = test_metrics.get('auc', 0)

                    print(f"  Test {test_name} - AUC: {eval_auc:.4f}, LOGLOSS: {eval_logloss:.4f}")

                    # Save test results with epoch number
                    test_result = {
                        'epoch_num': epoch,
                        'test_name': test_name,
                        'eval_logloss': eval_logloss,
                        'eval_auc': eval_auc
                    }
                    self.test_results.append(test_result)

                # Save test results immediately after each epoch evaluation
                self.save_test_results()

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

        # Evaluate on test sets (final evaluation if not done per epoch)
        if not eval_per_epoch and test_loaders is not None:
            print("\nEvaluating on test sets...")
            for test_name, test_loader in test_loaders.items():
                test_metrics = self.evaluate(test_loader, f'test_{test_name}')
                eval_logloss = test_metrics.get('loss', 0)
                eval_auc = test_metrics.get('auc', 0)

                print(f"\nTest {test_name} results:")
                # Print eval_logloss and eval_auc with sample counts
                test_batches = test_metrics.get('num_batches', 0)
                test_samples = test_metrics.get('num_samples', 0)
                print(f"  eval_logloss: {eval_logloss:.4f}")
                print(f"  eval_auc: {eval_auc:.4f}")
                print(f"  test_data: {test_batches} batches, {test_samples} samples")

                # Save test results (without epoch number for backward compatibility)
                test_result = {
                    'test_name': test_name,
                    'eval_logloss': eval_logloss,
                    'eval_auc': eval_auc
                }
                self.test_results.append(test_result)

        # Save training metrics to CSV
        self.save_training_metrics()

        # Save test results to CSV
        self.save_test_results()

        self.writer.close()
        print("Training completed!")

    def save_training_metrics(self):
        """Save training metrics to CSV file."""
        if not self.training_log:
            return

        import pandas as pd
        df = pd.DataFrame(self.training_log)

        # Format numerical columns to 4 decimal places
        for col in ['train_logloss', 'val_logloss', 'val_auc']:
            if col in df.columns:
                df[col] = df[col].round(4)

        log_file = os.path.join(self.log_dir, 'training_metrics.csv')
        df.to_csv(log_file, index=False, float_format='%.4f')
        print(f"Training metrics saved to: {log_file}")

    def save_test_results(self):
        """Save test results to CSV file."""
        if not self.test_results:
            return

        import pandas as pd
        df = pd.DataFrame(self.test_results)

        # Reorder columns if epoch_num exists
        if 'epoch_num' in df.columns:
            cols = ['epoch_num', 'test_name', 'eval_logloss', 'eval_auc']
            df = df[cols]

        # Format numerical columns to 4 decimal places
        for col in ['eval_logloss', 'eval_auc']:
            if col in df.columns:
                df[col] = df[col].round(4)

        log_file = os.path.join(self.log_dir, 'test_results.csv')
        df.to_csv(log_file, index=False, float_format='%.4f')
        print(f"Test results saved to: {log_file}")