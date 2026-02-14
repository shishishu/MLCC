import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss
from typing import List, Tuple


def compute_auc(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Compute Area Under the ROC Curve (AUC).

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities

    Returns:
        AUC score
    """
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()

    try:
        return roc_auc_score(y_true_np, y_pred_np)
    except ValueError:
        # Handle cases where only one class is present
        return 0.5


def compute_logloss(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-15) -> float:
    """Compute logarithmic loss.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        eps: Small epsilon to avoid log(0)

    Returns:
        Log loss
    """
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()

    # Clip predictions to avoid log(0)
    y_pred_np = np.clip(y_pred_np, eps, 1 - eps)

    return log_loss(y_true_np, y_pred_np)


def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor,
                   metrics: List[str]) -> dict:
    """Compute multiple metrics.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        metrics: List of metric names to compute

    Returns:
        Dictionary of computed metrics
    """
    results = {}

    for metric in metrics:
        if metric.lower() == 'auc':
            results['auc'] = compute_auc(y_true, y_pred)
        elif metric.lower() == 'logloss':
            results['logloss'] = compute_logloss(y_true, y_pred)
        else:
            print(f"Warning: Unknown metric '{metric}'. Skipping.")

    return results