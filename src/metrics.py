# src/metrics.py
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

EPS = 1e-7

def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / (denominator + EPS)


def binarize_predictions(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs > threshold).float()


def initialize_metric_sums():
    return {
        "iou": 0.0,
        "dice": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f_measure": 0.0,
    }


def compute_batch_metrics(preds: torch.Tensor, targets: torch.Tensor):
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    TP = (preds * targets).sum(dim=1)
    FP = (preds * (1 - targets)).sum(dim=1)
    FN = ((1 - preds) * targets).sum(dim=1)

    iou = _safe_divide(TP, TP + FP + FN)
    dice = _safe_divide(2 * TP, 2 * TP + FP + FN)
    precision = _safe_divide(TP, TP + FP)
    recall = _safe_divide(TP, TP + FN)
    f_measure = _safe_divide(2 * precision * recall, precision + recall)

    return {
        "iou": iou.mean().item(),
        "dice": dice.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f_measure": f_measure.mean().item(),
    }


def update_metric_sums(metric_sums: dict, batch_metrics: dict):
    for key in metric_sums:
        metric_sums[key] += batch_metrics[key]


def average_metric_sums(metric_sums: dict, num_batches: int) -> dict:
    return {key: value / num_batches for key, value in metric_sums.items()}


def compute_precision_recall_curve(model, loader, device):
    model.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy().ravel())
            all_targets.append(masks.cpu().numpy().ravel())
    
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    precision, recall, thresholds = precision_recall_curve(all_targets, all_probs)
    pr_auc = auc(recall, precision)

    return precision, recall, thresholds, pr_auc
