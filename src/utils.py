# src/utils.py
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_pr_curve_plot(recall, precision, pr_auc,save_path: str):
    plt.figure()
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# [FINAL REPORT CHANGE] Added save_results_txt to write all test metrics to a .txt file
# so v1 and v2 results can be compared side-by-side without re-reading console logs.
def save_results_txt(metrics: dict, loss: float, pr_auc: float, save_path: str, extra_info: str = ""):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        if extra_info:
            f.write(extra_info + "\n")
            f.write("-" * 40 + "\n")
        f.write(f"Test Loss:      {loss:.4f}\n")
        f.write(f"Test IoU:       {metrics['iou']:.4f}\n")
        f.write(f"Test Dice:      {metrics['dice']:.4f}\n")
        f.write(f"Test Precision: {metrics['precision']:.4f}\n")
        f.write(f"Test Recall:    {metrics['recall']:.4f}\n")
        f.write(f"Test F-measure: {metrics['f_measure']:.4f}\n")
        f.write(f"PR AUC:         {pr_auc:.4f}\n")


def save_loss_curve_plot(train_losses, val_losses, save_path: str):
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
