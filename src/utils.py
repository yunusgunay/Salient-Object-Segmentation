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
