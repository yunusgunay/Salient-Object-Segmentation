import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.dataset import get_image_mask_pairs, ECSSDDataset
from src.models.clip import CLIPSegmenter
from src.models.dino import DINOSegmenter
from src.utils import set_seed, ensure_dir, save_pr_curve_plot, save_results_txt


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return numerator / (denominator + eps)


@torch.no_grad()
def collect_logits_and_targets(model, loader, device):
    model.eval()
    all_logits = []
    all_targets = []

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_targets.append(masks.cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return logits, targets


@torch.no_grad()
def compute_metrics_from_logits(logits, targets, threshold):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    tp = (preds_flat * targets_flat).sum(dim=1)
    fp = (preds_flat * (1 - targets_flat)).sum(dim=1)
    fn = ((1 - preds_flat) * targets_flat).sum(dim=1)

    iou = _safe_divide(tp, tp + fp + fn)
    dice = _safe_divide(2 * tp, 2 * tp + fp + fn)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f_measure = _safe_divide(2 * precision * recall, precision + recall)

    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

    return {
        "loss": bce.item(),
        "iou": iou.mean().item(),
        "dice": dice.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f_measure": f_measure.mean().item(),
    }


def search_best_fusion(val_logits_clip, val_logits_dino, val_targets):
    best = {
        "dice": -1.0,
        "weight_dino": None,
        "threshold": None,
    }

    # DINO is usually stronger on this dataset, so search is biased toward higher DINO weights.
    for weight_dino in np.linspace(0.50, 0.95, 10):
        fused_val_logits = weight_dino * val_logits_dino + (1.0 - weight_dino) * val_logits_clip

        for threshold in np.linspace(0.30, 0.70, 17):
            metrics = compute_metrics_from_logits(fused_val_logits, val_targets, threshold=float(threshold))
            if metrics["dice"] > best["dice"]:
                best = {
                    "dice": metrics["dice"],
                    "weight_dino": float(weight_dino),
                    "threshold": float(threshold),
                    "val_metrics": metrics,
                }

    return best


def compute_pr_auc_from_logits(logits, targets):
    probs = torch.sigmoid(logits).cpu().numpy().ravel()
    targets_np = targets.cpu().numpy().ravel()
    precision, recall, _ = precision_recall_curve(targets_np, probs)
    pr_auc = auc(recall, precision)
    return precision, recall, pr_auc


def main():
    SEED = 42
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 16

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    DINO_MEAN = (0.485, 0.456, 0.406)
    DINO_STD = (0.229, 0.224, 0.225)

    images_dir = "data/images"
    masks_dir = "data/ground_truth_mask"

    clip_checkpoint_path = "outputs_v2/checkpoints/clip_v2_best_model.pth"
    dino_checkpoint_path = "outputs_v2/checkpoints/dino_v2_best_model.pth"

    prediction_dir = "outputs_v2/predictions"
    pr_curve_path = f"{prediction_dir}/late_fusion_pr_curve.png"
    results_txt_path = f"{prediction_dir}/late_fusion_results.txt"

    set_seed(SEED)
    ensure_dir(prediction_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pairs = get_image_mask_pairs(images_dir, masks_dir)
    train_pairs, temp_pairs = train_test_split(pairs, test_size=0.30, random_state=SEED, shuffle=True)
    val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.50, random_state=SEED, shuffle=True)
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")

    val_clip_ds = ECSSDDataset(val_pairs, image_size=IMAGE_SIZE, normalize_mean=CLIP_MEAN, normalize_std=CLIP_STD)
    test_clip_ds = ECSSDDataset(test_pairs, image_size=IMAGE_SIZE, normalize_mean=CLIP_MEAN, normalize_std=CLIP_STD)

    val_dino_ds = ECSSDDataset(val_pairs, image_size=IMAGE_SIZE, normalize_mean=DINO_MEAN, normalize_std=DINO_STD)
    test_dino_ds = ECSSDDataset(test_pairs, image_size=IMAGE_SIZE, normalize_mean=DINO_MEAN, normalize_std=DINO_STD)

    val_clip_loader = DataLoader(val_clip_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    test_clip_loader = DataLoader(test_clip_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    val_dino_loader = DataLoader(val_dino_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    test_dino_loader = DataLoader(test_dino_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    clip_model = CLIPSegmenter(freeze_encoder=True).to(device)
    dino_model = DINOSegmenter(freeze_encoder=True).to(device)

    print(f"Loading CLIP checkpoint: {clip_checkpoint_path}")
    clip_model.load_state_dict(torch.load(clip_checkpoint_path, map_location=device))

    print(f"Loading DINO checkpoint: {dino_checkpoint_path}")
    dino_model.load_state_dict(torch.load(dino_checkpoint_path, map_location=device))

    print("Collecting validation logits...")
    val_logits_clip, val_targets_clip = collect_logits_and_targets(clip_model, val_clip_loader, device)
    val_logits_dino, val_targets_dino = collect_logits_and_targets(dino_model, val_dino_loader, device)

    if not torch.allclose(val_targets_clip, val_targets_dino):
        raise RuntimeError("Validation targets mismatch between CLIP and DINO loaders.")

    best = search_best_fusion(val_logits_clip, val_logits_dino, val_targets_clip)

    print("Best validation fusion settings found:")
    print(f"  weight_dino: {best['weight_dino']:.2f}")
    print(f"  threshold:   {best['threshold']:.2f}")
    print(f"  val_dice:    {best['dice']:.4f}")

    print("Collecting test logits...")
    test_logits_clip, test_targets_clip = collect_logits_and_targets(clip_model, test_clip_loader, device)
    test_logits_dino, test_targets_dino = collect_logits_and_targets(dino_model, test_dino_loader, device)

    if not torch.allclose(test_targets_clip, test_targets_dino):
        raise RuntimeError("Test targets mismatch between CLIP and DINO loaders.")

    fused_test_logits = best["weight_dino"] * test_logits_dino + (1.0 - best["weight_dino"]) * test_logits_clip

    test_metrics = compute_metrics_from_logits(
        fused_test_logits,
        test_targets_clip,
        threshold=best["threshold"],
    )

    precision, recall, pr_auc = compute_pr_auc_from_logits(fused_test_logits, test_targets_clip)

    print("\nFinal Test Results (Late Fusion)")
    print(f"Test Loss:      {test_metrics['loss']:.4f}")
    print(f"Test IoU:       {test_metrics['iou']:.4f}")
    print(f"Test Dice:      {test_metrics['dice']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F-measure: {test_metrics['f_measure']:.4f}")
    print(f"PR AUC:         {pr_auc:.4f}")

    save_pr_curve_plot(recall, precision, pr_auc, pr_curve_path)
    save_results_txt(
        {
            "iou": test_metrics["iou"],
            "dice": test_metrics["dice"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f_measure": test_metrics["f_measure"],
        },
        test_metrics["loss"],
        pr_auc,
        results_txt_path,
        extra_info=(
            "Model: CLIP+DINO Late Fusion (weighted logit average)\n"
            f"Selected on validation: weight_dino={best['weight_dino']:.2f}, "
            f"threshold={best['threshold']:.2f}, val_dice={best['dice']:.4f}"
        ),
    )

    print(f"Saved PR curve to {pr_curve_path}")
    print(f"Saved results to {results_txt_path}")


if __name__ == "__main__":
    main()
