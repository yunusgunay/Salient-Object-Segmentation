import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.dataset import get_image_mask_pairs, AugmentedECSSDDataset
from src.losses import BCEDiceLoss
from src.metrics import (
    binarize_predictions,
    compute_batch_metrics,
    initialize_metric_sums,
    update_metric_sums,
    average_metric_sums,
    compute_precision_recall_curve,
)
from src.models.clip import CLIPSegmenter
from src.utils import set_seed, ensure_dir, save_pr_curve_plot, save_loss_curve_plot, save_results_txt


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    metric_sums = initialize_metric_sums()

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)
        preds = binarize_predictions(logits, threshold=0.5)
        batch_metrics = compute_batch_metrics(preds, masks)

        total_loss += loss.item()
        update_metric_sums(metric_sums, batch_metrics)

    avg_loss = total_loss / len(loader)
    avg_metrics = average_metric_sums(metric_sums, len(loader))
    return avg_loss, avg_metrics


def main():
    SEED = 42
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 16
    NUM_EPOCHS = 60
    LEARNING_RATE = 3e-4

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    images_dir = "data/images"
    masks_dir = "data/ground_truth_mask"

    checkpoint_dir = "outputs_v2/checkpoints"
    prediction_dir = "outputs_v2/predictions"
    best_model_path = f"{checkpoint_dir}/clip_v2_best_model.pth"
    pr_curve_path = f"{prediction_dir}/clip_v2_pr_curve.png"
    loss_curve_path = f"{prediction_dir}/clip_v2_loss_curve.png"
    results_txt_path = f"{prediction_dir}/clip_v2_results.txt"

    set_seed(SEED)
    ensure_dir(checkpoint_dir)
    ensure_dir(prediction_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pairs = get_image_mask_pairs(images_dir, masks_dir)

    train_pairs, temp_pairs = train_test_split(pairs, test_size=0.30, random_state=SEED, shuffle=True)
    val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.50, random_state=SEED, shuffle=True)
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")

    train_dataset = AugmentedECSSDDataset(
        train_pairs,
        image_size=IMAGE_SIZE,
        augment=True,
        normalize_mean=CLIP_MEAN,
        normalize_std=CLIP_STD,
    )
    val_dataset = AugmentedECSSDDataset(
        val_pairs,
        image_size=IMAGE_SIZE,
        augment=False,
        normalize_mean=CLIP_MEAN,
        normalize_std=CLIP_STD,
    )
    test_dataset = AugmentedECSSDDataset(
        test_pairs,
        image_size=IMAGE_SIZE,
        augment=False,
        normalize_mean=CLIP_MEAN,
        normalize_std=CLIP_STD,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    model = CLIPSegmenter(freeze_encoder=True).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
    )

    best_val_dice = -1.0
    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate_one_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f} | Val Prec: {val_metrics['precision']:.4f} | "
            f"Val Rec: {val_metrics['recall']:.4f}"
        )

        scheduler.step(val_metrics["dice"])

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Saved best model (Val Dice: {best_val_dice:.4f})")

    save_loss_curve_plot(train_losses, val_losses, loss_curve_path)
    print(f"\nSaved loss curve to {loss_curve_path}")
    print("Training finished.")

    print("\nLoading best model for final test evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_metrics = evaluate_one_epoch(model, test_loader, criterion, device)
    precision, recall, _, pr_auc = compute_precision_recall_curve(model, test_loader, device)

    print("\nFinal Test Results (CLIP v2)")
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Test IoU:       {test_metrics['iou']:.4f}")
    print(f"Test Dice:      {test_metrics['dice']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F-measure: {test_metrics['f_measure']:.4f}")
    print(f"PR AUC:         {pr_auc:.4f}")

    save_pr_curve_plot(recall, precision, pr_auc, pr_curve_path)
    save_results_txt(
        test_metrics,
        test_loss,
        pr_auc,
        results_txt_path,
        extra_info="Model: CLIP v2 (BCEDice + augmentation + scheduler, frozen encoder)",
    )
    print(f"Saved PR curve to {pr_curve_path}")
    print(f"Saved results to {results_txt_path}")


if __name__ == "__main__":
    main()
