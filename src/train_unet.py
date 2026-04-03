# src/train_unet.py
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import get_image_mask_pairs, ECSSDDataset
from metrics import binarize_predictions, compute_batch_metrics, initialize_metric_sums, update_metric_sums, average_metric_sums, compute_precision_recall_curve
from models.unet import UNet
from utils import set_seed, ensure_dir, save_pr_curve_plot


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

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
        images = images.to(device)
        masks = masks.to(device)

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
    # Configurations
    SEED = 42
    IMAGE_SIZE = (256, 256)
    BATCH_SIZE = 8
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3

    images_dir = "data/images"
    masks_dir = "data/ground_truth_mask"

    checkpoint_dir = "outputs/checkpoints"
    prediction_dir = "outputs/predictions"
    best_model_path = f"{checkpoint_dir}/unet_best_model.pth"
    pr_curve_path = f"{prediction_dir}/unet_pr_curve.png"

    # Setup
    set_seed(SEED)
    ensure_dir(checkpoint_dir)
    ensure_dir(prediction_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data pairs
    pairs = get_image_mask_pairs(images_dir, masks_dir)

    # Split data: 70% train, 15% val, 15% test
    train_pairs, temp_pairs = train_test_split(pairs, test_size=0.30, random_state=SEED, shuffle=True)
    val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.50, random_state=SEED, shuffle=True)
    print(f"Train samples: {len(train_pairs)}")
    print(f"Val samples: {len(val_pairs)}")
    print(f"Test samples: {len(test_pairs)}")

    # Datasets
    train_dataset = ECSSDDataset(train_pairs, image_size=IMAGE_SIZE)
    val_dataset = ECSSDDataset(val_pairs, image_size=IMAGE_SIZE)
    test_dataset = ECSSDDataset(test_pairs, image_size=IMAGE_SIZE)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Model, Loss, Optimizer
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_val_dice = -1.0

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate_one_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f} | "
            f"Val Precision: {val_metrics['precision']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f} | "
            f"Val F-measure: {val_metrics['f_measure']:.4f}"
        )

        # Save best model based on validation Dice score
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")
    
    print("\nTraining finished.")

    # Final test evaluation with best model
    print("\nLoading best model for final test evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_metrics = evaluate_one_epoch(model, test_loader, criterion, device)

    print("\nFinal Test Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test IoU: {test_metrics['iou']:.4f}")
    print(f"Test Dice: {test_metrics['dice']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F-measure: {test_metrics['f_measure']:.4f}")

    # Precision-Recall Curve
    precision, recall, thresholds, pr_auc = compute_precision_recall_curve(model, test_loader, device)
    save_pr_curve_plot(recall, precision, pr_auc, pr_curve_path)
    print(f"Saved Precision-Recall curve to {pr_curve_path}")
    print(f"PR AUC: {pr_auc:.4f}")


if __name__ == "__main__":
    main()
