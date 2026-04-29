# src/train_cnn_optuna.py
import optuna
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.dataset import get_image_mask_pairs, AugmentedECSSDDataset
from src.losses import BCEDiceLoss
from src.metrics import (
    binarize_predictions, compute_batch_metrics,
    initialize_metric_sums, update_metric_sums,
    average_metric_sums, compute_precision_recall_curve,
)
from src.models.cnn_improved import CNNWithSkips
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
def evaluate_one_epoch(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    metric_sums = initialize_metric_sums()

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        preds = binarize_predictions(logits, threshold=threshold)
        batch_metrics = compute_batch_metrics(preds, masks)

        total_loss += loss.item()
        update_metric_sums(metric_sums, batch_metrics)

    avg_loss = total_loss / len(loader)
    avg_metrics = average_metric_sums(metric_sums, len(loader))
    return avg_loss, avg_metrics


def make_dataloaders(batch_size, image_size=(256, 256)):
    SEED = 42

    images_dir = "data/images"
    masks_dir = "data/ground_truth_mask"

    pairs = get_image_mask_pairs(images_dir, masks_dir)

    train_pairs, temp_pairs = train_test_split(
        pairs,
        test_size=0.30,
        random_state=SEED,
        shuffle=True,
    )

    val_pairs, test_pairs = train_test_split(
        temp_pairs,
        test_size=0.50,
        random_state=SEED,
        shuffle=True,
    )

    train_dataset = AugmentedECSSDDataset(
        train_pairs,
        image_size=image_size,
        augment=True,
    )

    val_dataset = AugmentedECSSDDataset(
        val_pairs,
        image_size=image_size,
        augment=False,
    )

    test_dataset = AugmentedECSSDDataset(
        test_pairs,
        image_size=image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


def objective(trial):
    SEED = 42
    NUM_EPOCHS = 40

    set_seed(SEED)

    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16])
    bce_weight = trial.suggest_float("bce_weight", 0.3, 0.8)
    pos_weight_value = trial.suggest_float("pos_weight", 1.0, 4.0)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    threshold = trial.suggest_float("threshold", 0.35, 0.65)

    scheduler_factor = trial.suggest_categorical(
        "scheduler_factor",
        [0.3, 0.5, 0.7],
    )

    scheduler_patience = trial.suggest_categorical(
        "scheduler_patience",
        [5, 7, 10],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = make_dataloaders(
        batch_size=batch_size,
        image_size=(256, 256),
    )

    model = CNNWithSkips().to(device)

    pos_weight = torch.tensor([pos_weight_value], device=device)

    criterion = BCEDiceLoss(
        bce_weight=bce_weight,
        pos_weight=pos_weight,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=1e-5,
    )

    best_val_dice = -1.0

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss, val_metrics = evaluate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            threshold=threshold,
        )

        val_dice = val_metrics["dice"]
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Trial {trial.number} | Epoch [{epoch + 1}/{NUM_EPOCHS}] | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val Precision: {val_metrics['precision']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f}"
        )

        scheduler.step(val_dice)

        if val_dice > best_val_dice:
            best_val_dice = val_dice

        trial.report(best_val_dice, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_dice


def train_final_model(best_params):
    SEED = 42
    NUM_EPOCHS = 100

    set_seed(SEED)

    checkpoint_dir = "outputs_optuna_cnn/checkpoints"
    prediction_dir = "outputs_optuna_cnn/predictions"

    best_model_path = f"{checkpoint_dir}/cnn_optuna_best_model.pth"
    pr_curve_path = f"{prediction_dir}/cnn_optuna_pr_curve.png"
    loss_curve_path = f"{prediction_dir}/cnn_optuna_loss_curve.png"
    results_txt_path = f"{prediction_dir}/cnn_optuna_results.txt"

    ensure_dir(checkpoint_dir)
    ensure_dir(prediction_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    train_loader, val_loader, test_loader = make_dataloaders(
        batch_size=best_params["batch_size"],
        image_size=(256, 256),
    )

    model = CNNWithSkips().to(device)

    pos_weight = torch.tensor(
        [best_params["pos_weight"]],
        device=device,
    )

    criterion = BCEDiceLoss(
        bce_weight=best_params["bce_weight"],
        pos_weight=pos_weight,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=best_params["scheduler_factor"],
        patience=best_params["scheduler_patience"],
        min_lr=1e-5,
    )

    threshold = best_params["threshold"]

    best_val_dice = -1.0
    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss, val_metrics = evaluate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            threshold=threshold,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f} | "
            f"Val Precision: {val_metrics['precision']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f}"
        )

        scheduler.step(val_metrics["dice"])

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Saved best model. Val Dice: {best_val_dice:.4f}")

    save_loss_curve_plot(train_losses, val_losses, loss_curve_path)
    print(f"\nSaved loss curve to {loss_curve_path}")

    print("\nLoading best model for final test evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_metrics = evaluate_one_epoch(
        model,
        test_loader,
        criterion,
        device,
        threshold=threshold,
    )

    precision, recall, _, pr_auc = compute_precision_recall_curve(
        model,
        test_loader,
        device,
    )

    print("\nFinal Test Results - CNN Optuna")
    print(f"Best threshold:  {threshold:.4f}")
    print(f"Test Loss:       {test_loss:.4f}")
    print(f"Test IoU:        {test_metrics['iou']:.4f}")
    print(f"Test Dice:       {test_metrics['dice']:.4f}")
    print(f"Test Precision:  {test_metrics['precision']:.4f}")
    print(f"Test Recall:     {test_metrics['recall']:.4f}")
    print(f"Test F-measure:  {test_metrics['f_measure']:.4f}")
    print(f"PR AUC:          {pr_auc:.4f}")

    save_pr_curve_plot(recall, precision, pr_auc, pr_curve_path)

    save_results_txt(
        test_metrics,
        test_loss,
        pr_auc,
        results_txt_path,
        extra_info=(
            "Model: CNNWithSkips Optuna tuned\n"
            f"Best params: {best_params}"
        ),
    )

    print(f"Saved PR curve to {pr_curve_path}")
    print(f"Saved results to {results_txt_path}")


def main():
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
        ),
    )

    study.optimize(objective, n_trials=20)

    print("\nBest Trial")
    print(f"Best Val Dice: {study.best_value:.4f}")
    print("Best Params:")
    print(study.best_params)

    train_final_model(study.best_params)


if __name__ == "__main__":
    main()
