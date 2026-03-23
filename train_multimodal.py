"""
Training script for the Multimodal Fusion Galaxy Classifier.

Supports two image backbones selectable via --backbone:
    resnet18:        ~11.4M params, 512-d image embedding
    efficientnet_b0: ~4.2M params, 1280-d image embedding

Outputs (per backbone):
    checkpoints/multimodal_{backbone}.pth        — Best model weights
    checkpoints/metadata_stats.pth               — Metadata normalization stats
    outputs/multimodal_{backbone}/training_log.csv
    outputs/multimodal_{backbone}/confusion_matrix.png
    outputs/multimodal_{backbone}/training_curves.png
    outputs/multimodal_{backbone}/baseline_comparison.png

Usage:
    uv run train_multimodal.py                                        # ResNet-18, frozen
    uv run train_multimodal.py --backbone efficientnet_b0 --fine-tune # EfficientNet, fine-tuned
    uv run train_multimodal.py --epochs 30 --batch-size 64            # Custom settings
"""

import argparse
import os
import time
import csv
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
)

from dataset import (
    create_splits, get_transforms, get_class_weights,
    CLASS_NAMES, NUM_CLASSES
)
from dataset_multimodal import (
    MultimodalGalaxyDataset, get_metadata_stats, NUM_META_FEATURES
)
from multimodal_model import MultimodalFusionNet


# === Configuration ===
CHECKPOINT_DIR = "checkpoints"


def parse_args():
    parser = argparse.ArgumentParser(description="Train multimodal galaxy classifier")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cuda' or 'cpu' (auto-detected if not set)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (set to 0 if issues on Windows)")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "efficientnet_b0"],
                        help="Image backbone: 'resnet18' or 'efficientnet_b0'")
    parser.add_argument("--fine-tune", action="store_true",
                        help="Unfreeze the entire image backbone for fine-tuning")
    return parser.parse_args()


def get_device(requested=None):
    """Auto-detect best available device."""
    if requested:
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch with multimodal data."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, metadata, labels in tqdm(dataloader, desc="  Training", leave=False, dynamic_ncols=True):
        images = images.to(device)
        metadata = metadata.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images, metadata)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate multimodal model. Returns loss, accuracy, preds, labels."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, metadata, labels in tqdm(dataloader, desc="  Evaluating", leave=False, dynamic_ncols=True):
        images = images.to(device)
        metadata = metadata.to(device)
        labels = labels.to(device)

        logits = model(images, metadata)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def plot_training_curves(log_path, output_path, backbone):
    """Plot training and validation loss/accuracy curves."""
    epochs, train_losses, val_losses = [], [], []
    train_accs, val_accs = [], []

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            train_accs.append(float(row["train_acc"]))
            val_accs.append(float(row["val_acc"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_losses, "b-o", label="Train", markersize=3)
    ax1.plot(epochs, val_losses, "r-o", label="Val", markersize=3)
    ax1.set_title("Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, "b-o", label="Train", markersize=3)
    ax2.plot(epochs, val_accs, "r-o", label="Val", markersize=3)
    ax2.set_title("Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Multimodal Fusion ({backbone}) — Training Curves",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(labels, preds, output_path, backbone):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Greens", values_format="d")
    ax.set_title(f"Multimodal ({backbone}) — Confusion Matrix (Test Set)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_baseline_comparison(output_dir, output_path):
    """
    Compare baseline (ResNet-18 image-only), EfficientNet (image-only),
    and the current multimodal run side by side.
    """
    def read_log(path):
        epochs, val_accs, val_f1s = [], [], []
        if not os.path.exists(path):
            return None, None, None
        with open(path, "r") as f:
            for row in csv.DictReader(f):
                epochs.append(int(row["epoch"]))
                val_accs.append(float(row["val_acc"]))
                val_f1s.append(float(row["val_f1"]))
        return epochs, val_accs, val_f1s

    log_sources = {
        "Baseline (ResNet-18)":     os.path.join("outputs", "baseline", "training_log.csv"),
        "EfficientNet-B0":          os.path.join("outputs", "efficientnet", "training_log.csv"),
        "Multimodal (this run)":    os.path.join(output_dir, "training_log.csv"),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["blue", "orange", "green"]
    markers = ["o", "s", "^"]

    for (label, path), color, marker in zip(log_sources.items(), colors, markers):
        epochs, accs, f1s = read_log(path)
        if epochs is None:
            continue
        ax1.plot(epochs, accs, f"-{marker}", color=color, label=label, markersize=3)
        ax2.plot(epochs, f1s, f"-{marker}", color=color, label=label, markersize=3)

    for ax, ylabel, title in [
        (ax1, "Accuracy", "Val Accuracy"),
        (ax2, "F1 Score", "Val F1 (Macro)"),
    ]:
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("All Models — Performance Comparison", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Comparison plot: {output_path}")


def main():
    args = parse_args()
    output_dir = os.path.join("outputs", f"multimodal_{args.backbone}")
    checkpoint_name = f"multimodal_{args.backbone}.pth"

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"  MULTIMODAL FUSION CLASSIFIER ({args.backbone.upper()}) — TRAINING")
    print("=" * 60)

    # --- Device ---
    print("\n[1/7] Setting up device...")
    device = get_device(args.device)

    # --- Data ---
    print("\n[2/7] Loading data & creating splits...")
    train_df, val_df, test_df = create_splits()

    # Compute metadata stats from training set (prevents data leakage)
    print("\n[3/7] Computing metadata normalization stats...")
    meta_stats = get_metadata_stats(train_df)
    torch.save(meta_stats, os.path.join(CHECKPOINT_DIR, "metadata_stats.pth"))
    print(f"  Saved stats to {os.path.join(CHECKPOINT_DIR, 'metadata_stats.pth')}")

    train_ds = MultimodalGalaxyDataset(train_df, meta_stats=meta_stats,
                                        transform=get_transforms("train"))
    val_ds = MultimodalGalaxyDataset(val_df, meta_stats=meta_stats,
                                      transform=get_transforms("val"))
    test_ds = MultimodalGalaxyDataset(test_df, meta_stats=meta_stats,
                                       transform=get_transforms("test"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    # --- Model ---
    print("\n[4/7] Building multimodal model...")
    model = MultimodalFusionNet(
        num_classes=NUM_CLASSES,
        num_meta_features=NUM_META_FEATURES,
        backbone=args.backbone,
        freeze_backbone=not args.fine_tune,
    ).to(device)
    model.count_params()

    # --- Loss & Optimizer ---
    print("\n[5/7] Setting up training...")
    class_weights = get_class_weights(train_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.Adam([
        {"params": model.image_backbone.parameters(), "lr": args.lr * 0.01},
        {"params": list(model.meta_mlp.parameters()) + list(model.fusion_head.parameters()), "lr": args.lr},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    print(f"  Backbone:     {args.backbone}")
    print(f"  Fine-tune:    {args.fine_tune}")
    print(f"  Optimizer:    Adam (lr={args.lr})")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")

    # --- Training Loop ---
    print(f"\n[6/7] Training for {args.epochs} epochs...")
    log_path = os.path.join(output_dir, "training_log.csv")
    best_val_acc = 0.0
    best_epoch = 0

    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "time_sec"])

        for epoch in range(1, args.epochs + 1):
            start = time.time()

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
            val_f1 = f1_score(val_labels, val_preds, average="macro")

            scheduler.step(val_acc)

            elapsed = time.time() - start
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}",
                             f"{val_loss:.4f}", f"{val_acc:.4f}", f"{val_f1:.4f}",
                             f"{elapsed:.1f}"])
            log_file.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(),
                           os.path.join(CHECKPOINT_DIR, checkpoint_name))

            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  F1: {val_f1:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"{elapsed:.1f}s"
                  + (" ★" if epoch == best_epoch and val_acc == best_val_acc else ""))

    print(f"\n  Best val accuracy: {best_val_acc:.4f} (epoch {best_epoch})")

    # --- Test Evaluation ---
    print("\n[7/7] Evaluating on test set (using best model)...")
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, checkpoint_name),
                                     map_location=device, weights_only=True))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    test_f1 = f1_score(test_labels, test_preds, average="macro")

    print(f"\n  Test Accuracy: {test_acc:.4f}")
    print(f"  Test F1 (macro): {test_f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))

    # --- Save Plots ---
    plot_training_curves(log_path, os.path.join(output_dir, "training_curves.png"), args.backbone)
    plot_confusion_matrix(test_labels, test_preds, os.path.join(output_dir, "confusion_matrix.png"), args.backbone)
    plot_baseline_comparison(output_dir, os.path.join(output_dir, "baseline_comparison.png"))

    print(f"\n  Training curves:    {os.path.join(output_dir, 'training_curves.png')}")
    print(f"  Confusion matrix:   {os.path.join(output_dir, 'confusion_matrix.png')}")
    print(f"  Best checkpoint:    {os.path.join(CHECKPOINT_DIR, checkpoint_name)}")
    print(f"  Training log:       {log_path}")

    print("\n" + "=" * 60)
    print("  MULTIMODAL TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
