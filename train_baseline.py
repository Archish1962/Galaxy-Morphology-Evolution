"""
Training script for the Baseline Image-Only Galaxy Classifier.

Trains a frozen-backbone ResNet18 on galaxy morphology classification
with class-weighted cross-entropy loss.

Outputs:
    checkpoints/baseline_resnet18.pth   — Best model weights
    outputs/baseline/training_log.csv   — Per-epoch metrics
    outputs/baseline/confusion_matrix.png
    outputs/baseline/training_curves.png

Usage:
    python train_baseline.py                    # Auto-detects GPU/CPU
    python train_baseline.py --epochs 30        # Custom epoch count
    python train_baseline.py --batch-size 64    # Custom batch size
    python train_baseline.py --device cuda      # Force GPU
    python train_baseline.py --device cpu       # Force CPU
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
matplotlib.use("Agg")  # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
)

from dataset import (
    GalaxyDataset, create_splits, get_transforms,
    get_class_weights, CLASS_NAMES, NUM_CLASSES
)
from baseline_model import BaselineResNet18


# === Configuration ===
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = os.path.join("outputs", "baseline")


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline galaxy classifier")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cuda' or 'cpu' (auto-detected if not set)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (set to 0 if issues on Windows)")
    parser.add_argument("--fine-tune", action="store_true",
                        help="Unfreeze the entire network to fine-tune pre-trained weights")
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
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="  Training", leave=False, dynamic_ncols=True):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate on val/test set. Returns loss, accuracy, all preds and labels."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="  Evaluating", leave=False, dynamic_ncols=True):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def plot_training_curves(log_path, output_path):
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

    fig.suptitle("Baseline ResNet18 — Training Curves", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(labels, preds, output_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Baseline ResNet18 — Confusion Matrix (Test Set)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  BASELINE IMAGE CLASSIFIER — TRAINING")
    print("=" * 60)

    # --- Device ---
    print("\n[1/6] Setting up device...")
    device = get_device(args.device)

    # --- Data ---
    print("\n[2/6] Loading data & creating splits...")
    train_df, val_df, test_df = create_splits()

    train_ds = GalaxyDataset(train_df, transform=get_transforms("train"))
    val_ds = GalaxyDataset(val_df, transform=get_transforms("val"))
    test_ds = GalaxyDataset(test_df, transform=get_transforms("test"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    # --- Model ---
    print("\n[3/6] Building model...")
    model = BaselineResNet18(num_classes=NUM_CLASSES, freeze_backbone=not args.fine_tune).to(device)
    model.count_params()

    # --- Loss & Optimizer ---
    print("\n[4/6] Setting up training...")
    class_weights = get_class_weights(train_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.get_trainable_params(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    print(f"  Optimizer:    Adam (lr={args.lr})")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")

    # --- Training Loop ---
    print(f"\n[5/6] Training for {args.epochs} epochs...")
    log_path = os.path.join(OUTPUT_DIR, "training_log.csv")
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

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(),
                           os.path.join(CHECKPOINT_DIR, "baseline_resnet18.pth"))

            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  F1: {val_f1:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"{elapsed:.1f}s"
                  + (" ★" if epoch == best_epoch and val_acc == best_val_acc else ""))

    print(f"\n  Best val accuracy: {best_val_acc:.4f} (epoch {best_epoch})")

    # --- Test Evaluation ---
    print("\n[6/6] Evaluating on test set (using best model)...")
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "baseline_resnet18.pth"),
                                     map_location=device, weights_only=True))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    test_f1 = f1_score(test_labels, test_preds, average="macro")

    print(f"\n  Test Accuracy: {test_acc:.4f}")
    print(f"  Test F1 (macro): {test_f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))

    # --- Save Plots ---
    plot_training_curves(log_path, os.path.join(OUTPUT_DIR, "training_curves.png"))
    plot_confusion_matrix(test_labels, test_preds, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    print(f"  Training curves:    {os.path.join(OUTPUT_DIR, 'training_curves.png')}")
    print(f"  Confusion matrix:   {os.path.join(OUTPUT_DIR, 'confusion_matrix.png')}")
    print(f"  Best checkpoint:    {os.path.join(CHECKPOINT_DIR, 'baseline_resnet18.pth')}")
    print(f"  Training log:       {log_path}")

    print("\n" + "=" * 60)
    print("  BASELINE TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
