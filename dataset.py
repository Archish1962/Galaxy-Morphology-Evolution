"""
PyTorch Dataset and data utilities for Galaxy Morphology Classification.

Provides:
  - GalaxyDataset: Custom Dataset that loads galaxy images + labels
  - get_transforms: Train/val image transforms (ImageNet-normalized)
  - create_splits: Stratified train/val/test split (70/15/15)
  - get_class_weights: Balanced class weights for CrossEntropyLoss

Usage:
    from dataset import GalaxyDataset, create_splits, get_transforms, get_class_weights

    train_df, val_df, test_df = create_splits("galaxy_master_dataset.csv")
    train_ds = GalaxyDataset(train_df, IMAGE_DIR, transform=get_transforms("train"))
    weights = get_class_weights(train_df)
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os

# === Constants ===
IMAGE_DIR = "images_gz2"
MASTER_CSV = "galaxy_master_dataset.csv"

# Class label encoding (alphabetical for reproducibility)
CLASS_NAMES = ["disk", "edge_on", "smooth", "spiral"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASS_NAMES)

# ImageNet normalization stats (required for pretrained ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class GalaxyDataset(Dataset):
    """
    PyTorch Dataset for galaxy images + morphology labels.

    Each __getitem__ returns:
        image:  Tensor [3, 224, 224] (normalized)
        label:  int (0=disk, 1=edge_on, 2=smooth, 3=spiral)
    """

    def __init__(self, dataframe, image_dir=IMAGE_DIR, transform=None):
        """
        Args:
            dataframe: DataFrame with 'image_path' and 'morph_class' columns.
            image_dir: Directory containing the galaxy JPEG images.
            transform: torchvision transforms to apply to each image.
        """
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

        # Pre-encode labels as integers
        self.labels = self.df["morph_class"].map(CLASS_TO_IDX).values
        self.image_paths = self.df["image_path"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get label
        label = self.labels[idx]

        return image, label


def get_transforms(split="train"):
    """
    Get image transforms for a given split.

    Train: Augmentations + resize + normalize
    Val/Test: Resize + normalize only
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def create_splits(csv_path=MASTER_CSV, train_ratio=0.70, val_ratio=0.15,
                  test_ratio=0.15, random_state=42):
    """
    Stratified train/val/test split of the master dataset.

    Args:
        csv_path: Path to galaxy_master_dataset.csv
        train_ratio: Fraction for training (default 0.70)
        val_ratio: Fraction for validation (default 0.15)
        test_ratio: Fraction for testing (default 0.15)
        random_state: Random seed for reproducibility

    Returns:
        train_df, val_df, test_df: DataFrames for each split
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    df = pd.read_csv(csv_path)
    labels = df["morph_class"]

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df, test_size=(val_ratio + test_ratio),
        stratify=labels, random_state=random_state
    )

    # Second split: val vs test (from the temp set)
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test_ratio,
        stratify=temp_df["morph_class"], random_state=random_state
    )

    print(f"Split sizes — Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
    print(f"Split ratios — Train: {len(train_df)/len(df):.2%}  "
          f"Val: {len(val_df)/len(df):.2%}  Test: {len(test_df)/len(df):.2%}")

    # Verify stratification
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = split_df["morph_class"].value_counts(normalize=True)
        print(f"  {name}: " + "  ".join(f"{c}={dist.get(c, 0):.1%}" for c in CLASS_NAMES))

    return train_df, val_df, test_df


def get_class_weights(train_df):
    """
    Compute balanced class weights from training set labels.

    Returns:
        torch.Tensor of shape (NUM_CLASSES,) for CrossEntropyLoss(weight=...)
    """
    labels = train_df["morph_class"].values
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(CLASS_NAMES),
        y=labels
    )
    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    print("Class weights (balanced):")
    for cls, w in zip(CLASS_NAMES, weights_tensor):
        print(f"  {cls:10s}: {w:.4f}")

    return weights_tensor


# === Quick self-test ===
if __name__ == "__main__":
    print("=" * 50)
    print("  DATASET SELF-TEST")
    print("=" * 50)

    # Test splits
    print("\n[1] Creating stratified splits...")
    train_df, val_df, test_df = create_splits()

    # Test class weights
    print("\n[2] Computing class weights...")
    weights = get_class_weights(train_df)

    # Test dataset loading
    print("\n[3] Testing dataset loading...")
    train_ds = GalaxyDataset(train_df, transform=get_transforms("train"))
    val_ds = GalaxyDataset(val_df, transform=get_transforms("val"))

    print(f"  Train dataset size: {len(train_ds):,}")
    print(f"  Val dataset size:   {len(val_ds):,}")

    # Load a single sample
    img, label = train_ds[0]
    print(f"  Sample image shape: {img.shape}")
    print(f"  Sample label: {label} ({IDX_TO_CLASS[label]})")
    print(f"  Pixel range: [{img.min():.3f}, {img.max():.3f}]")

    print("\n✅ All checks passed!")
