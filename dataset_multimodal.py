"""
Multimodal PyTorch Dataset for Galaxy Morphology Classification.

Extends the image-only dataset to also return normalized metadata features:
    - redshift
    - color_u_r, color_g_i  (extinction-corrected)
    - fracdev_r, petror50_r_kpc, mu50_r  (structural)

The metadata is z-score standardized using training set statistics.

Usage:
    from dataset_multimodal import MultimodalGalaxyDataset, get_metadata_stats

    stats = get_metadata_stats(train_df)
    train_ds = MultimodalGalaxyDataset(train_df, IMAGE_DIR, stats, transform=get_transforms("train"))
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

from dataset import (
    IMAGE_DIR, MASTER_CSV, CLASS_TO_IDX, IMAGENET_MEAN, IMAGENET_STD,
    get_transforms, create_splits, get_class_weights
)

# The 6 metadata features used by the Metadata MLP
METADATA_FEATURES = [
    "redshift",
    "color_u_r",
    "color_g_i",
    "fracdev_r",
    "petror50_r_kpc",
    "mu50_r",
]
NUM_META_FEATURES = len(METADATA_FEATURES)


class MultimodalGalaxyDataset(Dataset):
    """
    PyTorch Dataset that returns (image, metadata, label) tuples.

    Each __getitem__ returns:
        image:    Tensor [3, 224, 224]
        metadata: Tensor [6] (z-score normalized)
        label:    int
    """

    def __init__(self, dataframe, image_dir=IMAGE_DIR, meta_stats=None, transform=None):
        """
        Args:
            dataframe:  DataFrame with image_path, morph_class, and metadata columns.
            image_dir:  Directory containing galaxy JPEG images.
            meta_stats: Dict with 'mean' and 'std' tensors for z-score normalization.
                        Compute from training set via get_metadata_stats().
            transform:  torchvision transforms for images.
        """
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

        # Pre-encode labels
        self.labels = self.df["morph_class"].map(CLASS_TO_IDX).values
        self.image_paths = self.df["image_path"].values

        # Extract and normalize metadata
        meta_values = self.df[METADATA_FEATURES].values.astype(np.float32)
        self.metadata = torch.tensor(meta_values, dtype=torch.float32)

        if meta_stats is not None:
            self.metadata = (self.metadata - meta_stats["mean"]) / (meta_stats["std"] + 1e-8)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Get metadata and label
        metadata = self.metadata[idx]
        label = self.labels[idx]

        return image, metadata, label


def get_metadata_stats(train_df):
    """
    Compute mean and std of metadata features from the TRAINING set only.

    These stats are used to z-score normalize metadata in all splits,
    preventing data leakage from val/test sets.

    Args:
        train_df: Training DataFrame

    Returns:
        Dict with 'mean' and 'std' as torch.Tensors of shape [6]
    """
    values = train_df[METADATA_FEATURES].values.astype(np.float32)
    stats = {
        "mean": torch.tensor(values.mean(axis=0), dtype=torch.float32),
        "std": torch.tensor(values.std(axis=0), dtype=torch.float32),
    }

    print("Metadata normalization stats (from training set):")
    for i, feat in enumerate(METADATA_FEATURES):
        print(f"  {feat:20s}: mean={stats['mean'][i]:.4f}  std={stats['std'][i]:.4f}")

    return stats


# === Quick self-test ===
if __name__ == "__main__":
    print("=" * 50)
    print("  MULTIMODAL DATASET SELF-TEST")
    print("=" * 50)

    # Create splits
    print("\n[1] Creating splits...")
    train_df, val_df, test_df = create_splits()

    # Compute stats from training set
    print("\n[2] Computing metadata stats...")
    stats = get_metadata_stats(train_df)

    # Create datasets
    print("\n[3] Creating multimodal datasets...")
    train_ds = MultimodalGalaxyDataset(train_df, meta_stats=stats, transform=get_transforms("train"))
    val_ds = MultimodalGalaxyDataset(val_df, meta_stats=stats, transform=get_transforms("val"))

    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    # Load one sample
    print("\n[4] Loading a sample...")
    img, meta, label = train_ds[0]
    print(f"  Image shape:    {img.shape}")
    print(f"  Metadata shape: {meta.shape}")
    print(f"  Metadata values: {meta}")
    print(f"  Label: {label}")

    print("\n✅ All checks passed!")
