"""
Phase 3: Embedding Extraction & Centroid Drift Analysis.

This script:
  1. Loads the trained multimodal fusion model
  2. Extracts 256-d fused embeddings for ALL galaxies
  3. Bins galaxies into 3 science-driven redshift bins
  4. Computes per-class centroid in each bin
  5. Measures centroid drift (Euclidean + Cosine) between adjacent bins
  6. Reports drift for z > 0.14 ONLY for smooth (other classes lack statistical validity)

Outputs:
    outputs/drift/embeddings.npy            — All 256-d embeddings
    outputs/drift/embedding_labels.csv      — Metadata for each embedding row
    outputs/drift/centroid_drift.csv        — Drift values per class per bin transition
    outputs/drift/drift_curves.png          — Drift curves visualization
    outputs/drift/bin_population.png        — Population counts per bin per class

Requires:
    - Trained model checkpoint: checkpoints/multimodal_fusion.pth
    - Metadata stats: checkpoints/metadata_stats.pth
    - Master dataset: galaxy_master_dataset.csv
    - Images: images_gz2/

Usage:
    python drift_analysis.py
    python drift_analysis.py --batch-size 64
    python drift_analysis.py --device cpu
"""

import argparse
import os
import csv
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.spatial.distance import cosine as cosine_distance

from dataset import (
    CLASS_NAMES, CLASS_TO_IDX, get_transforms, MASTER_CSV
)
from dataset_multimodal import (
    MultimodalGalaxyDataset, get_metadata_stats,
    METADATA_FEATURES, NUM_META_FEATURES
)
from multimodal_model import MultimodalFusionNet

# === Configuration ===
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = os.path.join("outputs", "drift")

# Science-driven redshift bins (per collaborator feedback)
# Spiral, disk, and edge_on vanish above z ≈ 0.13 due to SDSS resolution effects.
# Hart et al.'s debiasing helps but doesn't fully eliminate this.
REDSHIFT_BINS = [
    ("low_z",  0.01, 0.06),   # All classes well-populated
    ("mid_z",  0.06, 0.10),   # All classes well-populated
    ("high_z", 0.10, 0.14),   # Marginal for disk/edge_on
]

# Extended bin ONLY for smooth galaxies (others lack statistical validity beyond z=0.14)
SMOOTH_EXTENDED_BIN = ("ext_z", 0.14, 0.25)

# Minimum galaxies needed in a bin for a class to be statistically valid
MIN_BIN_COUNT = 30


def parse_args():
    parser = argparse.ArgumentParser(description="Embedding extraction & drift analysis")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detected if not set)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (set to 0 on Windows)")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0",
                        choices=["resnet18", "efficientnet_b0"],
                        help="Image backbone used during training")
    return parser.parse_args()


def get_device(requested=None):
    if requested:
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")
    return device


@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    """
    Extract 256-d fused embeddings from the multimodal model.

    Returns:
        embeddings: np.array [N, 256]
        labels: np.array [N]
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    for images, metadata, labels in tqdm(dataloader, desc="  Extracting embeddings", dynamic_ncols=True):
        images = images.to(device)
        metadata = metadata.to(device)

        emb = model.get_embedding(images, metadata)
        all_embeddings.append(emb.cpu().numpy())
        all_labels.append(labels.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return embeddings, labels



def assign_bins(redshifts, bins, smooth_ext_bin=None, morph_classes=None):
    """
    Assign each galaxy to a redshift bin.

    Args:
        redshifts: array of redshift values
        bins: list of (name, z_low, z_high) tuples
        smooth_ext_bin: optional extended bin for smooth-only galaxies
        morph_classes: array of morph class strings (needed if smooth_ext_bin is used)

    Returns:
        bin_assignments: array of bin name strings (or "none" if unassigned)
    """
    assignments = np.full(len(redshifts), "none", dtype=object)

    for name, z_low, z_high in bins:
        mask = (redshifts >= z_low) & (redshifts < z_high)
        assignments[mask] = name

    # Extended bin for smooth only
    if smooth_ext_bin is not None and morph_classes is not None:
        ext_name, ext_low, ext_high = smooth_ext_bin
        mask = (redshifts >= ext_low) & (redshifts < ext_high) & (morph_classes == "smooth")
        assignments[mask] = ext_name

    return assignments


def compute_centroids(embeddings, labels, bin_assignments, class_names, bin_names):
    """
    Compute mean centroid embedding per class per bin.

    Returns:
        centroids: dict of {(class, bin): np.array [256]}
        counts: dict of {(class, bin): int}
    """
    centroids = {}
    counts = {}

    for cls in class_names:
        cls_idx = CLASS_TO_IDX[cls]
        for bin_name in bin_names:
            mask = (labels == cls_idx) & (bin_assignments == bin_name)
            count = mask.sum()
            counts[(cls, bin_name)] = count

            if count >= MIN_BIN_COUNT:
                centroids[(cls, bin_name)] = embeddings[mask].mean(axis=0)
            else:
                centroids[(cls, bin_name)] = None

    return centroids, counts


def compute_drift(centroids, class_names, bin_names):
    """
    Compute centroid drift between adjacent redshift bins.

    Returns:
        drift_results: list of dicts with drift metrics
    """
    results = []

    for cls in class_names:
        for i in range(len(bin_names) - 1):
            bin_a = bin_names[i]
            bin_b = bin_names[i + 1]

            c_a = centroids.get((cls, bin_a))
            c_b = centroids.get((cls, bin_b))

            if c_a is not None and c_b is not None:
                euclidean = float(np.linalg.norm(c_a - c_b))
                cosine = float(cosine_distance(c_a, c_b))
                valid = True
            else:
                euclidean = float("nan")
                cosine = float("nan")
                valid = False

            results.append({
                "class": cls,
                "bin_from": bin_a,
                "bin_to": bin_b,
                "euclidean_drift": euclidean,
                "cosine_drift": cosine,
                "valid": valid,
            })

    return results


def plot_bin_populations(counts, bin_names, output_path):
    """Bar chart of galaxy counts per class per bin."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(bin_names))
    width = 0.18
    colors = {"disk": "#45B7D1", "edge_on": "#FFA07A", "smooth": "#FF6B6B", "spiral": "#4ECDC4"}

    for i, cls in enumerate(CLASS_NAMES):
        values = [counts.get((cls, b), 0) for b in bin_names]
        bars = ax.bar(x + i * width, values, width, label=cls, color=colors[cls], edgecolor="white")
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                        str(val), ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Redshift Bin", fontsize=12)
    ax.set_ylabel("Galaxy Count", fontsize=12)
    ax.set_title("Galaxy Population per Redshift Bin", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(bin_names, fontsize=10)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Draw minimum threshold line
    ax.axhline(y=MIN_BIN_COUNT, color="red", linestyle="--", alpha=0.5, label=f"Min threshold ({MIN_BIN_COUNT})")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_drift_curves(drift_results, output_path):
    """Plot Euclidean and Cosine drift curves per morphology class."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"disk": "#45B7D1", "edge_on": "#FFA07A", "smooth": "#FF6B6B", "spiral": "#4ECDC4"}
    markers = {"disk": "D", "edge_on": "^", "smooth": "o", "spiral": "s"}

    for cls in CLASS_NAMES:
        cls_results = [r for r in drift_results if r["class"] == cls and r["valid"]]
        if not cls_results:
            continue

        labels = [f"{r['bin_from']}→{r['bin_to']}" for r in cls_results]
        euclid = [r["euclidean_drift"] for r in cls_results]
        cosine = [r["cosine_drift"] for r in cls_results]

        ax1.plot(labels, euclid, f"-{markers[cls]}", color=colors[cls],
                 label=cls, markersize=8, linewidth=2)
        ax2.plot(labels, cosine, f"-{markers[cls]}", color=colors[cls],
                 label=cls, markersize=8, linewidth=2)

    ax1.set_title("Euclidean Centroid Drift", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Redshift Bin Transition")
    ax1.set_ylabel("Euclidean Distance")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Cosine Centroid Drift", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Redshift Bin Transition")
    ax2.set_ylabel("Cosine Distance")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Morphology Centroid Drift Across Cosmic Time",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = os.path.join("outputs", f"drift_{args.backbone}")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"  PHASE 3: EMBEDDING EXTRACTION & DRIFT ANALYSIS ({args.backbone})")
    print("=" * 60)

    # --- Device ---
    print("\n[1/6] Setting up device...")
    device = get_device(args.device)

    # --- Load Model ---
    print("\n[2/6] Loading trained multimodal model...")
    model_path = os.path.join(CHECKPOINT_DIR, f"multimodal_{args.backbone}.pth")
    stats_path = os.path.join(CHECKPOINT_DIR, "metadata_stats.pth")

    if not os.path.exists(model_path):
        print(f"  ERROR: Checkpoint not found at {model_path}")
        print("  Please run train_multimodal.py first!")
        return

    model = MultimodalFusionNet(num_classes=4, num_meta_features=NUM_META_FEATURES,
                                backbone=args.backbone)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"  Loaded checkpoint from {model_path}")

    # --- Load Data ---
    print("\n[3/6] Loading full dataset for embedding extraction...")
    df = pd.read_csv(MASTER_CSV)

    # Load metadata normalization stats (computed from training set during Phase 2)
    meta_stats = torch.load(stats_path, map_location="cpu", weights_only=True)
    print(f"  Loaded metadata stats from {stats_path}")

    dataset = MultimodalGalaxyDataset(
        df, meta_stats=meta_stats, transform=get_transforms("test")
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    print(f"  Dataset size: {len(dataset):,} galaxies")

    # --- Extract Embeddings ---
    print("\n[4/6] Extracting 256-d fused embeddings...")
    embeddings, labels = extract_embeddings(model, dataloader, device)
    print(f"  Embedding matrix shape: {embeddings.shape}")

    # Save raw embeddings
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)

    # Save per-row metadata for downstream use (t-SNE, UMAP, etc.)
    label_df = df[["dr7objid", "morph_class", "redshift"]].copy()
    label_df.to_csv(os.path.join(output_dir, "embedding_labels.csv"), index=False)
    print(f"  Saved embeddings + labels to {output_dir}/")

    # --- Redshift Binning ---
    print("\n[5/6] Binning galaxies by redshift...")
    redshifts = df["redshift"].values
    morph_classes = df["morph_class"].values

    # Build full bin list (3 main + 1 smooth-only extended)
    all_bins = REDSHIFT_BINS + [SMOOTH_EXTENDED_BIN]
    all_bin_names = [b[0] for b in all_bins]

    bin_assignments = assign_bins(redshifts, REDSHIFT_BINS, SMOOTH_EXTENDED_BIN, morph_classes)

    # Compute centroids
    centroids, counts = compute_centroids(embeddings, labels, bin_assignments, CLASS_NAMES, all_bin_names)

    # Print population table
    print(f"\n  {'Class':12s}", end="")
    for b in all_bin_names:
        print(f"  {b:>8s}", end="")
    print()
    print("  " + "-" * (12 + 10 * len(all_bin_names)))
    for cls in CLASS_NAMES:
        print(f"  {cls:12s}", end="")
        for b in all_bin_names:
            count = counts.get((cls, b), 0)
            valid = "✓" if centroids.get((cls, b)) is not None else "✗"
            print(f"  {count:>6d}{valid}", end="")
        print()

    # --- Compute Drift ---
    print("\n[6/6] Computing centroid drift...")

    # For main bins: all classes
    main_bin_names = [b[0] for b in REDSHIFT_BINS]
    drift_results = compute_drift(centroids, CLASS_NAMES, main_bin_names)

    # For extended bin: smooth only
    smooth_ext_bins = main_bin_names + [SMOOTH_EXTENDED_BIN[0]]
    smooth_ext_drift = compute_drift(centroids, ["smooth"], smooth_ext_bins)
    # Only add the ext_z transition (avoid duplicating main bin drifts)
    for r in smooth_ext_drift:
        if r["bin_to"] == SMOOTH_EXTENDED_BIN[0] or r["bin_from"] == SMOOTH_EXTENDED_BIN[0]:
            drift_results.append(r)

    # Save drift results
    drift_csv_path = os.path.join(output_dir, "centroid_drift.csv")
    with open(drift_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "bin_from", "bin_to",
                                                "euclidean_drift", "cosine_drift", "valid"])
        writer.writeheader()
        writer.writerows(drift_results)

    # Print drift table
    print(f"\n  {'Class':12s} {'Transition':18s} {'Euclidean':>10s} {'Cosine':>10s} {'Valid':>6s}")
    print("  " + "-" * 58)
    for r in drift_results:
        transition = f"{r['bin_from']}→{r['bin_to']}"
        eu = f"{r['euclidean_drift']:.4f}" if r["valid"] else "N/A"
        co = f"{r['cosine_drift']:.4f}" if r["valid"] else "N/A"
        v = "✓" if r["valid"] else "✗"
        print(f"  {r['class']:12s} {transition:18s} {eu:>10s} {co:>10s} {v:>6s}")

    # --- Plots ---
    plot_bin_populations(counts, all_bin_names, os.path.join(output_dir, "bin_population.png"))
    plot_drift_curves(drift_results, os.path.join(output_dir, "drift_curves.png"))

    print(f"\n  Drift CSV:       {drift_csv_path}")
    print(f"  Population plot: {os.path.join(output_dir, 'bin_population.png')}")
    print(f"  Drift curves:    {os.path.join(output_dir, 'drift_curves.png')}")

    print("\n" + "=" * 60)
    print("  PHASE 3 COMPLETE — DRIFT ANALYSIS DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
