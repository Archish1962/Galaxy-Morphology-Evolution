"""
Phase 4: t-SNE / UMAP Visualization of Galaxy Embeddings.

Reads pre-computed 256-d embeddings from Phase 3 and reduces them to 2D
using both UMAP and t-SNE. Outputs:

  1. Interactive Plotly HTML maps (hover shows galaxy image + redshift)
  2. Static side-by-side comparison PNG for reports

Supports both backbones. Run once per backbone, or auto-detects both.

Outputs (per backbone):
    outputs/phase4_{backbone}/umap_interactive.html    — Plotly hover map
    outputs/phase4_{backbone}/tsne_interactive.html    — Plotly hover map
    outputs/phase4_{backbone}/umap_static.png          — Static matplotlib
    outputs/phase4_{backbone}/tsne_static.png          — Static matplotlib

Usage:
    uv run phase4_visualize.py --backbone efficientnet_b0
    uv run phase4_visualize.py --backbone resnet18
    uv run phase4_visualize.py --backbone all           # Both (if both exist)
    uv run phase4_visualize.py --method umap            # UMAP only (faster)
    uv run phase4_visualize.py --method tsne            # t-SNE only
"""

import argparse
import os
import base64
import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Dimensionality reduction
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("  [WARNING] umap-learn not installed. Skipping UMAP. Run: uv add umap-learn")

# Interactive plotting
import plotly.graph_objects as go
import plotly.express as px

from dataset import CLASS_NAMES

# === Configuration ===
CLASS_COLORS = {
    "disk":    "#45B7D1",
    "edge_on": "#FFA07A",
    "smooth":  "#FF6B6B",
    "spiral":  "#4ECDC4",
}
CLASS_SYMBOLS = {
    "disk":    "circle",
    "edge_on": "triangle-up",
    "smooth":  "square",
    "spiral":  "diamond",
}

TSNE_PERPLEXITY = 40
TSNE_ITERATIONS = 1000
TSNE_SAMPLE_SIZE = 8000   # t-SNE is slow — subsample for speed
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
UMAP_SAMPLE_SIZE = 20000  # UMAP handles more — larger sample


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4 UMAP/t-SNE Visualization")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0",
                        choices=["resnet18", "efficientnet_b0", "all"],
                        help="Which backbone's embeddings to visualize")
    parser.add_argument("--method", type=str, default="both",
                        choices=["umap", "tsne", "both"],
                        help="Dimensionality reduction method")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Override sample size (default: method-specific)")
    return parser.parse_args()


def load_embeddings(backbone):
    """Load embeddings + labels from Phase 3 outputs."""
    drift_dir = os.path.join("outputs", f"drift_{backbone}")
    emb_path = os.path.join(drift_dir, "embeddings.npy")
    label_path = os.path.join(drift_dir, "embedding_labels.csv")

    if not os.path.exists(emb_path):
        print(f"  [SKIP] No embeddings found for {backbone} at {emb_path}")
        print(f"  Run: uv run drift_analysis.py --backbone {backbone}")
        return None, None

    embeddings = np.load(emb_path)
    labels_df = pd.read_csv(label_path)
    print(f"  Loaded {len(embeddings):,} embeddings ({backbone})")
    return embeddings, labels_df


def image_to_base64(img_path, size=(80, 80)):
    """Convert a galaxy image to a base64 string for Plotly hover."""
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def build_hover_text(labels_df, sample_idx):
    """Build clean text hover showing class, redshift, and object ID."""
    texts = []
    for idx in sample_idx:
        row = labels_df.iloc[idx]
        cls = row.get("morph_class", "unknown")
        z = row.get("redshift", 0.0)
        objid = row.get("dr7objid", "")
        texts.append(
            f"<b>Class:</b> {cls.capitalize()}<br>"
            f"<b>Redshift:</b> z = {z:.4f}<br>"
            f"<b>ObjID:</b> {objid}"
        )
    return texts


def make_plotly_scatter(coords_2d, labels_df, sample_idx, hover_texts, title, output_path):
    """Create an interactive Plotly scatter plot with clean text hover tooltips."""
    classes = [labels_df.iloc[i]["morph_class"] for i in sample_idx]

    fig = go.Figure()

    for cls in CLASS_NAMES:
        mask = np.array([c == cls for c in classes])
        x = coords_2d[mask, 0]
        y = coords_2d[mask, 1]
        texts = [t for t, m in zip(hover_texts, mask) if m]

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers",
            name=cls.capitalize(),
            marker=dict(
                color=CLASS_COLORS[cls],
                size=4,
                opacity=0.75,
                symbol=CLASS_SYMBOLS[cls],
                line=dict(width=0),
            ),
            hovertext=texts,
            hovertemplate="%{hovertext}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family="Arial Black"), x=0.5),
        xaxis=dict(title="Dim 1", showgrid=True, gridcolor="#2a2a2a", tickfont=dict(color="white")),
        yaxis=dict(title="Dim 2", showgrid=True, gridcolor="#2a2a2a", tickfont=dict(color="white")),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="white"),
        legend=dict(
            title="Morphology Class",
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="white",
            borderwidth=1,
        ),
        width=1200,
        height=800,
        hoverlabel=dict(
            bgcolor="#0f3460",
            bordercolor="white",
            font=dict(size=13, family="Arial", color="white"),
        ),
    )

    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"  Saved interactive plot → {output_path}")
    return fig


def make_static_scatter(coords_2d, labels_df, sample_idx, title, output_path):
    """Create a static matplotlib scatter plot for reports."""
    classes = [labels_df.iloc[i]["morph_class"] for i in sample_idx]

    fig, ax = plt.subplots(figsize=(12, 10), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")

    for cls in CLASS_NAMES:
        mask = np.array([c == cls for c in classes])
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=CLASS_COLORS[cls], label=cls.capitalize(),
            s=3, alpha=0.6, linewidths=0, rasterized=True
        )

    ax.set_title(title, color="white", fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("Dim 1", color="white")
    ax.set_ylabel("Dim 2", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    legend = ax.legend(
        fontsize=11, markerscale=4,
        facecolor="#1a1a2e", edgecolor="#666",
        labelcolor="white"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  Saved static plot → {output_path}")


def run_tsne(embeddings, sample_size):
    """Run t-SNE on a subsample of embeddings."""
    n = min(sample_size, len(embeddings))
    idx = np.random.choice(len(embeddings), size=n, replace=False)
    idx.sort()
    sub = embeddings[idx]

    print(f"  Running t-SNE on {n:,} galaxies "
          f"(perplexity={TSNE_PERPLEXITY}, iters={TSNE_ITERATIONS})...")
    tsne = TSNE(
        n_components=2, perplexity=TSNE_PERPLEXITY,
        max_iter=TSNE_ITERATIONS, random_state=42,
        verbose=1, n_jobs=-1
    )
    coords = tsne.fit_transform(sub)
    return coords, idx


def run_umap(embeddings, sample_size):
    """Run UMAP on a subsample of embeddings."""
    if not UMAP_AVAILABLE:
        return None, None

    n = min(sample_size, len(embeddings))
    idx = np.random.choice(len(embeddings), size=n, replace=False)
    idx.sort()
    sub = embeddings[idx]

    print(f"  Running UMAP on {n:,} galaxies "
          f"(n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST})...")
    reducer = umap.UMAP(
        n_components=2, n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST, random_state=42,
        verbose=True
    )
    coords = reducer.fit_transform(sub)
    return coords, idx


def process_backbone(backbone, method, sample_size_override):
    """Run full Phase 4 pipeline for one backbone."""
    print(f"\n{'='*60}")
    print(f"  Processing: {backbone}")
    print(f"{'='*60}")

    # Load
    embeddings, labels_df = load_embeddings(backbone)
    if embeddings is None:
        return

    output_dir = os.path.join("outputs", f"phase4_{backbone}")
    os.makedirs(output_dir, exist_ok=True)

    # Merge image paths if not in labels CSV (join from master CSV)
    if "image_path" not in labels_df.columns:
        try:
            master = pd.read_csv("galaxy_master_dataset.csv", usecols=["dr7objid", "image_path"])
            labels_df = labels_df.merge(master, on="dr7objid", how="left")
        except Exception:
            labels_df["image_path"] = None

    # Resolve image paths: master CSV stores bare filenames like "110939.jpg"
    # We need to prepend the images_gz2/ folder and convert to absolute paths
    img_dir = os.path.abspath("images_gz2")
    if "image_path" in labels_df.columns:
        labels_df["image_path"] = labels_df["image_path"].apply(
            lambda p: os.path.join(img_dir, os.path.basename(str(p)))
            if pd.notna(p) else None
        )

    # ---- UMAP ----
    if method in ("umap", "both") and UMAP_AVAILABLE:
        sz = sample_size_override or UMAP_SAMPLE_SIZE
        umap_coords, umap_idx = run_umap(embeddings, sz)

        if umap_coords is not None:
            hover_texts = build_hover_text(labels_df, umap_idx)

            make_plotly_scatter(
                umap_coords, labels_df, umap_idx, hover_texts,
                title=f"Galaxy Morphology — UMAP ({backbone})",
                output_path=os.path.join(output_dir, "umap_interactive.html"),
            )
            make_static_scatter(
                umap_coords, labels_df, umap_idx,
                title=f"Galaxy Morphology — UMAP ({backbone})",
                output_path=os.path.join(output_dir, "umap_static.png"),
            )

            # Save 2D coords for Streamlit
            np.save(os.path.join(output_dir, "umap_coords.npy"), umap_coords)
            np.save(os.path.join(output_dir, "umap_sample_idx.npy"), umap_idx)

    # ---- t-SNE ----
    if method in ("tsne", "both"):
        sz = sample_size_override or TSNE_SAMPLE_SIZE
        tsne_coords, tsne_idx = run_tsne(embeddings, sz)

        hover_texts = build_hover_text(labels_df, tsne_idx)

        make_plotly_scatter(
            tsne_coords, labels_df, tsne_idx, hover_texts,
            title=f"Galaxy Morphology — t-SNE ({backbone})",
            output_path=os.path.join(output_dir, "tsne_interactive.html"),
        )
        make_static_scatter(
            tsne_coords, labels_df, tsne_idx,
            title=f"Galaxy Morphology — t-SNE ({backbone})",
            output_path=os.path.join(output_dir, "tsne_static.png"),
        )

        # Save 2D coords for Streamlit (used for image lookup in galaxy_app.py)
        np.save(os.path.join(output_dir, "tsne_coords.npy"), tsne_coords)
        np.save(os.path.join(output_dir, "tsne_sample_idx.npy"), tsne_idx)

    print(f"\n  Phase 4 complete for {backbone}!")
    print(f"  Outputs in: {output_dir}/")


def main():
    global UMAP_AVAILABLE
    args = parse_args()

    print("=" * 60)
    print("  PHASE 4: t-SNE / UMAP GALAXY EMBEDDING VISUALIZATION")
    print("=" * 60)

    if not UMAP_AVAILABLE and args.method in ("umap", "both"):
        print("\n  [!] umap-learn not installed. Installing now...")
        os.system("uv add umap-learn")
        try:
            import umap as _umap  # noqa
            UMAP_AVAILABLE = True
        except ImportError:
            print("  [!] Install failed. Falling back to t-SNE only.")
            args.method = "tsne"

    backbones = ["resnet18", "efficientnet_b0"] if args.backbone == "all" else [args.backbone]

    for backbone in backbones:
        process_backbone(backbone, args.method, args.sample_size)

    print("\n" + "=" * 60)
    print("  PHASE 4 COMPLETE")
    print("  Open the .html files in your browser for the interactive maps!")
    print("=" * 60)


if __name__ == "__main__":
    main()
