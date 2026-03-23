# Galaxy Morphology Evolution Lab

![Streamlit Status](https://img.shields.io/badge/Streamlit-App_Live-FF4B4B?logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch)
![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-3776AB?logo=python)

This repository contains an end-to-end deep learning pipeline to investigate **how galaxy morphology evolves across cosmic time**. By combining raw image data from the Sloan Digital Sky Survey (SDSS DR7 / Galaxy Zoo 2) with spectroscopic metadata, we track how specific galaxy classes (disk, edge-on, smooth, spiral) drift in a high-dimensional learned embedding space.

---

## Scientific Context

As we look deeper into the universe (higher redshift $z$), galaxy resolution and surface brightness degrade. This makes morphological classification difficult.

**Our approach:**
1. Train a **multimodal neural network** combining raw image cutouts with robust SDSS metadata (color, structure size).
2. Extract **256-dimensional embeddings** representing the abstract morphology of each galaxy.
3. Compute the **Centroid Drift** across redshift bins ($low\_z \rightarrow mid\_z \rightarrow high\_z$).

**Key Findings:**
* **Disk galaxies drift the most** as redshift increases, largely due to resolution bias confusing them with smooth galaxies.
* **Spiral galaxies are the most stable**, indicating spiral arms remain robust classification features even at higher distances.
* **Multimodal fusion outperforms image-only models** by ~2% in accuracy and 4% in macro F1 score.

---

## Project Structure

The project is divided into four distinct phases, plus a final interactive dashboard.

### 1. Baselines (`train_baseline.py`, `train_efficientnet.py`)
Trains pure-image classifiers (ResNet-18 and EfficientNet-B0) to establish a performance benchmark.

### 2. Multimodal Fusion (`train_multimodal.py`)
Trains a custom `MultimodalFusionNet` that injects metadata features (redshift, color $u-r$, color $g-i$, structural $fracdev\_r$) into the intermediate features of a pre-trained image backbone.

### 3. Drift Analysis (`drift_analysis.py`)
The core science script. It pushes all 47k galaxies through the trained model, bins them by redshift, computes class centroids, and measures Euclidean/Cosine drift.

### 4. Interactive Maps (`phase4_visualize.py`)
Flattens the 256-d embedding space down to 2 dimensions using **UMAP** and **t-SNE** to create interactive HTML visualisations of the galaxy clusters.

### 5. The Dashboard (`galaxy_app.py`)
A comprehensive Streamlit application that wraps all EDA, training curves, drift analysis results, and a live image classifier into a single UI.

---

## Installation & Setup

This project uses `uv` for lightning-fast Python dependency management. Look at `pyproject.toml` for the exact requirements.

### 1. Clone & Install Dependencies
```bash
git clone <repository_url>
cd data-science-projecr
uv sync
```

### 2. Data Requirements
You need the Galaxy Zoo 2 dataset in the project root:
* `galaxy_master_dataset.csv`
* `images_gz2/` (folder containing 47,089 galaxy `.jpg` cutouts)

### 3. Run the Streamlit Dashboard
The easiest way to explore the project results and test the live classifier is to launch the dashboard:
```bash
uv run streamlit run galaxy_app.py
```

*Note: Pre-trained checkpoints (`multimodal_resnet18.pth`) must be present in the `checkpoints/` directory for the Live Classifier to work.*

---

## Training the Models from Scratch

If you want to re-run the experiments or train on a different dataset, you can run the scripts sequentially.

**1. Train the Baseline Image Models**
```bash
uv run train_baseline.py --epochs 20 --fine-tune
uv run train_efficientnet.py --epochs 20 --fine-tune
```

**2. Train the Multimodal Models**
```bash
# EfficientNet backbone
uv run train_multimodal.py --backbone efficientnet_b0 --epochs 20 --fine-tune

# ResNet-18 backbone (Best performing)
uv run train_multimodal.py --backbone resnet18 --epochs 20 --fine-tune
```

**3. Run the Science (Drift Analysis)**
```bash
uv run drift_analysis.py --backbone resnet18
```

**4. Generate UMAP/t-SNE Embeddings**
```bash
uv run phase4_visualize.py --backbone resnet18 --method both
```

---

## Environment Notes
- **Windows Users:** The `--num-workers 0` flag is set by default in the training scripts to prevent PyTorch DataLoader deadlocks on Windows machines.
- **GPU Acceleration:** The `pyproject.toml` automatically configures PyTorch to use the `cu128` (CUDA 12.8) index for NVIDIA GPU acceleration.

## Documentation
For a granular history of design decisions, bugs fixed, and exact parameters used throughout development, see the `project_changelog.md`.
