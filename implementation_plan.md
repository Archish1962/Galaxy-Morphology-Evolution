# Galaxy Morphology Evolution — Implementation Plan

This plan covers the full project from data merging to final analysis of morphological evolution.

## Current State

| Asset | Status | Location |
|-------|--------|----------|
| GZ2 images | ✅ Ready | `images_gz2/images/` |
| `gz2_hart16.csv` | ✅ Ready | Root dir (Labels: Morphology vote fractions) |
| `gz2_filename_mapping.csv` | ✅ Ready | Root dir (dr7objid → image path mapping) |
| `gz2sample.csv` | ✅ Ready | Root dir (Metadata: Redshift + Photometry) |

---

## Project Phases

### Phase 0 — Data Integration
- **Script:** `prepare_dataset.py`
- Merge all sources into `galaxy_master_dataset.csv`.
- Derived features: `color_u_r`, `color_g_i`.
- Filtering for "clean" samples and valid images.

### Phase 1 — Baseline Image Classifier
- **Script:** `train_baseline.py`
- Fine-tune ResNet18 (frozen backbone) on morphology classes.
- Metrics: Accuracy, F1-Score, Confusion Matrix.

### Phase 2 — Multimodal Fusion Model
- **Script:** `train_multimodal.py`
- Combine ResNet image embeddings with Metadata MLP output.
- Analyze if physical context improves classification.

### Phase 3 — Latent Embedding & Drift Analysis
- **Script:** `centroid_drift.py`
- Extract fused embeddings per galaxy.
- Group by redshift bins.
- Compute centroid movement (evolution proxy) per class.

### Phase 4 — Final Visualization
- **Script:** `visualize_results.py`
- Latent space plots (t-SNE/UMAP).
- Centroid drift curves.

---

## Project Flow Reference

The step-by-step checklist is tracked in:
- [task.md](file:///C:/Users/krish/.gemini/antigravity/brain/730a424a-3f40-4948-8c67-7d744481088a/task.md)

---

## Required Installations

Run this command to install all necessary Python packages:

```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib seaborn umap-learn Pillow tqdm
```
