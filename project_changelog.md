# Galaxy Morphology Evolution — Project Changelog

Every decision, change, and rationale is documented here in reverse chronological order.

---

## 2026-03-21 — Final Interactive Dashboard (`galaxy_app.py`)

### What was done
Built a comprehensive, 6-page interactive Streamlit dashboard to present the entire project pipeline, EDA, model comparisons, scientific drift analysis, and a real-time multimodal classifier.

### Key Features
1. **🏠 Home:** Project overview, pipeline summary, and 4 key scientific findings regarding morphological drift.
2. **🔬 EDA:** 5-tab exploration of the raw Galaxy Zoo 2 data (class distribution, redshift, color-redshift correlation, structural FRACDEV_R, and an interactive galaxy image browser).
3. **📊 Model Performance:** Summary table and side-by-side training curves + confusion matrices for all 4 models (ResNet baseline, EfficientNet baseline, MM EfficientNet, MM ResNet).
4. **🌀 Drift Analysis:** Interactive Plotly bar charts showing Euclidean drift across redshift transitions for both backbones.
5. **🗺️ Galaxy Map:** Live, interactive UMAP and t-SNE scatter plots. Uses Plotly `customdata` to map points to the master dataset, allowing users to click any dot and instantly view the actual galaxy image and its SDSS metadata.
6. **🤖 Live Classifier:** Upload a galaxy images (or pick a random one from the dataset), adjust its SDSS structural/spectroscopic features using sliders, and get an instant real-time prediction from the trained PyTorch multimodal model.

### Bug Fixes during Development
- Resolved `streamlit` and `altair` package version conflicts by pinning `streamlit>=1.35.0` and `pandas>=2.0.0` in `pyproject.toml`.
- Fixed PyTorch model loading issue where the `state_dict` was saved directly rather than nested.
- Fixed Plotly `on_select` image matching by enforcing strict string casting on `dr7objid` to handle pandas float/int type discrepancies.
- Polished UI to remove redundant emojis for a cleaner, professional look.

---

## 2026-03-20 — Phase 4: Embedding Visualizations (`phase4_visualize.py`)

### What was done
Created the script to reduce the 256-dimensional multimodal embeddings down to 2D for human interpretation using UMAP and t-SNE.

### Features
- Support for generating plots for both `resnet18` and `efficientnet_b0` backbones.
- Dynamic sampling (`--sample-size 8000`) to prevent visual clutter and keep interactive HTML plots performant.
- Saves raw 2D coordinates (`umap_coords.npy`, `tsne_coords.npy`) and indices (`_sample_idx.npy`) for seamless loading into the Streamlit app.
- Generates static static `.png` files for quick review.

### Tech Stack
- `umap-learn` for manifold approximation
- `scikit-learn` for t-SNE
- `plotly` for interactive HTML generation with hover data (Class, Redshift, ObjID).

---

## 2026-03-20 — Phase 3: Centroid Drift Analysis Run & Results

### What was done
Ran `drift_analysis.py` against the trained `multimodal_efficientnet_b0.pth` checkpoint. Extracted **256-d fused embeddings** for all 47,089 galaxies and computed centroid drift across three redshift bins.

### Bug fixes before running
- `drift_analysis.py` had no tqdm progress bar on the embedding extraction loop, causing it to appear frozen.
- Added `from tqdm import tqdm` (missing import) and wrapped the extraction loop with a progress bar.
- Fixed `--num-workers` default from `2` → `0` (Windows deadlock fix).
- Added `--backbone` flag (`resnet18` / `efficientnet_b0`) so the script loads the correct checkpoint (`multimodal_{backbone}.pth`) and saves outputs to `outputs/drift_{backbone}/`.
- Updated `MultimodalFusionNet` instantiation to pass `backbone=args.backbone`.

### Run command
```powershell
uv run drift_analysis.py --backbone efficientnet_b0
```

### Phase 3 Results (EfficientNet-B0 backbone)

| Class | low_z → mid_z (Euclidean) | mid_z → high_z (Euclidean) |
|---|---|---|
| disk | **3.4280** | 1.5490 |
| smooth | 3.2070 | 1.7446 |
| spiral | 2.8997 | 1.0345 |
| edge_on | 2.4765 | 1.6421 |
| smooth (high_z → ext_z) | 2.2616 | — |

### Key scientific findings
1. **Disk galaxies drift the most** from low→mid redshift (Euclidean: 3.43). This reflects real SDSS resolution bias: at higher redshift, disk galaxies become harder to distinguish from smooth galaxies.
2. **Universal pattern:** Every class drifts more from `low_z→mid_z` than from `mid_z→high_z`. The largest morphological confusion zone is z = 0.06–0.10.
3. **Spiral galaxies are most stable at high-z** (mid→high drift = 1.03, lowest of all), suggesting their distinctive features (arms) remain recognisable longest.
4. **Smooth extended bin confirmed:** Only smooth galaxies had ≥30 galaxies beyond z = 0.14 (2,210 galaxies), validating the collaborator's redshift binning recommendation.

### Outputs
- `outputs/drift_efficientnet_b0/embeddings.npy`
- `outputs/drift_efficientnet_b0/embedding_labels.csv`
- `outputs/drift_efficientnet_b0/centroid_drift.csv`
- `outputs/drift_efficientnet_b0/bin_population.png`
- `outputs/drift_efficientnet_b0/drift_curves.png`

---

## 2026-03-20 — Final Grade Multimodal Training (Differential LR + Label Smoothing)

### What was done
Modified `train_multimodal.py` with two targeted improvements to produce better-quality embeddings for Phase 3.

### Changes made to `train_multimodal.py`

**1. Label Smoothing (one-line change)**
```diff
- criterion = nn.CrossEntropyLoss(weight=class_weights)
+ criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
```
- Prevents the model from over-committing to one class with 100% certainty.
- Results in smoother, better-separated clusters in the 256-d embedding space.
- Note: Loss values will appear higher (~1.0) even when accuracy is good — this is by design.

**2. Differential Learning Rates (split optimizer)**
```diff
- optimizer = torch.optim.Adam(model.get_trainable_params(), lr=args.lr)
+ optimizer = torch.optim.Adam([
+     {"params": model.image_backbone.parameters(), "lr": args.lr * 0.01},
+     {"params": list(model.meta_mlp.parameters()) + list(model.fusion_head.parameters()), "lr": args.lr},
+ ])
```
- Image backbone: `1e-5` (100× slower) — preserves ImageNet pre-trained knowledge.
- Metadata MLP + Fusion Head: `1e-3` (full speed) — fast learning for newly initialised layers.

### Final Grade Training Results
- **Backbone:** EfficientNet-B0, 20 epochs, fine-tuned
- **Best Val Accuracy:** 75.29% (epoch 19)
- **Test Accuracy:** 75.28% — nearly identical to Val (no overfitting)
- **Test F1 (macro):** 0.6737
- **Key observation:** Train Acc (71%) < Val Acc (75%) indicates slight underfitting — model needs more epochs to fully converge. The strict class-weighted loss and label smoothing make the loss appear stuck around ~1.0, but accuracy is meaningfully improving.

---

## 2026-03-20 — Multimodal Model Refactor: Dual Backbone Support

### What was done
Refactored `multimodal_model.py` and `train_multimodal.py` to support both `resnet18` and `efficientnet_b0` as interchangeable image backbones.

### Changes to `multimodal_model.py`
- Added `backbone` argument (`"resnet18"` or `"efficientnet_b0"`).
- Added `_build_image_branch()` helper that selects the correct pretrained network and returns `(backbone, embed_dim)`.
- Fusion head input dimension now **auto-scales**: `embed_dim + 128` (640 for ResNet-18, 1408 for EfficientNet-B0).
- All other interfaces (`forward`, `get_embedding`, `count_params`, `get_trainable_params`) unchanged.
- Self-test confirms both backbones pass forward pass `[2,4]` and embedding shape `[2,256]` checks.

### Changes to `train_multimodal.py`
- Added `--backbone` CLI flag (`choices=["resnet18", "efficientnet_b0"]`, default `"resnet18"`).
- Added `--fine-tune` flag — passes `freeze_backbone=not args.fine_tune` to the model.
- Fixed `total_mem` → `total_memory` bug (same bug fixed earlier in `train_baseline.py`).
- Fixed `--num-workers` default from `2` → `0` (Windows deadlock prevention).
- Added `tqdm` progress bars to training and evaluation loops.
- Added `ReduceLROnPlateau` scheduler stepped on `val_acc`.
- Checkpoint and output paths are now backbone-specific:
  - `checkpoints/multimodal_{backbone}.pth`
  - `outputs/multimodal_{backbone}/`

---

## 2026-03-20 — EfficientNet-B0 Model & Training Script

### What was done
Created two new separate files for EfficientNet-B0, keeping the existing ResNet-18 pipeline untouched.

### New files
**`efficientnet_model.py`**
- `EfficientNetB0Classifier` class mirroring the `BaselineResNet18` interface.
- Loaded from `models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)`.
- Replaces `net.classifier[1]` with `Linear(1280, 4)`.
- Supports `freeze_backbone`, `get_embedding()`, `count_params()`.
- Total: ~4.0M params. Self-test passed.

**`train_efficientnet.py`**
- Standalone training script saving to `checkpoints/efficientnet_b0.pth` and `outputs/efficientnet/`.
- Inherits all improvements from `train_baseline.py` (fine-tune, scheduler, tqdm).

### EfficientNet-B0 results (image-only, fine-tuned, 20 epochs)
- **Test Accuracy: 78.8%** — outperformed ResNet-18 by ~2%.
- Notably better at detecting the "disk" class (recall improved from 0.40 → 0.57 in multimodal run).

---

## 2026-03-20 — ResNet-18 Baseline Improvements

### What was done
Improved the frozen baseline model with three targeted changes.

### Changes to `dataset.py`
- Increased `RandomRotation` from 30° → **90°** (galaxies have no preferred orientation).
- Added `RandomResizedCrop(224, scale=(0.8, 1.0))` for more view-angle diversity.

### Changes to `train_baseline.py`
- Added **`--fine-tune` flag**: when set, unfreezes the entire ResNet-18 backbone for end-to-end training.
- Added **`ReduceLROnPlateau` scheduler** (`mode="max"`, `factor=0.5`, `patience=2`) stepped on `val_acc`.
- Fixed **`total_mem` → `total_memory`** typo in `get_device()` (was causing `AttributeError`).
- Set **`num_workers` default to `0`** (prevents Windows DataLoader deadlocks).
- Added **`tqdm` progress bars** to training and evaluation loops.
- Added **`--fine-tune`** flag to `parse_args`.

### ResNet-18 results (fine-tuned, 20 epochs)
- **Test Accuracy: 76.7%** — up from ~63% with frozen backbone.

---

## 2026-03-20 — GPU / CUDA Environment Setup

### What was done
Fixed PyTorch installation to use CUDA acceleration instead of CPU-only.

### Problem
`torch.cuda.is_available()` returned `False` despite an NVIDIA GPU being present. The installed PyTorch was the CPU-only build.

### Fix
Updated `pyproject.toml` to add the CUDA 12.8 PyPI index:
```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```
And added explicit index markers to the torch/torchvision dependencies. Ran `uv sync` to reinstall with GPU support.

### Result
`torch.cuda.is_available()` → `True`. All subsequent training runs used CUDA automatically.

---



## 2026-03-19 — Phase 3: Embedding & Drift Analysis (`drift_analysis.py`)

### What was done
Built `drift_analysis.py` — the script that extracts embeddings from the trained model and computes how galaxy morphology representations shift across cosmic time.

### How it works
1. Loads trained multimodal model from `checkpoints/multimodal_fusion.pth`
2. Feeds ALL 47,089 galaxies through the model to extract 256-d fused embeddings
3. Bins galaxies into **3 science-driven redshift bins** (per collaborator):
   - `low_z`: z = 0.01 – 0.06 (all classes)
   - `mid_z`: z = 0.06 – 0.10 (all classes)
   - `high_z`: z = 0.10 – 0.14 (all classes, marginal for disk/edge_on)
   - `ext_z`: z = 0.14 – 0.25 (**smooth only** — other classes lack statistical validity due to SDSS resolution effects)
4. Computes mean centroid per class per bin
5. Measures **Euclidean** and **Cosine** drift between adjacent bin centroids
6. Enforces minimum 30 galaxies per bin/class for statistical validity

### Key design decisions
- **Why 3 bins + 1 extended?** Collaborator identified that spiral, disk, and edge_on vanish above z ≈ 0.13 due to SDSS resolution limits. Reporting drift for empty bins would be scientifically meaningless.
- **Why both distance metrics?** Euclidean captures magnitude of shift. Cosine captures directional change in embedding space. Both provide complementary insight.
- **Minimum threshold (30):** Bins with fewer galaxies produce noisy centroids. Invalid bins are flagged and excluded from drift curves.

### How to run
```bash
# After train_multimodal.py has completed:
python drift_analysis.py
python drift_analysis.py --batch-size 64 --device cuda
```

### Outputs
- `outputs/drift/embeddings.npy` — Raw 256-d embeddings for all galaxies
- `outputs/drift/embedding_labels.csv` — Per-row metadata (objid, class, redshift)
- `outputs/drift/centroid_drift.csv` — Drift values per class per bin transition
- `outputs/drift/bin_population.png` — Population visualization
- `outputs/drift/drift_curves.png` — Euclidean + Cosine drift curves

## 2026-03-19 — Multimodal Fusion Model (Phase 2)

### What was done
Built all 3 Phase 2 files. **Training NOT run** — user will hand off to someone with GPU access.

### Files created

**`dataset_multimodal.py`**
- Extends the image-only dataset to return `(image, metadata, label)` tuples
- Metadata features (6 total): `redshift`, `color_u_r`, `color_g_i`, `fracdev_r`, `petror50_r_kpc`, `mu50_r`
- Z-score normalization computed from **training set only** (prevents data leakage)
- Stats saved to `checkpoints/metadata_stats.pth` for reuse in Phase 3

**`multimodal_model.py`**
- **Image branch:** Frozen ResNet18 → 512-d embedding
- **Metadata branch:** MLP `[6 → 64 → 128]` with BatchNorm + ReLU + Dropout
- **Fusion:** `concat(512, 128)` = 640 → `Linear(640, 256)` → ReLU → `Linear(256, 4)`
- The **256-d layer** is the fused embedding used for Phase 3 drift analysis
- Total: 11,351,300 params | Trainable: **174,788 (1.5%)**

**`train_multimodal.py`**
- Same structure as `train_baseline.py` but handles multimodal data
- Automatically generates a **baseline vs multimodal comparison plot** if baseline was trained first
- Saves metadata normalization stats alongside model checkpoint

### How to run
```bash
# IMPORTANT: Run train_baseline.py FIRST, then:
python train_multimodal.py
python train_multimodal.py --epochs 30 --batch-size 64
python train_multimodal.py --num-workers 0  # Windows fix
```

### Self-test results (CPU, no training)
- Dataset: returns `(image[3,224,224], metadata[6], label)` ✅
- Forward pass: `[2,3,224,224]` + `[2,6]` → logits `[2,4]` ✅
- Embedding: `[2,3,224,224]` + `[2,6]` → `[2,256]` ✅

## 2026-03-19 — Baseline Model & Training Script (`baseline_model.py`, `train_baseline.py`)

### What was done
Built the ResNet18 baseline architecture and the full training pipeline. **Training was NOT run** — the user will hand these files to someone with GPU access.

### `baseline_model.py`
- ResNet18 pretrained on ImageNet with **all backbone layers frozen**
- Only the final classification head is trainable: `Linear(512, 4)`
- Total params: 11,176,512 | Trainable: **2,052 (0.02%)**
- Includes `get_embedding()` method for Phase 3 (extracts 512-d vector before classifier)

### `train_baseline.py`
- **Auto-detects GPU/CPU** — works on any machine
- Uses **class-weighted cross-entropy** (balanced weights from `dataset.py`)
- Saves best model checkpoint to `checkpoints/baseline_resnet18.pth`
- Generates training curves and confusion matrix automatically
- CLI args: `--epochs`, `--batch-size`, `--lr`, `--device`, `--num-workers`

### How to run
```bash
# Default (20 epochs, batch=32, auto GPU/CPU)
python train_baseline.py

# Custom
python train_baseline.py --epochs 30 --batch-size 64

# Force CPU (if GPU has issues)
python train_baseline.py --device cpu

# Windows fix if DataLoader crashes
python train_baseline.py --num-workers 0
```

### Self-test results (CPU, no training)
- Forward pass: input `[2, 3, 224, 224]` → output `[2, 4]` ✅
- Embedding: input `[2, 3, 224, 224]` → output `[2, 512]` ✅

## 2026-03-19 — PyTorch Dataset Module (`dataset.py`)

### What was done
Created `dataset.py` — the reusable data loading module for all training scripts.

### Components
1. **`GalaxyDataset`** — Custom PyTorch Dataset. Loads a galaxy JPEG, applies transforms, returns `(image_tensor, label_int)`.
2. **`get_transforms(split)`** — Train split gets augmentations (random flip, rotation, color jitter). Val/Test get resize + normalize only. All use ImageNet stats (required for pretrained ResNet).
3. **`create_splits()`** — Stratified 70/15/15 split. Stratification ensures each split has identical class proportions.
4. **`get_class_weights()`** — Uses `sklearn.compute_class_weight('balanced')` to produce inverse-frequency weights per collaborator's recommendation.

### Self-test results
- Train: 32,962 | Val: 7,063 | Test: 7,064
- Class weights: disk=3.09, edge_on=3.16, smooth=0.56, spiral=0.64
- Image tensor shape: `[3, 224, 224]`, pixel range: `[-2.12, 1.50]` (ImageNet-normalized)
- All class proportions identical across splits ✅

### Design decisions
- **Class ordering:** Alphabetical (`disk=0, edge_on=1, smooth=2, spiral=3`) for reproducibility.
- **Augmentations:** Horizontal/vertical flip + rotation + color jitter. Galaxies have no preferred orientation, so flips are physically valid.
- **No random crop:** Galaxy images are already centered cutouts — cropping risks cutting off the galaxy itself.

## 2026-03-18 — Collaborator Feedback: Class Imbalance & Redshift Binning

### Source
External collaborator with domain expertise reviewed EDA plots.

### Feedback 1: Class Imbalance (Action Required in Phase 1)

The 50k sample is **representative** of the full 239k catalog — proportions match almost exactly. The ~3k dropped as "uncertain" (5.8%) is expected.

**However**, the 5:1 ratio between dominant and minority classes is a real concern:
- smooth + spiral = 39,553 (84%)
- disk + edge_on = 7,536 (16%)

**Decision:** Use `sklearn.utils.class_weight.compute_class_weight('balanced')` to compute inverse-frequency class weights and pass them to `CrossEntropyLoss`. This is critical because disk/edge_on embeddings must be well-learned for meaningful drift analysis in Phase 3.

### Feedback 2: Redshift Binning (Action Required in Phase 3)

The redshift histogram revealed a **known SDSS resolution effect**: spiral, disk, and edge_on galaxies essentially vanish above z ≈ 0.13. At higher redshifts, features become unresolvable and volunteers classify everything as smooth. Hart et al.'s debiasing helps but doesn't fully eliminate this.

**Decision:** Use **3 analysis bins** instead of 4 quantile bins:

| Bin | Range | All classes populated? |
|---|---|---|
| Low-z | z = 0.01 – 0.06 | ✅ Yes |
| Mid-z | z = 0.06 – 0.10 | ✅ Yes |
| High-z | z = 0.10 – 0.14 | ✅ Marginal for disk/edge_on |

Beyond z = 0.14, drift should **only be reported for smooth**. Other classes lack statistical validity at those distances.

### Impact
- Phase 1: Add class-weighted loss (one-liner fix)
- Phase 3: Change from 4 quantile bins → 3 science-driven bins with per-class validity checks

## 2026-03-18 — Exploratory Data Analysis (`eda.py`)

### What was done
Created `eda.py` and ran it to generate 5 diagnostic plots in `outputs/eda/`.

### Plots generated
1. **class_distribution.png** — Bar chart confirming class imbalance: smooth (45%) and spiral (39%) dominate, with disk (8.1%) and edge_on (7.9%) as minority classes.
2. **redshift_per_class.png** — All 4 classes span the full redshift range [0.01, 0.25], with peak density around z ≈ 0.04–0.08. Smooth galaxies have a slightly broader tail to higher redshifts.
3. **sample_images.png** — Visual sanity check confirms labels match visual morphology: smooth = featureless blobs, spiral = visible arms, disk = face-on disks, edge_on = elongated profiles.
4. **color_redshift.png** — Extinction-corrected u−r and g−i colors show expected trends: smooth (red) galaxies tend redder, spirals tend bluer. Colors tighten with increasing redshift.
5. **structural_features.png** — FRACDEV_R cleanly separates smooth (high values) from disk/spiral (low values), confirming it's a strong morphology proxy.

### Key observations
- **Class imbalance:** disk and edge_on are ~5× smaller than smooth/spiral. May need weighted sampling or class-weighted loss during training.
- **Redshift coverage:** All classes well-represented across z range. 4 quantile bins will work well.
- **Data looks clean:** No obvious anomalies in images, colors, or structural features.

## 2026-03-18 — Master Dataset Built (`prepare_dataset.py`)

### What was done
Created `prepare_dataset.py` and ran it to produce `galaxy_master_dataset.csv`.

### Results
- **47,089 usable galaxies** with images, labels, metadata, and clean photometry
- **17 columns**, **zero null values**
- Morphology distribution:
  - `smooth`: 21,184 (45.0%)
  - `spiral`: 18,369 (39.0%)
  - `disk`: 3,815 (8.1%)
  - `edge_on`: 3,721 (7.9%)

### Pipeline steps (in order)
1. Loaded 239,695 morphology labels from `gz2_hart16.csv`
2. Filtered `gz2_filename_mapping.csv` to "original" sample only (245,609 → used for matching)
3. Loaded 325,704 metadata rows from `gz2sample.csv`
4. Dropped 59 rows with bad photometry (magnitude > 30)
5. Computed extinction-corrected colors (`color_u_r`, `color_g_i`)
6. Inner-joined labels + metadata = 239,638 galaxies
7. Inner-joined + image mapping = 209,251 galaxies
8. Filtered to only those with images on disk = 49,990 galaxies
9. Assigned morphology classes from vote fractions
10. Removed "artifact" and "uncertain" classes → **47,089 final galaxies**

### Final columns
`dr7objid`, `image_path`, `morph_class`, `ra`, `dec`, `redshift`, `redshifterr`, `petromag_u/g/r/i/z`, `color_u_r`, `color_g_i`, `fracdev_r`, `petror50_r_kpc`, `mu50_r`


---

## 2026-03-18 — Image Extraction Strategy

### What was done
Extracted exactly **50,000 galaxy images** from the full `images_gz2.zip` (Zenodo, 243,434 total images) into `images_gz2/`.

### How images were selected
Images were **NOT randomly selected**. The extraction script (`extract_images.py`) used a **priority-based strategy**:

1. Loaded `gz2_hart16.csv` (morphology labels) and `gz2sample.csv` (metadata with redshift + photometry).
2. Computed the intersection of galaxy IDs present in **both** catalogs — these are galaxies that have labels AND metadata, making them usable in the full pipeline.
3. Used `gz2_filename_mapping.csv` to map those galaxy IDs (`dr7objid`) to image filenames (`asset_id.jpg`).
4. Extracted priority images first (those with both labels + metadata), then filled remaining slots with other images.

### Why 50,000?
- The full 243k dataset is unnecessarily large for fine-tuning a pretrained ResNet18 (10k–20k is often sufficient).
- 50k gives ~3,000+ images per group when split across ~4 redshift bins × ~4 morphology classes, which is statistically robust for centroid drift analysis.
- Keeps disk usage and training time manageable.

### Why not random?
Random selection risks including images that have no matching labels or metadata, making them useless for training. Priority selection maximizes the number of usable galaxies in the master dataset.

---

## 2026-03-18 — Data Improvements Incorporated

### Source
Recommendations received from external reviewer with SDSS expertise.

### Changes adopted (all 5)

**1. Extinction-corrected colors**
- Instead of using raw `PETROMAG_*` values for color computation, we now correct for Galactic dust using the `EXTINCTION_*` columns from `gz2sample.csv`.
- Formula: `color_u_r = (PETROMAG_U - EXTINCTION_U) - (PETROMAG_R - EXTINCTION_R)`
- Rationale: Galactic dust reddens light unevenly across bands. Without correction, observed colors are contaminated by our galaxy's dust, not the target galaxy's properties.

**2. Three extra structural features added to metadata**
- `FRACDEV_R`: Fraction of light from de Vaucouleurs (elliptical) profile. 0 = pure disk, 1 = pure elliptical.
- `PETROR50_R_KPC`: Half-light radius in kiloparsecs (physical galaxy size).
- `MU50_R`: Surface brightness at half-light radius (structural compactness).
- Rationale: These encode structural information complementary to color, making the metadata MLP more powerful.

**3. Filter `gz2_filename_mapping.csv` to "original" sample only**
- The mapping file contains 5 sub-samples: `original`, `stripe82`, `stripe82_coadd_1`, `stripe82_coadd_2`, `extra`.
- Only the `original` sample (245,609 rows) maps to the standard GZ2 images.
- Others are repeat/deep observations of the same galaxies — keeping them would create duplicate entries.

**4. Use 4 quantile-based redshift bins (not 5 equal-width)**
- With 5 equal-width bins, the highest bin (z=0.20–0.25) has only ~1,391 galaxies.
- 4 quantile bins ensure roughly equal population per bin, which produces more reliable centroid statistics.

**5. Drop bad photometry rows (magnitude > 30)**
- SDSS uses values >30 as sentinel for failed photometric measurements.
- Only 59 rows affected — trivial to drop, but leaving them would corrupt color features.

---

## 2026-02-18 — Metadata Dataset Decision

### Decision
Use `gz2sample.csv` instead of `zoo2MainSpecz.csv` for the SDSS DR7 metadata (redshift + photometry).

### Why
| Criteria | `zoo2MainSpecz.csv` | `gz2sample.csv` |
|----------|---------------------|-----------------|
| Redshift values | ❌ Missing | ✅ `REDSHIFT` column |
| Photometry (magnitudes) | ❌ Missing | ✅ `PETROMAG_U/G/R/I/Z` |
| Overlap with labels | 239,639 | **239,695 (100%)** |
| CasJobs query needed? | Yes | **No** |

### Impact
Eliminated the need for manual SDSS CasJobs SQL queries. All metadata is now available in a single CSV file.

---

## 2026-02-18 — Project Initialization

### Datasets acquired
1. **Galaxy Zoo 2 Images** — `images_gz2.zip` from Zenodo (243,434 SDSS cutouts)
2. **GZ2 Debiased Morphology Catalog** — `gz2_hart16.csv` (Hart et al. 2016, ~239,695 galaxies)
3. **Image Filename Mapping** — `gz2_filename_mapping.csv` from Zenodo
4. **SDSS Metadata** — `gz2sample.csv` (redshift + photometry for 325,704 galaxies)

### Project goal
Study how galaxy morphology changes across cosmic time using population-level latent embedding drift, not individual galaxy prediction.

### Pipeline overview
1. Data integration → master dataset
2. Baseline image classifier (ResNet18, frozen backbone)
3. Multimodal fusion model (ResNet18 + Metadata MLP)
4. Embedding extraction + redshift-binned centroid drift analysis
5. Visualization (t-SNE/UMAP, drift curves)
