# Galaxy Morphology Evolution — Collaborator Handoff Guide

This guide explains exactly what files you need, what scripts to run, and the order to run them in to execute the machine learning pipeline for this project.

## 📂 1. Files You Need (The "Payload")

The original raw datasets (~3.5 GB of ZIP files and CSVs) have already been processed. **You do NOT need to run any data extraction or preparation scripts.** 

You only need to transfer the following items to your machine:

1. **`galaxy_master_dataset.csv`** 
   - *Why you need it:* This is the final, pre-processed "source of truth". It contains exactly 47,089 rows, with matched labels, clean photometry, and extinction-corrected colors. The PyTorch data loaders read directly from this file.
2. **`images_gz2/` (Folder)**
   - *Why you need it:* Contains the 50,000 cropped galaxy JPEG images that perfectly map to the rows in the master CSV. 
3. **The ML Python Scripts**
   - `dataset.py`, `baseline_model.py`, `train_baseline.py`
   - `dataset_multimodal.py`, `multimodal_model.py`, `train_multimodal.py`
4. **`requirements.txt`**

---

## ⚙️ 2. Environment Setup

Before running anything, ensure your environment has the required packages.

```bash
# If using a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies:
pip install -r requirements.txt
```

---

## 🚀 3. Execution Order

The scripts must be run in the following exact order. The scripts will automatically detect your GPU if you have CUDA installed, otherwise they will gracefully fall back to CPU.

### Step 1: Train the Baseline Model
**Command:**
```bash
python train_baseline.py --epochs 20 --batch-size 32
```
*(Note: If you are on Windows and the DataLoader crashes, add `--num-workers 0` to the command).*

**Why you are running this:**
This script trains an image-only ResNet18 (with a frozen backbone) using the galaxy images. We use inverse-frequency class-weighted loss to ensure minority classes (like edge-on disks) are learned equally well. 

**What it produces:**
- Best checkpoint saved to `checkpoints/baseline_resnet18.pth`
- A documented baseline performance metric to prove that adding metadata (in Step 2) actually improves the model.
- Training curves and a confusion matrix in `outputs/baseline/`.

### Step 2: Train the Multimodal Model
**Command:**
```bash
python train_multimodal.py --epochs 20 --batch-size 32
```

**Why you are running this:**
This script trains the true Fusion model. It takes the frozen ResNet18 image embeddings and concatenates them with a 128-d output from a Metadata MLP (which processes the galaxy's redshift, color, and structural properties). 

**What it produces:**
- Computes and saves z-score normalization stats from the training set to `checkpoints/metadata_stats.pth`.
- Best checkpoint saved to `checkpoints/multimodal_fusion.pth`.
- Generates `outputs/multimodal/baseline_comparison.png` — a direct visual comparison proving the performance difference between the Baseline and Multimodal approaches.

---

## 🎯 Next Steps After Training

### Step 3: Run Drift Analysis *(after Step 2 completes)*
**Command:**
```bash
python drift_analysis.py --batch-size 64
```

**Why you are running this:**
This is the core scientific analysis. It extracts 256-dimensional embeddings from each galaxy using the trained model, groups them into redshift bins (which correspond to different cosmic epochs), and measures how the morphological "fingerprint" of each class shifts between bins. This centroid drift is the key metric the project was designed to measure.

**What it produces:**
- `outputs/drift/embeddings.npy` — Raw embeddings for all 47k galaxies (used for Phase 4 t-SNE/UMAP)
- `outputs/drift/centroid_drift.csv` — The final drift measurements
- `outputs/drift/drift_curves.png` — Visualization of how each class drifts across cosmic time
- `outputs/drift/bin_population.png` — Galaxy counts per class per redshift bin

**Important Notes:**
- This script uses **3 science-driven redshift bins** (not equal-width): 0.01–0.06, 0.06–0.10, 0.10–0.14
- An extended 4th bin (0.14–0.25) is only computed for **smooth galaxies** — other classes don't have enough galaxies beyond z=0.14 to be meaningful
- Any bin with fewer than 30 galaxies is automatically flagged as invalid

Once this finishes, Phases 1–3 are fully complete. Hand the `outputs/` folder back to proceed with Phase 4 (t-SNE/UMAP visualization + interpretation).
