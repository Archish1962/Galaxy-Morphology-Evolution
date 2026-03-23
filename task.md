# Galaxy Morphology Evolution - Project Roadmap

## Phase 0: Data Integration & Preprocessing
- [x] Establish directory structure <!-- id: 1 -->
- [x] Create master integration script ([prepare_dataset.py](file:///c:/Files/Coding/Projects/Data%20Science%20Projecr/prepare_dataset.py)) <!-- id: 2 -->
    - [x] Select [gz2sample.csv](file:///c:/Files/Coding/Projects/Data%20Science%20Projecr/gz2sample.csv) as metadata source <!-- id: 3 -->
    - [x] Merge labels, mapping, and metadata into [galaxy_master_dataset.csv](file:///c:/Files/Coding/Projects/Data%20Science%20Projecr/galaxy_master_dataset.csv) <!-- id: 4 -->
    - [x] Calculate derived features (extinction-corrected colors, structural features) <!-- id: 5 -->
- [x] Image verification: Ensure all IDs in master have valid image files <!-- id: 6 -->
- [x] Extract 50k priority images from ZIP <!-- id: 6b -->

## Phase 0.5: Exploratory Data Analysis
- [x] Create [eda.py](file:///c:/Files/Coding/Projects/Data%20Science%20Projecr/eda.py) <!-- id: 22 -->
    - [x] Morphology class distribution (bar chart) <!-- id: 23 -->
    - [x] Redshift distribution per morphology class <!-- id: 24 -->
    - [x] Sample galaxy images per class (grid) <!-- id: 25 -->
    - [x] Color-redshift scatter plots <!-- id: 26 -->

## Phase 1: Baseline Image Classifier
- [x] Implement PyTorch Dataset class ([dataset.py](file:///c:/Files/Coding/Projects/Data%20Science%20Projecr/dataset.py)) <!-- id: 7 -->
    - [x] Image transforms (resize, normalize, augment) <!-- id: 7a -->
    - [x] Stratified train/val/test split <!-- id: 7b -->
- [x] Build ResNet18 baseline model ([baseline_model.py](file:///c:/Files/Coding/Projects/Data%20Science%20Projecr/baseline_model.py)) <!-- id: 8 -->
- [x] Training loop with class-weighted loss ([train_baseline.py](file:///c:/Files/Coding/Projects/Data%20Science%20Projecr/train_baseline.py)) <!-- id: 9 -->
- [x] Evaluate baseline performance (Accuracy, F1, Confusion Matrix) <!-- id: 10 -->

## Phase 2: Multimodal Fusion Model
- [x] Implement Metadata MLP (Input: Redshift + Colors + Structural) <!-- id: 11 -->
- [x] Create Fusion Network (Concat Image + Metadata embeddings) <!-- id: 12 -->
- [x] Train Multimodal model with class-weighted loss ([train_multimodal.py](file:///c:/Files/Coding/Projects/Data%20Science%20Projecr/train_multimodal.py)) <!-- id: 13 -->
- [x] Compare performance vs Baseline <!-- id: 14 -->

## Phase 3: Embedding & Drift Analysis
- [ ] Extract latent embeddings for entire dataset <!-- id: 15 -->
- [ ] Implement 3 science-driven redshift bins (0.01–0.06, 0.06–0.10, 0.10–0.14) <!-- id: 16 -->
- [ ] Compute Centroids per morphology class per bin <!-- id: 17 -->
- [ ] Calculate Centroid Drift (report >z=0.14 only for smooth) <!-- id: 18 -->

## Phase 4: Visualization & Reporting
- [ ] Generate t-SNE/UMAP visualizations of embeddings <!-- id: 19 -->
- [ ] Plot Centroid Drift curves per morphology <!-- id: 20 -->
- [ ] Final interpretation of evolutionary trends <!-- id: 21 -->
