# Estimating Human Ratings from Face Images (Classical Machine Learning)
**Course:** AIP301 Machine Learning Lab · **Submitted to:** Dr. Shibna Hussain · **Institution:** IIIT Kota  
**Team Name:** Neural Ninjas (Batch A3)  
**Hardware:** NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM, CUDA) + AMD Radeon 780M (OpenCL) + 16 CPU Threads  
**Core Constraint:** Pure Classical Machine Learning & Computer Vision (Zero CNNs, Zero Pre-trained Deep Models, Zero Transfer Learning).

---

## 1. Project Overview & Verdict
This repository implements an end-to-end, reproducible, leakage-controlled classical machine learning pipeline to estimate human consensus ratings on facial photographs from the SCUT-FBP5500 benchmark dataset.

On an unseen, group-disjoint test partition ($n = 818$), our 3-pipeline ensemble achieves:
$$\text{MAE} = \mathbf{0.3272}, \quad \text{RMSE} = \mathbf{0.4243}, \quad \text{Pearson } r = \mathbf{0.8033}, \quad R^2 = \mathbf{0.6272}$$
This matches **$86.6\%$ of the theoretical rater-agreement ceiling ($r = 0.927$)** derived via split-half reliability and the Spearman-Brown prophecy formula.

---

## 2. Theoretical & Engineering Foundations

### What HOG Means in Face Detection
In facial analysis, **Histogram of Oriented Gradients (HOG)** divides the face into small spatial grid cells ($8 \times 8$ pixels) and computes the local gradient orientation and magnitude:
- **What it captures:** Sharp intensity transitions defining facial contours—the jawline, nasal ridge, eye boundaries, and eyebrow arches—while being invariant to uniform lighting shifts.
- **Why the dimension is 8,000+:** A $128 \times 128$ image produces $(15 \times 15)$ overlapping blocks with $2 \times 2$ cells and 9 orientation bins each, yielding $(15 \times 15) \times 4 \times 9 = \mathbf{8,100}$ features. This provides a detailed, non-learned structural representation.

### Why PCA with 100 Features Isn't "Too Much"
Retaining 100 principal components is standard, mathematically compact, and avoids the curse of dimensionality:
- **98.8% Dimensional Compression:** Compressing 8,458 combined features down to 100 components is a **$98.8\%$ reduction** in dimensionality.
- **Variance Retention:** The top 100 components capture **$> 88\%$ of total variance** across aligned human portraits.
- **Healthy Sample-to-Feature Ratio:** With $n_{\text{train}} \approx 3,300$ samples, $p = 100$ yields $\frac{n}{p} \approx 33$ samples per feature, guaranteeing a well-conditioned empirical covariance matrix and preventing double-descent interpolation spikes.

### The Duplicate Percolation Discovery
Comparing 5,500 images pairwise requires $\approx 15,122,250$ comparisons.
- **Perceptual Compression (dHash):** Each photo is compressed to a $9 \times 8$ grayscale matrix (72 pixels) and converted to a 64-bit difference hash.
- **Bitwise Hamming Matching:** Pairwise comparison requires only a single XOR and population count (`popcount`) operation.
- **The Percolation Catastrophe at Threshold 6:** A standard Hamming threshold ($\le 6$ of 64 bits) triggered catastrophic graph percolation—**$75.5\%$ of the dataset (4,150 images) merged into duplicate clusters**, with one giant cluster containing 1,620 images! Because all photos are frontal portraits with studio backgrounds, transitive chaining via Union-Find merged unrelated faces.
- **The Solution:** We enforced **threshold 0**, isolating 39 true duplicate clusters (88 images), and physically moved **49 redundant duplicate files** into `duplicate/`, producing `labels_clean.txt` with **5,451 strictly unique images**.

---

## 3. Project File Structure & Numbered Running Order

The codebase is organized into sequential, numbered scripts that can be run step-by-step:

```
neural_ninja/
│
├── 01_dataset_audit_and_dedup.py     # STEP 1: Audit data, detect percolation, create disjoint splits
├── 02_extract_features.py            # STEP 2: Multi-core extraction of HOG, LBP, COL, EDGE, GEO
├── 03_train_models_and_cv.py         # STEP 3: Repeated 15-fold group CV, train B0, B1, P1, P2, P3
├── 04_run_experiments_and_studies.py # STEP 4: PCA vs KMeans, ratios, dimension study, fairness audits
├── 05_predict_face.py                # STEP 5: Standalone inference CLI with uncertainty intervals
│
├── config.py                         # Hardware probes (RTX 5050 / Radeon 780M / RAM), paths, checkpoints
├── features.py                       # Classical feature extraction functions & OpenCV Haar cascade loaders
├── models.py                         # Estimators: RidgeCV, HistGB, and From-Scratch PCA+MLP with Adam
│
├── Images/                           # 5,451 unique facial portrait photographs
├── duplicate/                        # 49 physically isolated duplicate images + duplicate_mapping.txt
├── labels.txt                        # Original label records
├── labels_clean.txt                  # Cleaned label records for 5,451 unique images
│
├── out/                              # Generated figures, CSV metric tables, and bundle.joblib
│   ├── task1_audit.json              # Dataset audit statistics
│   ├── task3_feature_table.csv       # Feature economics (dimensions, latency, storage)
│   ├── task8_cv_summary.csv          # 15-fold cross-validation performance comparison
│   ├── test_results.csv              # Final held-out test evaluation
│   ├── bundle.joblib                 # Serialized production inference models & uncertainty bounds
│   └── *.png                         # 11 analytical plots (target distribution, error gallery, etc.)
│
├── result/                           # Concisely formatted academic reports & assignment answers
│   ├── FINAL_PROJECT_REPORT.md       # Master lab report answering all 11 assignment tasks
│   ├── STUDENT_1_P1_APPEARANCE.md    # Student 1 report: Appearance (HOG + LBP + COL -> Ridge)
│   ├── STUDENT_2_P2_GEOMETRY.md      # Student 2 report: Facial Ratios & Symmetry -> HistGB
│   ├── STUDENT_3_P3_FUSION_SCRATCH.md# Student 3 report: From-Scratch PCA + Backprop MLP + Adam
│   └── STUDENT_4_AUDITOR_ETHICIST.md # Student 4 report: Audit, Percolation, Zero Leakage, Fairness
│
└── requirements.txt                  # Python dependencies
```

---

## 4. Hardware Acceleration Architecture

Optimized for local execution on modern mixed hardware:
- **NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM, CUDA):** Accelerates matrix multiplications and OpenCV image operations.
- **AMD Radeon 780M (OpenCL):** Automatically enables OpenCL acceleration (`cv2.ocl.setUseOpenCL(True)`) for parallel color conversions and resizing.
- **System RAM Management (~12-16 GB limit):**
  - Features stored in `float32` arrays, cutting memory footprint by $50\%$ compared to standard `float64`.
  - Image feature extraction batches in chunks of 32 images across CPU threads.
  - Intermediate checkpoints saved to disk to allow memory garbage collection between stages.

---

## 5. Execution Guide

### Step 1: Environment Setup

```bash
# Using Python 3.10 to 3.15
python -m venv .venv
.venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### Step 2: Download & Extract Dataset (One-Command Setup)

The dataset is too large to commit to Git and is hosted on Kaggle:
- **Kaggle Dataset Link:** [SCUT-FBP5500 v2 Facial Beauty Scores](https://www.kaggle.com/datasets/pranavchandane/scut-fbp5500-v2-facial-beauty-scores) (`pranavchandane/scut-fbp5500-v2-facial-beauty-scores`)

Run our automatic downloader script, which uses `kagglehub` to download the dataset and unpacks the images directly into `./Images/` (flattening any nested folders):

```bash
python download_dataset.py
```

### Step 3: Running the Numbered Pipeline

```bash
# STEP 1: Audit dataset and establish group-disjoint splits
python 01_dataset_audit_and_dedup.py --data_dir Images --labels labels.txt --out out

# STEP 2: Extract visual feature representations (HOG, LBP, COL, EDGE, GEO)
python 02_extract_features.py --data_dir Images --labels labels.txt --out out

# STEP 3: Train candidate models and run 15-fold repeated group cross-validation
python 03_train_models_and_cv.py --out out

# STEP 4: Run academic studies (PCA vs KMeans, Dimension Study, Fairness Audits)
python 04_run_experiments_and_studies.py --out out

# STEP 5: Run single-image inference with uncertainty bounds
python 05_predict_face.py Images/AF1.jpg --bundle out/bundle.joblib
```

---

## 6. Git & GitHub Push Instructions

The repository is configured with `.gitignore` to exclude `.venv/`, `Images/` (> 1 GB), `checkpoints/`, and temporary caches:

```bash
# Initialize and stage repository files
git add .

# Create initial commit
git commit -m "feat: classical ML pipeline for SCUT-FBP5500 face rating estimation"

# Set default branch to main
git branch -M main

# Link to your GitHub remote repository
git remote add origin https://github.com/<YOUR-USERNAME>/<YOUR-REPO-NAME>.git

# Push to GitHub
git push -u origin main
```

---

## 6. Empirical Performance Summary

### Repeated Group Cross-Validation (15 Folds on Development Data, $n = 4,633$)

| Model | Representation | Dimension ($p$) | CV MAE | CV RMSE ($\pm \sigma$) | Pearson $r$ | Error vs. Baseline B0 | Fit Time | Model Size |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0 (Baseline)** | Dummy (Train Mean) | 0 | $0.5681$ | $0.6875 \pm 0.0109$ | $0.0000$ | $0.0\%$ | 0.001 s | 1 KB |
| **B1 (Baseline)** | Raw 32×32 Pixels + Ridge | 1,024 | $0.4485$ | $0.5745 \pm 0.0152$ | $0.5490$ | $-16.4\%$ | 1.650 s | 30 KB |
| **P1 (Student 1)** | Appearance (HOG+LBP+COL $\to$ Ridge) | 8,284 | **$0.3436$** | **$0.4477 \pm 0.0104$** | **$0.7590$** | **$-34.9\%$** | 7.958 s | 233 KB |
| **P2 (Student 2)** | Geometry (GEO+EDGE $\to$ HistGB) | 174 | $0.3681$ | $0.4752 \pm 0.0127$ | $0.7250$ | $-30.9\%$ | 2.734 s | 3.68 MB |
| **P3 (Student 3)** | From-Scratch PCA + MLP (Adam) | $8,458 \to 100$ | $0.3561$ | $0.4640 \pm 0.0123$ | $0.7390$ | $-32.5\%$ | **1.156 s** | 185 KB |
| **Test Set** | **Ensemble Mean (P1 + P2 + P3)** | — | **0.3272** | **0.4243** | **0.8033** | **$-38.3\%$** | — | — |

### Substantial Original Code Requirement (Student 3)
- Custom Randomized SVD PCA implemented in pure NumPy.
- Custom feedforward neural network ($100 \to 64 \to 1$) with manual backpropagation and Adam optimizer.
- Numerical gradient checking confirmed analytical backpropagation gradients to **$4.05 \times 10^{-9}$ relative error**.

---

## 7. Key Findings & Ethical Disclaimer

1. **Optical Blur:** A Mann-Whitney U test confirmed that blur statistically significantly degrades model accuracy ($p = 0.0446$).
2. **Regression to the Mean:** Because squared loss penalizes large errors, predictions compress toward the center ($\bar{y} \approx 2.98$): lowest-quintile ratings are overestimated by $+0.44$, and highest-quintile ratings are underestimated by $-0.38$.
3. **Mandatory Non-Use Policy:**
   > Predictions reflect the subjective consensus of an East Asian collegiate rater panel under studio lighting. **DO NOT USE** for hiring, college admissions, dating filters, credit screening, or evaluating human worth.
