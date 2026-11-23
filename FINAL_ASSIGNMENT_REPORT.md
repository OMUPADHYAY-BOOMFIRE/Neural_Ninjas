# Project-Based Learning Assignment Report
## Estimating Human Ratings from Face Images (Classical Machine Learning)
**Team Name:** Neural Ninjas (Batch A3)  
**Course:** Machine Learning Lab (AIP301)  
**Dataset:** SCUT-FBP5500 Benchmark Dataset (5,500 Faces, 1.0–5.0 Continuous Rating Target)  
**Hardware Platform:** NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM, CUDA 13.1) + AMD Radeon 780M (OpenCL) + 16 CPU Threads  
**Core Constraint:** Pure Classical Machine Learning & Computer Vision (Zero CNNs, Zero Pre-trained Deep Models, Zero Transfer Learning)

---

## Table of Contents
1. [Executive Summary & Decision Narrative](#1-executive-summary--decision-narrative)
2. [Task 1: Data Audit, Duplicate Control & The Percolation Discovery](#2-task-1-data-audit-duplicate-control--the-percolation-discovery)
3. [Task 2: Target Formulation & Mathematical Reliability](#3-task-2-target-formulation--mathematical-reliability)
4. [Task 3: Dual Image Representations & Feature Economics](#4-task-3-dual-image-representations--feature-economics)
5. [Task 4 & 8: Baseline Architectures & Tri-Pipeline Cross-Validation](#5-task-4--8-baseline-architectures--tri-pipeline-cross-validation)
6. [Task 5: Group-Disjoint Partitioning & Leakage Proof](#6-task-5-group-disjoint-partitioning--leakage-proof)
7. [Task 6: Empirical Risk & Dimensionality Study ($p < n, p \approx n, p > n$)](#7-task-6-empirical-risk--dimensionality-study)
8. [Task 7: Feature Selection, Sparsity & Jaccard Stability](#8-task-7-feature-selection-sparsity--jaccard-stability)
9. [Task 9: Error Diagnostics & Image Quality Mann-Whitney Test](#9-task-9-error-diagnostics--image-quality-mann-whitney-test)
10. [Task 10: Demographic Fairness Audit & Regression to the Mean](#10-task-10-demographic-fairness-audit--regression-to-the-mean)
11. [Task 11: Production Inference Pipeline & Uncertainty Protocol](#11-task-11-production-inference-pipeline--uncertainty-protocol)
12. [Critical Introspection: What We Did Exceptionally Well vs. Really Badly](#12-critical-introspection-what-we-did-exceptionally-well-vs-really-badly)
13. [Team Debate Logs & Failed Experiments Record](#13-team-debate-logs--failed-experiments-record)
14. [Individual Handwritten Reflection Templates](#14-individual-handwritten-reflection-templates)

---

## 1. Executive Summary & Decision Narrative

### 1.1 The Research Problem
A digital media research group has amassed a dataset of portrait photographs with ground-truth attractiveness ratings compiled from human evaluators. The group requires a predictive estimation engine for unrated facial photographs without utilizing black-box convolutional neural networks (CNNs) or pre-trained deep representations (e.g., ResNet, VGGFace). Our mandate was to engineer a fully reproducible, leakage-free classical machine learning study, design handcrafted visual descriptors, evaluate empirical risk across dimensional boundaries, and determine whether classical representations support reliable prediction.

### 1.2 High-Level Verdict
**The data supports reliable prediction to a moderate-to-strong degree ($r = 0.8033, R^2 = 0.6272$), with critical caveats regarding extreme ratings and optical blur.**
- A linear shrinkage estimator (Ridge) trained on high-dimensional edge orientation histograms (HOG) combined with micro-texture (LBP) and global chromatic tone outperformed raw pixel baselines by **$25.3\%$ in RMSE**.
- An ensemble combining structural appearance, geometric neoclassical ratios, and a from-scratch neural network achieved a **Mean Absolute Error (MAE) of $0.3272$** on an unseen held-out test split of 818 images.
- However, our fairness and residual audits demonstrate that predictions **regress significantly toward the mean** on extreme ratings, and optical blur statistically significantly degrades model confidence ($p = 0.0446$).

---

## 2. Task 1: Data Audit, Duplicate Control & The Percolation Discovery

### 2.1 Baseline Dataset Audit
The SCUT-FBP5500 dataset was audited across disk and label records:
- **Total Image Files on Disk:** 5,500 JPEG images.
- **Total Label Records in `labels.txt`:** 5,500 corresponding lines.
- **Missing Score / Corrupt Files:** 0 unreadable files; all 5,500 images successfully parsed by OpenCV.
- **Native Spatial Resolution:** Uniformly $350 \times 350$ pixels across all samples (no aspect ratio warping in the raw files).
- **Rating Range:** Bounded continuous interval from a minimum of **$1.0167$** to a maximum of **$4.7500$** (Mean = $2.9814$, Median = $2.9000$, Standard Deviation = $0.6875$, Skewness = $+0.2442$).

```
Target Distribution:
  Min:  1.0167  |  Q1: 2.4500  |  Median: 2.9000  |  Mean: 2.9814  |  Q3: 3.5167  |  Max: 4.7500
  Interquartile Range (IQR): 1.0667  |  Standard Deviation: 0.6875
```

### 2.2 Why Duplicates Create Misleading Results
In facial analysis, duplicate photographs or repeated images of the same individual create two severe failure modes:
1. **Train-Test Leakage (Optimism Bias):** If image $A$ is in the training set and near-identical image $A'$ is in the test set, a high-capacity model memorizes lighting, clothing, and background noise. The reported test accuracy measures **retrieval memory**, not generalizable facial structure.
2. **Loss Gradient Inflation:** If identical faces repeat multiple times in the training fold, the loss optimizer assigns $k$-times higher penalty to errors on that individual, skewing linear regression weights toward that subject's idiosyncratic features.

### 2.3 The Algorithmic Duplicate Search & The "Percolation" Discovery
Full $N^2$ raw pixel comparisons on 5,500 images require $\frac{5500 \times 5499}{2} = 15,122,250$ matrix distance calculations, which is computationally prohibitive.

#### Our Multi-Stage Disjoint Set Algorithm:
1. **Perceptual Compression (dHash):** Each photo is resized to a tiny $9 \times 8$ grayscale matrix (72 pixels). The sign of adjacent horizontal intensity differences ($s[:, 1:] > s[:, :-1]$) generates 64 boolean bits packed into a single unsigned 64-bit scalar (`uint64`).
2. **Bitwise Hamming Matching:** Pairwise comparison reduces to a single bitwise XOR: `dist = popcount(hash_A ^ hash_B)`.
3. **Disjoint Set Union (DSU / Union-Find):** When two images have distance $\le \text{dup\_thr}$, `union(find(i), find(j))` joins their equivalence classes with path compression.

#### The Failure / Critical Discovery at Threshold 6:
Initially, following standard literature, we evaluated a Hamming threshold of $\le 6$ (out of 64 bits). **This resulted in catastrophic graph percolation:**
- **At threshold 6:** 89,796 pairs were flagged, and **4,150 images (75.5% of the entire dataset) collapsed into interconnected duplicate clusters**, with one giant cluster containing 1,620 images!
- **Theoretical Explanation:** Because all 5,500 images are aligned frontal portrait photographs with gray backgrounds, their hashes naturally lie within 4 to 8 bits of each other. Due to the transitive property of Union-Find ($A \sim B$ and $B \sim C \implies A \sim C$), chaining connected distant faces into a single giant component.

#### The Systematic Threshold Sweep:
We executed an empirical percolation sweep across Hamming thresholds:

| Threshold (`--dup_thr`) | Flagged Pairs | Images Grouped | % of Dataset | Max Cluster Size | Verdict |
|:---:|:---:|:---:|:---:|:---:|:---|
| **0 (Exact bitwise match)** | **50** | **88 (39 groups)** | **1.6%** | **5** | **Optimal: Pure identical duplicates isolated** |
| **1** | 521 | 425 | 7.7% | 115 | Transitive chaining begins |
| **2** | 2,462 | 1,092 | 19.9% | 424 | Unacceptable percolation |
| **4** | 21,311 | 2,789 | 50.7% | 1,137 | Over half of dataset merged |
| **6** | 89,796 | 4,150 | 75.5% | 1,620 | Catastrophic giant component |

### 2.4 Physical Dataset Curation
Rather than ignoring duplicates or keeping them in the training folds, we physically curated the dataset:
- Created the directory [`duplicate/`](file:///c:/Users/om12u/Downloads/neural_ninja/duplicate).
- For each of the 39 exact duplicate clusters, **retained exactly one representative photograph in `Images/`** and moved the **49 redundant duplicate files** into `duplicate/`.
- Generated [`duplicate/duplicate_mapping.txt`](file:///c:/Users/om12u/Downloads/neural_ninja/duplicate/duplicate_mapping.txt), documenting the retained ID, moved duplicate IDs, and original scores.
- Generated [`labels_clean.txt`](file:///c:/Users/om12u/Downloads/neural_ninja/labels_clean.txt) containing exactly **5,451 validated, unique images**.

---

## 3. Task 2: Target Formulation & Mathematical Reliability

### 3.1 What the Target Measures vs. What It Does NOT Measure
- **What It Measures:** The statistical arithmetic mean rating of approximately 60 human raters belonging to a specific evaluator panel (primarily university students in East Asia), under uniform studio conditions, on a discrete Likert scale of 1 to 5.
- **What It Does NOT Measure:** It does **not** measure objective beauty, human worth, moral character, intelligence, competence, or cross-cultural universal attractiveness. Any assumption that this scalar represents intrinsic facial quality is scientifically invalid.

### 3.2 Target Metric Formulations
We compared three statistical aggregations of human ratings:
1. **Arithmetic Mean ($\mu$):** $\mu = \frac{1}{M}\sum_{i=1}^M r_i$. Standard expected value. Highly sensitive to rogue outlying annotators.
2. **Median ($M$):** 50th percentile rank. Completely robust to extreme annotator bias, but discards nuanced panel consensus.
3. **10% Trimmed Mean ($\mu_{0.1}$):** Drops top and bottom 10% of ratings before averaging. Balances efficiency with outlier resistance.

```
Target Aggregation Comparison:
  MAE(Mean vs. Median):        0.0824 rating units
  MAE(Mean vs. Trimmed Mean):  0.0211 rating units
  Correlation(Mean, Median):   r = 0.9841
  Mean Inter-Rater Std Dev:    0.684 rating units
```

### 3.3 The Theoretical Performance Ceiling (Split-Half Reliability)
A fundamental tenet of measurement theory dictates that **no model can agree with human evaluators better than the evaluators agree with themselves**.
We partitioned raters randomly into two disjoint subsets ($h_1, h_2$), computed the mean per image, and applied the **Spearman-Brown prophecy formula**:
$$r_{\text{split-half}} = \text{corr}(h_1, h_2) \approx 0.865$$
$$r_{\text{ceiling}} = \frac{2 \cdot r_{\text{split-half}}}{1 + r_{\text{split-half}}} \approx \mathbf{0.927}$$
**Implication:** A Pearson correlation of $r \approx 0.927$ represents the theoretical ceiling ($100\%$ of explainable consensus variance). Our final ensemble correlation of **$r = 0.8033$** achieves **$86.6\%$ of the theoretical human ceiling**.

---

## 4. Task 3: Dual Image Representations & Feature Economics

To empirically test feature compression and avoid arbitrary resolution choices, two distinct representations were extracted in parallel across all 5,451 images:

```
Variant A (low_rgb):   128x128 pixels, 3 channels (RGB + HSV color spaces)
Variant B (high_gray):  48x48 pixels, 1 channel (Grayscale intensity)
```

### 4.1 Mathematical Formulation of Feature Blocks
1. **Histogram of Oriented Gradients (HOG):**
   - Breaks spatial crops into $8 \times 8$ cells, computes unsigned gradient vectors $(\nabla_x, \nabla_y)$, bins angles $\theta = \arctan(\nabla_y / \nabla_x)$ into 9 orientation bins, and normalizes over $2 \times 2$ cell blocks using L2-Hys.
   - Captures facial outline, jaw contour, nasal ridge, eye boundaries.
2. **Local Binary Patterns (LBP):**
   - Computes 8-neighbor circular comparisons: $LBP_{P,R} = \sum_{p=0}^{P-1} s(g_p - g_c)2^p$. We extract uniform patterns on a $4 \times 4$ spatial grid.
   - Captures micro-texture: skin grain, pores, micro-wrinkles, hair density.
3. **Color Histograms (COL):**
   - For `low_rgb`, computes 8 bins each across Hue, Saturation, Value ($24$ features).
   - Captures complexion, overall warmth, contrast with background.
4. **Edge Density Grid (EDGE):**
   - Applies Canny edge detection ($\sigma=1.0$, adaptive thresholds) and Sobel gradient magnitude on an $8 \times 8$ grid ($128$ features).
   - Captures spatial edge distribution across 64 anatomical zones.
5. **Facial Geometry Ratios (GEO):**
   - Hand-engineered neoclassical ratios detected via classical Haar Cascades (`face`, `eye`, `smile`):
     - Aspect ratio ($w/h$), inter-ocular distance to face width ($\Delta x_{\text{eyes}} / w_{\text{face}}$), mouth-to-jaw ratio.
     - Bilateral symmetry correlation: Pearson $r$ between left and right facial halves.
     - Upper-facial hair occlusion proxy: forehead darkness fraction.

### 4.2 Comprehensive Feature Economics Table

| Variant | Block Name | Dimensionality ($p$) | Extraction Latency | RAM Storage | Missing Rate (`nan_frac`) | Analytical Value |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **`low_rgb`** ($128\text{px}$) | **HOG** | 8,100 | 6.35 ms / img | 176.6 MB | 0.00% | Dominant structural contrast representation |
| **`low_rgb`** ($128\text{px}$) | **LBP** | 160 | 3.48 ms / img | 3.49 MB | 0.00% | Micro-texture descriptor |
| **`low_rgb`** ($128\text{px}$) | **COL** | 24 | 0.18 ms / img | 0.52 MB | 0.00% | Global HSV tone and saturation |
| **`low_rgb`** ($128\text{px}$) | **EDGE** | 128 | 1.39 ms / img | 2.79 MB | 0.00% | Canny/Sobel zone density |
| **`low_rgb`** ($128\text{px}$) | **GEO** | 46 | 35.37 ms / img | 1.00 MB | **9.30%** | Neoclassical anatomical proportions |
| **`high_gray`** ($48\text{px}$) | **HOG** | 900 | 1.09 ms / img | 19.62 MB | 0.00% | Coarse gradient outline |
| **`high_gray`** ($48\text{px}$) | **LBP** | 160 | 0.83 ms / img | 3.49 MB | 0.00% | Coarse texture |
| **`high_gray`** ($48\text{px}$) | **COL** | 16 | 0.07 ms / img | 0.35 MB | 0.00% | 16-bin intensity distribution |
| **`high_gray`** ($48\text{px}$) | **EDGE** | 128 | 0.82 ms / img | 2.79 MB | 0.00% | Coarse edge grid |
| **`high_gray`** ($48\text{px}$) | **GEO** | 40 | 3.51 ms / img | 0.87 MB | **27.50%** | Detector fails on small landmarks |

---

## 5. Task 4 & 8: Baseline Architectures & Tri-Pipeline Cross-Validation

### 5.1 The Candidate Implementations
We designed two baselines and three diverse candidate architectures:
1. **Baseline B0 (Dummy Regressor):** Predicts the mean of the training fold. Measures null information.
2. **Baseline B1 (Raw Pixel Ridge):** Standardized $32 \times 32$ raw grayscale pixels ($1,024$ features) with L2 Ridge regression. Tests whether engineered features outperform raw pixels.
3. **Pipeline P1 (Appearance - Classicist):** Concatenation of HOG, LBP, and Color Histograms ($8,284$ features). Standardized and fitted with `RidgeCV` ($\alpha \in [10^1, 10^5]$).
4. **Pipeline P2 (Geometry - Explainable):** Concatenation of Geometry Ratios and Edge Density ($174$ features). Imputed via mean imputation and fitted with `HistGradientBoostingRegressor` (max iterations=300, learning rate=0.05, early stopping). Handles non-linear feature interactions.
5. **Pipeline P3 (Fusion From-Scratch - Mathematician):**
   - **Substantial Original Code Requirement:** Implemented entirely in NumPy without scikit-learn model calls.
   - **From-Scratch PCA:** Randomized SVD (Halko et al., 2011) with power iterations to compute top 100 whitened eigenvectors:
     $$\mathbf{Z} = (\mathbf{X} - \boldsymbol{\mu})\mathbf{V}_k (\boldsymbol{\Lambda}_k + \epsilon\mathbf{I})^{-\frac{1}{2}}$$
   - **From-Scratch MLP:** 1-hidden-layer feedforward network ($100 \to 64 \to 1$) with ReLU activation, L2 weight decay ($\lambda=0.01$), Adam optimization ($\beta_1=0.9, \beta_2=0.999$), inner group-aware validation split, and patience-based early stopping.

### 5.2 Repeated Group Cross-Validation Protocol (Task 8)
Evaluated across **3 independent repeats $\times$ 5 folds = 15 total folds** on development data ($n=4,633$). Reporting mean $\pm$ standard deviation across folds:

| Architecture | Representation | CV MAE | CV RMSE ($\pm \sigma$) | CV MedAE | CV Pearson $r$ | Model Size | Fit Time |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0 (Mean)** | None | $0.5681 \pm 0.009$ | $0.6875 \pm 0.0109$ | $0.5050$ | $0.0000$ | 1 KB | 0.001 s |
| **B1 (Raw32)** | $32\text{px}$ Pixels ($1,024$) | $0.4485 \pm 0.011$ | $0.5745 \pm 0.0152$ | $0.3580$ | $0.5490$ | 30 KB | 1.650 s |
| **P1 (Appearance)**| `low_rgb` ($8,284$) | **$0.3436 \pm 0.008$** | **$0.4477 \pm 0.0104$** | **$0.2720$** | **$0.7590$** | 233 KB | 7.958 s |
| **P2 (Geometry)** | `low_rgb` ($174$) | $0.3681 \pm 0.009$ | $0.4752 \pm 0.0127$ | $0.3010$ | $0.7250$ | 3.68 MB | 2.734 s |
| **P3 (From-Scratch)**| `low_rgb` ($8,458 \to 100$) | $0.3561 \pm 0.009$ | $0.4640 \pm 0.0123$ | $0.2840$ | $0.7390$ | 185 KB | 1.156 s |
| **P1 (Appearance)**| `high_gray` ($1,076$) | $0.3748 \pm 0.009$ | $0.4814 \pm 0.0125$ | $0.3010$ | $0.7140$ | 31 KB | 1.771 s |
| **P2 (Geometry)** | `high_gray` ($168$) | $0.3896 \pm 0.010$ | $0.5009 \pm 0.0123$ | $0.3210$ | $0.6870$ | 3.52 MB | 2.540 s |
| **P3 (From-Scratch)**| `high_gray` ($1,244 \to 100$) | $0.3768 \pm 0.008$ | $0.4861 \pm 0.0101$ | $0.3040$ | $0.7120$ | 180 KB | 1.120 s |

```
Key Metric Comparison:
  B0 RMSE: 0.6875  ─────────────────────────── (Baseline)
  B1 RMSE: 0.5745  ─────────────────── (-16.4% error)
  P2 RMSE: 0.4752  ───────────── (-30.9% error)
  P3 RMSE: 0.4640  ─────────── (-32.5% error)
  P1 RMSE: 0.4477  ───────── (-34.9% error; Best Single Model)
```

---

## 6. Task 5: Group-Disjoint Partitioning & Leakage Proof

### 6.1 Split Integrity
- All duplicate groups identified during the audit were constrained to remain strictly within the same fold partition.
- **Split Distribution:**
  - **Training:** 3,817 unique images ($70.0\%$)
  - **Validation:** 816 unique images ($15.0\%$)
  - **Test (Held-Out):** 818 unique images ($15.0\%$)
  - **Group Overlap (Dev $\cap$ Test):** Exactly **0**.

### 6.2 Quantitative Leakage Verification (Task 5)
To mathematically certify that no test image was a near-duplicate of a training image:
1. **Hash Leakage Check:** For every test image $t \in \text{Test}$, we computed:
   $$\min_{d \in \text{Dev}} \text{Hamming}(h_t, h_d) = \mathbf{1} \quad (\text{Count} \le 0 = \mathbf{0})$$
   Zero test images had a Hamming distance of 0 to any development image.
2. **Pixel-PCA Cosine Similarity Check:** Projected test and dev raw images into a 50-component PCA space fitted strictly on Dev. Only 1 image exhibited cosine similarity $> 0.98$ (due to uniform plain backgrounds), confirming strict subject-disjoint isolation.

---

## 7. Task 6: Empirical Risk & Dimensionality Study ($p < n, p \approx n, p > n$)

Using fine HOG features ($6\text{px}$ cells, yielding $p_{\max} = 14,000$ columns) and fixed training sample size $n_{\text{train}} \approx 3,080$, we systematically evaluated empirical validation error across feature dimensions:
$$p \in \{154, 770, 1540, 2772, 3080, 3388, 6160, 9240\}$$

```
Performance vs. Dimension Behavior:
  p = 0.05 n (p=154)   :  Val RMSE = 0.521  |  Train RMSE = 0.485  (Underfitting)
  p = 0.50 n (p=1540)  :  Val RMSE = 0.456  |  Train RMSE = 0.410  (Stable learning)
  p = 1.00 n (p=3080)  :  Val RMSE peaks for weak Ridge (alpha=0.01) at 0.842 +/- 0.08
  p = 3.00 n (p=9240)  :  Weak Ridge interpolates (Train RMSE -> 0.002), Val RMSE = 0.815
                          Tuned Ridge recovers (Val RMSE = 0.448 +/- 0.009)
```

### Theoretical Insights:
1. **The Interpolation Peak at $p \approx n$:** When the number of features equals the number of training samples, the empirical covariance matrix $\mathbf{X}^T\mathbf{X}$ becomes ill-conditioned (minimum singular value $\sigma_{\min} \to 0$). For weakly regularized models, the inversion explodes, creating massive variance across cross-validation folds.
2. **Over-parameterization ($p > n$):** The weakly regularized model fits the label noise perfectly (training error collapses to zero), resulting in classical over-fitting.
3. **The Role of L2 Regularization:** Tuned `RidgeCV` automatically scales its penalty parameter $\alpha \propto p$, restricting the effective degrees of freedom $\text{df}(\alpha) = \sum \frac{\sigma_i^2}{\sigma_i^2 + \alpha}$ and neutralizing the double-descent penalty.

---

## 8. Task 7: Feature Selection, Sparsity & Jaccard Stability

Selection methods were executed strictly inside cross-validation training folds (never touching validation or test sets) on Pipeline P1 ($8,284$ features):

| Selection / Reduction Method | Retained Features ($p$) | Mean CV RMSE | Mean CV MAE | Fit Time | Model Size | Jaccard Selection Stability |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (None - All Features)** | 8,284 | **0.4520** | **0.3457** | 7.89 s | 0.23 MB | N/A (deterministic) |
| **L1 / Lasso Sparsity ($\le 300$)** | **300** | **0.4546** | **0.3514** | **1.44 s** | **0.23 MB** | **0.4025** |
| **PCA Linear Reduction (100)** | 100 | 0.4847 | 0.3763 | 2.13 s | 3.55 MB | N/A (dense projection) |
| **SelectKBest ($f$-regression 300)** | 300 | 0.5062 | 0.3912 | 1.00 s | 0.33 MB | **0.8126** |
| **Top-20% Variance Threshold** | 1,672 | 0.5423 | 0.4232 | 4.23 s | 0.21 MB | 0.2382 |

### Key Trade-Off Insights:
- **L1 / Lasso is the Optimal Production Compressor:** Reduces feature count by **$96.4\%$** ($8,284 \to 300$), accelerates training by **$5.5\times$**, and incurs an imperceptible RMSE penalty of just **$0.0026$**.
- **The Failure of Variance Thresholding:** High variance in image pixels often reflects peripheral background lighting shifts rather than facial geometry, leading to poor regression performance ($\text{RMSE} = 0.5423$).
- **Selection Stability (Jaccard Index):** Univariate $f$-regression selects virtually identical features across folds ($J = 0.8126$), whereas Lasso selects correlated substitutes across folds, yielding lower stability ($J = 0.4025$) despite superior predictive power.

---

## 9. Task 9: Error Diagnostics & Image Quality Mann-Whitney Test

### 9.1 Quantitative Image Quality Analysis
We isolated the **top 10% worst absolute residuals** ($n = 82$, error $|y - \hat{y}| \ge 0.72$) against the remaining $90\%$ of test images ($n = 736$):

| Quality Metric | Top 10% Worst Residuals | Remaining 90% Test Set | Mann-Whitney $U$ Test $p$-value | Statistical Significance ($\alpha=0.05$) |
|:---|:---:|:---:|:---:|:---:|
| **Sharpness (`blur_lapvar`)** | **760.91** | **821.07** | **$p = 0.0446$** | **Statistically Significant** |
| **Mean Brightness** | 157.62 | 158.99 | $p = 0.4292$ | Not Significant |
| **Contrast (Pixel Std Dev)** | 77.97 | 77.19 | $p = 0.2626$ | Not Significant |
| **Haar Face Detection Rate** | 100.0% | 99.46% | N/A | Equivalent |

**Physical Interpretation:** Blurriness attenuates spatial gradient magnitude ($\nabla I$), reducing the sharpness of HOG cell histograms and leading to systematic underestimation of facial structure.

### 9.2 The Anonymous 20-Image Error Gallery
From [`fig_error_gallery.png`](file:///c:/Users/om12u/Downloads/neural_ninja/out/fig_error_gallery.png), recurring patterns emerge across three distinct failure modes:
1. **Large Overestimates (Predicted $\gg$ True):** Faces with studio lighting, wide smiles, and high symmetry, but rated lower by annotators due to perceived age or non-traditional styling.
2. **Large Underestimates (Predicted $\ll$ True):** Faces with slight head tilt (yaw/pitch) or heavy dark bangs covering the forehead. The Haar cascade fails to locate the forehead and pupil center accurately, depressing the geometric ratio score despite high human ratings.
3. **High Disagreement (High Pipeline Variance):** Occurs predominantly on images with extreme lighting contrasts or partial occlusions (e.g., thick glasses frames), where P1 (appearance) and P2 (geometry) disagree by more than $0.5$ rating units.

---

## 10. Task 10: Demographic Fairness Audit & Regression to the Mean

### 10.1 Demographic Error Distribution (Metadata Audits)
We analyzed predictions across the four metadata groups labeled in the dataset filenames:

| Demographic Group | Test Sample Count ($n$) | Ground Truth Mean ($y$) | Model Predicted Mean ($\hat{y}$) | Signed Bias ($\hat{y} - y$) | Test MAE | Median Absolute Error |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Asian Female (AF)** | 315 | 3.027 | 3.056 | $+0.029$ | 0.356 | 0.281 |
| **Asian Male (AM)** | 286 | 2.881 | 2.913 | $+0.032$ | 0.311 | 0.262 |
| **Caucasian Female (CF)** | 115 | 3.196 | 3.085 | **$-0.111$** | 0.325 | 0.262 |
| **Caucasian Male (CM)** | 102 | 3.021 | 2.915 | **$-0.106$** | 0.289 | 0.231 |

- **Kruskal-Wallis Test on Absolute Error:** $H = 4.12, p = 0.248$. There is no statistically significant difference in absolute error magnitude across demographic groups.
- **Signed Bias Discrepancy:** The model exhibits a systematic negative signed bias on Caucasian portraits ($-0.11$) and a slight positive bias on Asian portraits ($+0.03$). This reflects demographic frequency imbalances in training data and differences in ground-truth annotator distributions.

### 10.2 Regression to the Mean Across Rating Quantiles
Segmenting test ground truth into quintiles illustrates the classical regression-to-the-mean phenomenon:

| Rating Quantile | Rating Interval | Count ($n$) | Mean Signed Error (Bias) | MAE | Interpretation |
|:---|:---:|:---:|:---:|:---:|:---|
| **R1 (Lowest Quintile)** | $[1.02, 2.20]$ | 169 | **$+0.437$** | 0.454 | Severe overestimation (predicted too high) |
| **R2 (Low-Mid)** | $(2.20, 2.70]$ | 163 | $+0.159$ | 0.224 | Slight overestimation |
| **R3 (Mid)** | $(2.70, 3.20]$ | 165 | $+0.041$ | 0.201 | Accurate / balanced |
| **R4 (Mid-High)** | $(3.20, 3.85]$ | 161 | $-0.088$ | 0.218 | Slight underestimation |
| **R5 (Highest Quintile)**| $(3.85, 4.75]$ | 160 | **$-0.382$** | 0.412 | Severe underestimation (predicted too low) |

**Mathematical Cause:** Minimizing mean squared error ($\sum (y - \hat{y})^2$) on bounded targets pulls extreme predictions toward the empirical center of mass ($\bar{y} \approx 2.98$).

---

## 11. Task 11: Production Inference Pipeline & Uncertainty Protocol

The standalone inference engine [`predict.py`](file:///c:/Users/om12u/Downloads/neural_ninja/predict.py) bundles the trained estimators, scalers, and empirical cross-validation error distributions into [`out/bundle.joblib`](file:///c:/Users/om12u/Downloads/neural_ninja/out/bundle.joblib):

```bash
python predict.py test_face.jpg --bundle out/bundle.joblib
```

### Safety Features Implemented:
1. **Empirical 80% Confidence Interval:** Extracted from the 10th and 90th percentiles of cross-validation residuals ($[-0.48, +0.42]$):
   $$\text{Interval} = [\hat{y} - 0.42, \; \hat{y} + 0.48]$$
2. **Multi-Model Disagreement Warning:** If the standard deviation between P1, P2, and P3 exceeds the 75th percentile of CV disagreement ($\sigma_{\text{models}} > 0.18$), an automated `[!] Low Confidence` warning is generated.
3. **Automated Optical Quality Checks:**
   - Image resolution $< 100\text{px} \implies$ Flagged as unreliable.
   - Aspect ratio deviation $|h/w - 1| > 0.3 \implies$ Flagged for geometric distortion.
   - Laplacian blur variance outside 5th–95th training percentile $\implies$ Flagged for out-of-focus blur.
   - Haar frontal face detector failure $\implies$ Flagged for landmark occlusion.
4. **Standing Ethical Warning:** The CLI prints a mandatory disclaimer:
   > *"Do NOT use for hiring, admissions, dating filters, credit screening, or judging real people. Estimates reflect a specific rater panel's subjective consensus and regress heavily toward the mean."*

---

## 12. Critical Introspection: What We Did Exceptionally Well vs. Really Badly

### What We Did Exceptionally Well (Beyond the Brief)
1. **Mathematical Rigor in Duplicate Detection:** Discovering and documenting the **percolation catastrophe** at Hamming threshold 6 and systematically proving that threshold 0 is required for strict isolation. Moving 49 redundant duplicate files to `duplicate/` and logging their mapping in `duplicate_mapping.txt` established a zero-leakage benchmark ($0$ test images with Hamming $\le 0$).
2. **From-Scratch MLP with Exact Gradient Verification:** Hand-crafting randomized SVD PCA and an Adam-optimized MLP in pure NumPy, and verifying backprop gradients against finite differences to **$4.05 \times 10^{-9}$ precision**, exceeded the coursework baseline.
3. **Hardware Acceleration Architecture:** Explicitly probing and binding CUDA on the RTX 5050 Laptop GPU (8 GB VRAM) alongside OpenCL on the Radeon 780M, while capping memory under 12 GB via batching and `float32` matrix storage.
4. **Empirical Verification of Blur Bias:** Proving via a non-parametric Mann-Whitney U test ($p = 0.0446$) that blur drives top-decile prediction error.

### What We Did Really Badly / Critical Weaknesses
1. **Initial Blind Acceptance of `--dup_thr 6`:** We initially trusted a standard threshold without checking connected component sizes, which chained 75.5% of the dataset into duplicate groups. Had we not audited component sizes, our train/test splits would have been heavily distorted.
2. **Haar Cascade Fragility on Compressed Images:** Relying on classical Viola-Jones Haar cascades at $48 \times 48$ was an engineering failure ($27.5\%$ landmark detection loss). Classical box filters cannot resolve pupil centers at low resolutions.
3. **The Complete Failure of KMeans Clustering:** We hypothesized that unsupervised KMeans clustering might uncover meaningful rating tiers. In practice, KMeans produced near-zero rating separation ($\eta^2 < 0.05$) and simply clustered lighting conditions.
4. **Inability to Correct Regression to the Mean:** Because Ridge and MLP optimize L2 loss, our models severely compress the dynamic range: ground-truth ratings span $1.02\text{--}4.75$, but model predictions span only $1.85\text{--}4.10$.

---

## 13. Team Debate Logs & Failed Experiments Record

### Debate 1: The Target Metric (Mean vs. Median)
- **Classicist:** Mean is standard, easy to explain, and matches Kaggle labels.
- **Skeptic:** Median is robust to rogue annotators who give 1.0 or 5.0 to every face.
- **Decision:** Target correlation between mean and median was $r = 0.9841$, with a mean difference of only $0.08$ rating units. We retained the **mean** to preserve label compatibility, but documented the split-half ceiling.

### Debate 2: Duplicate Threshold (Loose vs. Strict)
- **Classicist:** Use Hamming threshold 6 to catch faces with slight lighting changes.
- **Mathematician & Skeptic:** Demonstrated that threshold 6 merges 4,150 images into giant clusters via transitive chaining.
- **Decision:** Enforced **threshold 0**, catching 39 exact duplicate groups (88 images) and physically moving 49 redundant files to `duplicate/`.

### Debate 3: Unsupervised Preprocessing (PCA vs. KMeans)
- **Classicist:** KMeans cluster distances are intuitive and explainable.
- **Mathematician:** Proved that Ridge on PCA features achieved $\text{RMSE} = 0.464$, while Ridge on KMeans distances achieved $\text{RMSE} = 0.582$ ($\approx$ baseline).
- **Decision:** Rejected KMeans for production; documented as a failed negative experiment.

### Debate 4: Retaining Color Features
- **Ethicist:** Color histograms correlate with skin tone and could amplify demographic bias.
- **Engineer & Classicist:** Dropping color increases RMSE by $0.024$.
- **Decision:** Retained color in P1 to maximize predictive performance, but added an explicit demographic bias audit showing the Caucasian/Asian signed error gap.

---

## 14. Individual Handwritten Reflection Templates

*(Transcribe these templates into your individual handwritten submissions)*

```
================================================================================
MEMBER 1: THE SKEPTIC (Audit, Duplicates, Leakage Control)
================================================================================
1. What did I learn?
   I learned that duplicate detection using graph-based Union-Find is deceptively
   dangerous. A threshold that seems reasonable (Hamming distance 6 out of 64)
   can trigger percolation and merge 75% of a dataset into a giant component.
2. What worked well?
   Downsampling images to 9x8 grayscale and computing 64-bit dHashes made pairwise
   comparisons instantaneous via bitwise XOR and popcount, letting us audit 15
   million pairs in under 3 seconds.
3. What was most challenging?
   Designing a truly subject-disjoint split when the dataset provides no subject
   identity tags—only coarse filename prefixes.
4. What will I do differently next time?
   I will immediately plot connected component cluster size histograms before
   committing to any Union-Find threshold.

================================================================================
MEMBER 2: THE MATHEMATICIAN (From-Scratch PCA & MLP Implementation)
================================================================================
1. What did I learn?
   I learned that deriving analytic backpropagation by hand and validating it
   against centered numerical gradients gives an irreplaceable understanding of
   optimization dynamics and numerical stability.
2. What worked well?
   The gradient check reached a relative error of 4.05e-09, proving exact mathematical
   correctness. Our from-scratch MLP achieved r = 0.7510 on the test set.
3. What was most challenging?
   Deriving the vector-matrix chain rule for batch Adam updates with L2 regularization
   and preventing exploding gradients during early epochs without PyTorch.
4. What will I do differently next time?
   I will implement mini-batch gradient clipping earlier in the training loop
   to allow higher initial learning rates.

================================================================================
MEMBER 3: THE CLASSICIST (Feature Engineering: HOG, LBP, Geometry)
================================================================================
1. What did I learn?
   Classical descriptors carry remarkable signal. HOG combined with LBP and Ridge
   regression reached r = 0.7869, proving that spatial contrast and skin micro-
   texture account for the vast majority of human rating variance.
2. What worked well?
   L1 Lasso selection successfully reduced 8,284 features to 300 with virtually
   zero loss in accuracy (RMSE 0.4546 vs 0.4520) while speeding up training 5.5x.
3. What was most challenging?
   Haar cascade detectors failed completely at 48x48 resolution (27.5% missing rate),
   forcing us to engineer robust fallback imputation strategies.
4. What will I do differently next time?
   I will test multi-scale HOG cell sizes (e.g., combining 4px, 8px, and 16px cells)
   rather than relying on a single cell grid.

================================================================================
MEMBER 4: THE ENGINEER & ETHICIST (Pipeline Orchestration, Fairness & Predict CLI)
================================================================================
1. What did I learn?
   I learned that evaluation metrics alone hide severe structural biases: our model
   achieved r = 0.8033, yet severely regressed extreme ratings to the mean and
   exhibited statistically significant sensitivity to optical blur (p = 0.0446).
2. What worked well?
   Our checkpoint and resume architecture allowed seamless recovery across crashes,
   and the predict.py CLI reliably outputs empirical 80% confidence intervals.
3. What was most challenging?
   Auditing demographic fairness when rater demographics are unknown and only
   coarse subject metadata exists.
4. What will I do differently next time?
   I will integrate an explicit calibration layer (e.g., isotonic regression) to
   counteract regression-to-the-mean on extreme ratings.
================================================================================
```

---

### Final Submission File Map

| Deliverable Required | Script / Artifact Location |
|:---|:---|
| **Clean Modular Scripts** | [`config.py`](file:///c:/Users/om12u/Downloads/neural_ninja/config.py), [`features.py`](file:///c:/Users/om12u/Downloads/neural_ninja/features.py), [`models.py`](file:///c:/Users/om12u/Downloads/neural_ninja/models.py), [`preprocessing.py`](file:///c:/Users/om12u/Downloads/neural_ninja/preprocessing.py), [`evaluation.py`](file:///c:/Users/om12u/Downloads/neural_ninja/evaluation.py), [`run_pipeline.py`](file:///c:/Users/om12u/Downloads/neural_ninja/run_pipeline.py), [`predict.py`](file:///c:/Users/om12u/Downloads/neural_ninja/predict.py) |
| **Reproducible Notebook** | [`neural_ninjas_colab.ipynb`](file:///c:/Users/om12u/Downloads/neural_ninja/neural_ninjas_colab.ipynb) |
| **Data Audit & Manifest** | [`out/task1_audit.json`](file:///c:/Users/om12u/Downloads/neural_ninja/out/task1_audit.json), [`out/split_manifest.csv`](file:///c:/Users/om12u/Downloads/neural_ninja/out/split_manifest.csv), [`duplicate/duplicate_mapping.txt`](file:///c:/Users/om12u/Downloads/neural_ninja/duplicate/duplicate_mapping.txt) |
| **Model Comparison Table**| [`out/task8_cv_summary.csv`](file:///c:/Users/om12u/Downloads/neural_ninja/out/task8_cv_summary.csv), [`out/test_results.csv`](file:///c:/Users/om12u/Downloads/neural_ninja/out/test_results.csv) |
| **Visual Diagnostic Plots**| [`out/fig01_target_distribution.png`](file:///c:/Users/om12u/Downloads/neural_ninja/out/fig01_target_distribution.png), [`out/fig05_leakage_check.png`](file:///c:/Users/om12u/Downloads/neural_ninja/out/fig05_leakage_check.png), [`out/fig08_metric_comparison.png`](file:///c:/Users/om12u/Downloads/neural_ninja/out/fig08_metric_comparison.png), [`out/fig_scoreboard_and_cost.png`](file:///c:/Users/om12u/Downloads/neural_ninja/out/fig_scoreboard_and_cost.png), [`out/fig_learning_curves.png`](file:///c:/Users/om12u/Downloads/neural_ninja/out/fig_learning_curves.png), [`out/fig_performance_vs_dimension.png`](file:///c:/Users/om12u/Downloads/neural_ninja/out/fig_performance_vs_dimension.png), [`out/fig_feature_selection.png`](file:///c:/Users/om12u/Downloads/neural_ninja/out/fig_feature_selection.png), [`out/fig_confusion_matrices.png`](file:///c:/Users/om12u/Downloads/neural_ninja/out/fig_confusion_matrices.png), [`out/fig_residuals.png`](file:///c:/Users/om12u/Downloads/neural_ninja/out/fig_residuals.png), [`out/fig_error_gallery.png`](file:///c:/Users/om12u/Downloads/neural_ninja/out/fig_error_gallery.png), [`out/fig_group_and_range_errors.png`](file:///c:/Users/om12u/Downloads/neural_ninja/out/fig_group_and_range_errors.png) |
| **Trained Production Bundle**| [`out/bundle.joblib`](file:///c:/Users/om12u/Downloads/neural_ninja/out/bundle.joblib) |
