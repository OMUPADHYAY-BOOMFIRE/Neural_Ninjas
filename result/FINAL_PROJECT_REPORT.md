# Estimating Human Ratings from Face Images — Final Project Report
**Course:** AIP301 Machine Learning Lab · **Submitted to:** Dr. Shibna Hussain · **Institution:** IIIT Kota  
**Team:** Neural Ninjas (Batch A3)  
**Members:** Student 1 (Appearance - P1), Student 2 (Geometry - P2), Student 3 (From-Scratch Fusion - P3), Student 4 (Auditor & Ethicist)  
**Dataset:** SCUT-FBP5500 (5,500 Faces, continuous rating target 1.0–5.0)  
**Hardware:** NVIDIA GeForce RTX 5050 (8 GB VRAM, CUDA) + AMD Radeon 780M (OpenCL) + 16 CPU Threads  
**Constraint:** Pure classical machine learning and computer vision (no CNNs, no pre-trained deep models, no transfer learning).

---

## 1. Executive Summary & Verdict
Can classical machine learning reliably predict human aesthetic consensus from facial photographs without deep neural networks?
**Yes, with clearly defined mathematical limits.** On a strictly group-disjoint, unseen test partition of 818 images, our tri-pipeline ensemble achieves:
$$\text{MAE} = \mathbf{0.3272}, \quad \text{RMSE} = \mathbf{0.4243}, \quad \text{Pearson } r = \mathbf{0.8033}, \quad R^2 = \mathbf{0.6272}$$
This reaches **$86.6\%$ of the theoretical rater-agreement ceiling ($r = 0.927$)** established by split-half reliability.

### Key Discoveries & Caveats:
1. **The Duplicate Percolation Catastrophe:** Setting a standard perceptual hash threshold ($\le 6$ bits) caused transitive chaining via Union-Find, collapsing $75.5\%$ of the dataset into duplicate clusters. Enforcing threshold 0 isolated 39 true clusters (88 images), and we physically moved 49 redundant duplicate files to `duplicate/` to guarantee zero test leakage.
2. **From-Scratch Rigor:** Student 3 implemented randomized SVD PCA and a feedforward neural network with manual backpropagation and Adam optimizer in pure NumPy, verified to a numerical gradient error of **$4.05 \times 10^{-9}$**.
3. **Statistical Bias from Blur:** A non-parametric Mann-Whitney U test confirmed that optical blur significantly degrades predictions ($p = 0.0446$).
4. **Regression to the Mean:** Because squared loss optimizes expected value, predictions compress toward the center ($\bar{y} \approx 2.98$): lowest-quintile ratings are overestimated by $+0.44$, and highest-quintile ratings are underestimated by $-0.38$.

---

## 2. Task 1: Data Audit & Duplicate Percolation Control

### 2.1 Baseline Audit
- **Files Checked:** 5,500 JPEG images; 5,500 lines in `labels.txt`.
- **Corrupt Files:** 0 unreadable files; all 5,500 images successfully parsed by OpenCV.
- **Resolution:** Uniformly $350 \times 350$ pixels.
- **Target Distribution:** Continuous range $[1.0167, 4.7500]$; Mean = $2.9814$, Median = $2.9000$, $\text{SD} = 0.6875$, Skewness = $+0.2442$.

### 2.2 Why Duplicates Mislead
If duplicate photos cross the train-test boundary, models memorize lighting, clothing, and background rather than learning facial structure. The reported score reflects **retrieval memorization**, yielding falsely optimistic performance.

### 2.3 The Percolation Catastrophe Sweep
We downsampled images to $9 \times 8$ grayscale to compute 64-bit difference hashes (`dHash`), followed by Disjoint Set Union (Union-Find):

| Hamming Threshold | Duplicate Pairs Flagged | Images Clustered | % Dataset | Max Cluster Size | Verdict |
|:---:|:---:|:---:|:---:|:---:|:---|
| **6** | 89,796 | 4,150 | 75.5% | 1,620 | Catastrophic percolation |
| **4** | 21,311 | 2,789 | 50.7% | 1,137 | Giant component |
| **2** | 2,462 | 1,092 | 19.9% | 424 | Spurious transitive links |
| **1** | 521 | 425 | 7.7% | 115 | Chaining begins |
| **0 (Exact match)** | **50** | **88 (39 groups)** | **1.6%** | **5** | **Optimal: Isolates true duplicates** |

*Physical Curation:* Retained 1 master image per cluster in `Images/`, moved **49 redundant duplicate files** to `duplicate/`, generated `duplicate_mapping.txt`, and produced `labels_clean.txt` containing **5,451 strictly unique images**.

---

## 3. Task 2: Target Formulation & Mathematical Reliability
- **What It Measures:** The average subjective score of ~60 East Asian university student evaluators under studio lighting on a 1–5 Likert scale.
- **What It Does NOT Measure:** It does not measure intrinsic beauty, intelligence, human dignity, or cross-cultural universal attractiveness.
- **Aggregation Metrics:**
  - $\text{MAE}(\text{Mean}, \text{Median}) = 0.0824$ rating units; Correlation $r = 0.9841$.
  - Inter-rater Standard Deviation: $0.684$ units.
  - Split-Half Reliability $r = 0.865 \implies$ Spearman-Brown theoretical ceiling **$r = 0.927$**.
  - We retain the arithmetic mean as our target, noting that no model can validly exceed $r \approx 0.93$.

---

## 4. Task 3: Dual Image Representations & Feature Economics
To evaluate compression trade-offs, two variants were extracted in parallel:
1. `low_rgb` ($128 \times 128$, RGB + HSV, 3 channels)
2. `high_gray` ($48 \times 48$, Grayscale, 1 channel)

| Feature Block | Description | Dim ($p$) 128px | Dim ($p$) 48px | Latency (128px) | Storage (128px) | Missing Rate (128px) | Missing Rate (48px) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **HOG** | 8×8 cells, 9 bins, 2×2 blocks | 8,100 | 900 | 6.35 ms / img | 176.6 MB | 0.00% | 0.00% |
| **LBP** | Uniform patterns, 4×4 grid | 160 | 160 | 3.48 ms / img | 3.49 MB | 0.00% | 0.00% |
| **COL** | 8 bins across H, S, V | 24 | 16 | 0.18 ms / img | 0.52 MB | 0.00% | 0.00% |
| **EDGE**| Canny + Sobel on 8×8 grid | 128 | 128 | 1.39 ms / img | 2.79 MB | 0.00% | 0.00% |
| **GEO** | Haar ratios + bilateral symmetry | 46 | 40 | 35.37 ms / img | 1.00 MB | **9.30%** | **27.50%** |
| **Total**| **All 5 Blocks Combined** | **8,458** | **1,244** | **46.77 ms / img** | **184.4 MB** | — | — |

*Key Finding:* Compressing to 48 px causes landmark detection failure to jump from $9.3\%$ to **$27.5\%$** because Haar cascades cannot resolve eye centers at low resolution.

---

## 5. Tasks 4 & 8: Baselines & Cross-Validation Results
Repeated group-aware cross-validation (**3 repeats $\times$ 5 folds = 15 folds**) on development data ($n = 4,633$):

| Architecture | Description & Features | CV MAE | CV RMSE ($\pm \sigma$) | CV MedAE | CV Pearson $r$ | Fit Time | Model Size |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0 (Baseline)** | Dummy (Predicts train mean) | $0.5681 \pm 0.009$ | $0.6875 \pm 0.0109$ | 0.5050 | 0.0000 | 0.001 s | 1 KB |
| **B1 (Baseline)** | Raw 32×32 Pixels + Ridge ($1,024$) | $0.4485 \pm 0.011$ | $0.5745 \pm 0.0152$ | 0.3580 | 0.5490 | 1.650 s | 30 KB |
| **P1 (Student 1)** | Appearance (HOG+LBP+COL $\to$ Ridge) ($8,284$) | **$0.3436 \pm 0.008$** | **$0.4477 \pm 0.0104$** | **$0.2720$** | **$0.7590$** | 7.958 s | 233 KB |
| **P2 (Student 2)** | Geometry (GEO+EDGE $\to$ HistGB) ($174$) | $0.3681 \pm 0.009$ | $0.4752 \pm 0.0127$ | $0.3010$ | $0.7250$ | 2.734 s | 3.68 MB |
| **P3 (Student 3)** | From-Scratch PCA + MLP ($8,458 \to 100$) | $0.3561 \pm 0.009$ | $0.4640 \pm 0.0123$ | $0.2840$ | $0.7390$ | **1.156 s** | 185 KB |
| **Final Test** | **Ensemble Mean (P1 + P2 + P3)** ($n = 818$) | **0.3272** | **0.4243** | **0.2580** | **0.8033** | — | — |

*Performance Gains:*
- P1 reduces RMSE by **$34.9\%$** vs B0 and **$22.1\%$** vs raw pixels B1.
- P2 uses only 174 features and provides transparent tree feature importances (symmetry = $18.4\%$, eye spacing = $14.1\%$).
- P3 is written from scratch, fits in 1.15 seconds, and achieves $-32.5\%$ error reduction.
- The 3-pipeline ensemble achieves test $r = 0.8033$ ($R^2 = 0.6272$), confirming that combining appearance, geometry, and neural projection produces synergistic gains.

---

## 6. Task 5: Group-Disjoint Splits & Zero-Leakage Proof
- **Split Breakdown:** Train 3,817 ($70\%$), Val 816 ($15\%$), Test 818 ($15\%$).
- **Overlap Check:** Dev $\cap$ Test duplicate groups = **0**.
- **Hamming Distance Check:** For every test image, $\min_{d \in \text{Dev}} \text{Hamming}(h_{\text{test}}, h_d) \ge \mathbf{1}$. Zero test images share identical hashes with dev images.
- **PCA Cosine Similarity:** Only 1 test image exceeded cosine similarity $> 0.98$ (due to solid studio background).

---

## 7. Task 6: Empirical Dimensionality Study ($p < n, p \approx n, p > n$)
Fixed training samples $n_{\text{train}} \approx 3,080$, sweeping fine HOG features up to $p = 9,240$:

| Dimension Regime | Dimension ($p$) | Ratio $p/n$ | Train RMSE | Val RMSE (Weak Ridge $\alpha=0.01$) | Val RMSE (Tuned RidgeCV) |
|---|:---:|:---:|:---:|:---:|:---:|
| Underfitting | 154 | $0.05$ | 0.485 | 0.521 | 0.518 |
| Safe Classical | 1,540 | $0.50$ | 0.410 | 0.456 | 0.452 |
| **Interpolation Peak** | **3,080** | **$1.00$** | 0.298 | **0.842 ± 0.08** | **0.449 ± 0.01** |
| Over-parameterized | 9,240 | $3.00$ | 0.002 | 0.815 ± 0.07 | **0.448 ± 0.009** |

*Insight:* At $p = n$, the empirical Gram matrix $\mathbf{X}^T\mathbf{X}$ becomes ill-conditioned ($\sigma_{\min} \to 0$), causing weak Ridge to explode. Tuned Ridge automatically scales $\alpha \propto p$, restricting effective degrees of freedom $\text{df}(\alpha)$ and recovering optimal generalization.

---

## 8. Task 7: Feature Selection, Sparsity & Jaccard Stability
Evaluated inside training folds on Pipeline P1 ($8,284$ features):

| Selection Method | Retained ($p$) | Mean CV RMSE | Mean CV MAE | Fit Time | Jaccard Stability ($J$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Full Set (No Selection)** | 8,284 | **0.4520** | **0.3457** | 7.89 s | — |
| **L1 / Lasso Sparsity** | **300** | **0.4546** | **0.3514** | **1.44 s** | 0.4025 |
| **PCA Reduction** | 100 | 0.4847 | 0.3763 | 2.13 s | — |
| **SelectKBest ($f$-regression)** | 300 | 0.5062 | 0.3912 | 1.00 s | **0.8126** |
| **Top-20% Variance** | 1,672 | 0.5423 | 0.4232 | 4.23 s | 0.2382 |

*Trade-off Verdict:* L1/Lasso removes **$96.4\%$ of features** with only $+0.0026$ RMSE penalty and $5.5\times$ speedup. Variance filtering fails because pixel variance tracks background lighting rather than facial anatomy.

---

## 9. Task 9: Error Diagnostics & Mann-Whitney Quality Test
- **Top 10% Worst Absolute Errors ($n = 82$, $|y - \hat{y}| \ge 0.72$) vs. Remaining 90% ($n = 736$):**
  - **Sharpness (`blur_lapvar`):** 760.91 vs 821.07, Mann-Whitney **$p = 0.0446$** (statistically significant at $\alpha = 0.05$).
  - **Brightness:** $p = 0.4292$ (not significant).
  - **Contrast:** $p = 0.2626$ (not significant).
- **Error Gallery Patterns:** Overestimates occur on smiling older individuals; underestimates occur on heavy forehead bangs or tilted poses; high model disagreement occurs on thick eyeglasses.

---

## 10. Task 10: Demographic Fairness & Regression to the Mean
- **Demographic Subgroups:**
  - Asian Female ($n=315$): True 3.027, Pred 3.056, Bias $+0.029$, MAE 0.356.
  - Asian Male ($n=286$): True 2.881, Pred 2.913, Bias $+0.032$, MAE 0.311.
  - Caucasian Female ($n=115$): True 3.196, Pred 3.085, Bias **$-0.111$**, MAE 0.325.
  - Caucasian Male ($n=102$): True 3.021, Pred 2.915, Bias **$-0.106$**, MAE 0.289.
  - Kruskal-Wallis test on $|error|$: $H = 4.12, p = 0.248$ (no significant difference in absolute error magnitude). The negative bias on Caucasian faces reflects annotator panel demographics and training imbalances.
- **Regression to the Mean:**
  - Lowest quintile ($[1.02, 2.20]$): Bias **$+0.437$** (predicted too high).
  - Highest quintile ($[3.85, 4.75]$): Bias **$-0.382$** (predicted too low).
  - True ratings span $1.02\text{--}4.75$; predictions span only $1.85\text{--}4.10$.

---

## 11. Task 11: Production Deployment & Ethical Guardrails
The inference CLI [`predict.py`](file:///c:/Users/om12u/Downloads/neural_ninja/predict.py) packages the trained models in [`out/bundle.joblib`](file:///c:/Users/om12u/Downloads/neural_ninja/out/bundle.joblib) with 4 production safeguards:
1. **Empirical 80% Confidence Interval:** $[\hat{y} - 0.42, \hat{y} + 0.48]$.
2. **Model Disagreement Warning:** Triggered if standard deviation across P1, P2, P3 exceeds $0.18$.
3. **Quality Sanity Checks:** Flags images with resolution $< 100\text{px}$, non-square aspect ratio, or abnormal blur.
4. **Mandatory Non-Use Disclaimer:**
   > *"DO NOT USE for hiring, admissions, dating filtering, or judging human worth. Predictions model subjective panel opinions and regress heavily toward the mean."*

---

## 12. Team Debates, Agreements & Disagreements

| Decision Point | Team Perspectives | Outcome & Agreed Rationale |
|:---|:---|:---|
| **Target: Mean vs. Median** | Student 4 argued for median to resist rogue outliers; Student 1 preferred mean. | Retained **mean** because $\text{corr}(\text{mean}, \text{median}) = 0.9841$ and $\text{MAE} = 0.082$. Documented the split-half ceiling ($r = 0.927$). |
| **Percolation Threshold** | Student 1 proposed threshold 6; Students 3 & 4 demonstrated that threshold 6 chained 75.5% of images. | Enforced **threshold 0**, isolating 39 pure duplicate clusters and moving 49 redundant files to `duplicate/`. |
| **KMeans as Preprocessing** | Student 2 hypothesized cluster distances would capture rating tiers. | Student 3 proved Ridge on KMeans achieved $\text{RMSE} = 0.582$ ($\eta^2 < 0.05$). **KMeans rejected** as a negative experiment. |
| **Dropping Color Features** | Student 4 argued dropping color reduces racial bias; Student 1 noted RMSE worsens by $+0.024$. | **Retained color** to preserve accuracy, but conducted an explicit demographic fairness audit. |

---

## 13. Critical Introspection: What We Did Well vs. Really Badly

### What We Did Exceptionally Well:
1. **Percolation Discovery:** Uncovering that Hamming threshold 6 created a giant 1,620-image cluster prevented a catastrophic data leakage error that would have invalidated the entire study.
2. **From-Scratch Verification:** Hand-crafting Randomized SVD PCA and the backpropagation MLP with a $4.05 \times 10^{-9}$ gradient check exceeded standard coursework requirements.
3. **Hardware Utilization:** Probed and verified CUDA acceleration on the RTX 5050 GPU and OpenCL on the Radeon 780M, managing batch memory to remain under 12 GB RAM.

### What We Did Really Badly:
1. **Initial Blind Trust in Default Thresholds:** We initially accepted `--dup_thr 6` without inspecting connected component sizes, which would have ruined train/test isolation.
2. **Haar Cascade Fragility on Compressed Images:** Relying on Viola-Jones Haar cascades at 48 px was an engineering mistake ($27.5\%$ landmark failure).
3. **Unresolved Range Compression:** We could not eliminate regression to the mean using L2 loss estimators without distorting calibration.

---

## 14. Individual Handwritten Reflection Templates

### Student 1 (Appearance - P1)
1. *What did I learn?* High-dimensional HOG features ($p = 8,100$) contain strong predictive signal when regularized with Ridge, even when $p > n$.
2. *What worked well?* L1/Lasso selection compressed 8,284 features to 300 with negligible accuracy loss ($+0.0026$ RMSE).
3. *What was challenging?* Dealing with sensitivity to background contrast and illumination.

### Student 2 (Geometry - P2)
1. *What did I learn?* Bilateral symmetry and facial neoclassical ratios carry real aesthetic signal, but require non-linear tree models.
2. *What worked well?* P2 condensed facial structure into just 174 explainable features with an RMSE within $6\%$ of P1.
3. *What was challenging?* Haar cascades fail when faces tilt or have dark bangs covering the forehead.

### Student 3 (From-Scratch Fusion - P3)
1. *What did I learn?* Deriving randomized SVD PCA and manual backpropagation from scratch revealed how whitening eliminates ill-conditioned covariance matrices.
2. *What worked well?* Our numerical gradient check confirmed backprop analytical gradients to $4.05 \times 10^{-9}$ precision.
3. *What was challenging?* Tuning Adam learning rates and weight decay to prevent the MLP from overfitting noisy annotator scores.

### Student 4 (Auditor & Ethicist)
1. *What did I learn?* Graph-based duplicate detection using Union-Find is vulnerable to percolation; small threshold changes can merge an entire dataset.
2. *What worked well?* Establishing zero-leakage splits and proving the blur bias using non-parametric Mann-Whitney tests.
3. *What was challenging?* Communicating that equal MAE across demographic groups does not guarantee ethical fairness when signed biases differ.
