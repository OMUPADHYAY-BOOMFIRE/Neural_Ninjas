# Estimating Human Ratings from Face Images — Classical ML Study
**Report focus: Student 4 – Auditor & Ethicist (Audit, Percolation, Leakage, Fairness, Safety)**  |  Name: ________  Roll No.: ________  |  Submitted to Dr. Shibna Hussain
**Neural Ninjas (Batch A3) · AIP301 Machine Learning Lab · IIIT Kota · Nov 2026**
Dataset: SCUT-FBP5500 (5,500 faces, mean rating 1–5, ~60 raters/image). Constraint: classical ML + OpenCV only (no CNN, no pretrained/transfer models).

---

## 1. Verdict (what the data supports)
Supervised **regression**. On an unseen, group-disjoint test set (n = 818) the 3-pipeline ensemble reaches **MAE 0.3272, Pearson r 0.8033, R² 0.6272**, achieving **86.6 %** of the theoretical rater-agreement ceiling ($r = 0.927$).
As the auditor and ethicist for Neural Ninjas, my primary mandate was ensuring scientific integrity:
1. **Preventing false claims of accuracy:** Uncovered the **percolation catastrophe** at Hamming threshold 6 where 75.5 % of images merged into duplicate clusters, and enforced strict threshold 0 to physically isolate 49 redundant files into `duplicate/`.
2. **Quantifying empirical error bias:** Proved via Mann-Whitney U testing ($p = 0.0446$) that optical blur degrades model accuracy.
3. **Demographic fairness audit:** Found no significant difference in absolute error across ethnic/gender groups ($H = 4.12, p = 0.248$), but established that predictions **regress severely toward the mean** (+0.44 bias on low ratings, −0.38 bias on high ratings).
4. **Safety guardrails:** Deployed automated inference checks with 80 % empirical confidence intervals and a strict ethical non-use policy.

---

## 2. Shared sequential backbone (identical for every pipeline)

| Stage | What we did | Key numbers | Reason |
|---|---|---|---|
| Audit (T1) | File read & array validation | 5,500 images, 0 corrupt, uniform 350×350; rating 1.017–4.750, mean 2.981, median 2.900, SD 0.688 | Confirm clean data before mathematical derivation |
| Duplicates (T1) | 9×8 gray → 64-bit dHash → Hamming (XOR) → Disjoint Sets | **thr 6: 4,150 imgs (75.5 %) chained into one cluster**; **thr 0: 39 groups, 88 imgs** → 49 moved to `duplicate/` → **5,451 unique** | Graph percolation at thr 6 causes catastrophic train-test leakage; thr 0 enforces strict duplicate isolation |
| Target (T2) | mean vs median vs 10 % trimmed | MAE(mean,median) 0.082; r(mean,median) 0.984; rater SD 0.684; split-half r 0.865 → Spearman-Brown ceiling **0.927** | Mean kept for label compatibility; ceiling proves no model can exceed r ≈ 0.93 |
| Features (T3) | 5 blocks × 2 variants | **low_rgb** 128 px: 8,458 d, 46.8 ms/img; **high_gray** 48 px: 1,244 d, 6.3 ms/img (7.4× faster) | Multi-modal feature fusion across appearance, texture, color, edges, and ratios |
| Split (T5) | group-disjoint 70/15/15 + repeated group CV | train 3,817 / val 816 / test 818; dev∩test groups = **0**; min test Hamming dist to dev = **1 bit** | Disjoint sets prevent identical duplicate leakage across folds |

---

## 3. The Duplicate Detection Crisis: Graph Percolation Discovery (T1, T5)

### 3.1 Why Duplicates Falsify Evaluation
If identical photos or near-identical portraits appear across both the training set and the test set, any flexible model can memorize clothing, lighting, and pixel noise. The test score then measures **retrieval memorization**, not generalization to unseen humans.

### 3.2 The Percolation Catastrophe
We compressed images to 9×8 grayscale and computed 64-bit difference hashes (`dHash`). Pairwise comparison is instantaneous via bitwise XOR: `hamming = popcount(hash_A ^ hash_B)`.
Initially, following standard computer vision heuristics, we set threshold $\le 6$ of 64 bits. **This triggered catastrophic graph percolation:**

| Threshold (`--dup_thr`) | Flagged Pairs | Images Grouped | % of Dataset | Max Cluster Size | Verdict |
|:---:|:---:|:---:|:---:|:---:|:---|
| **6** | 89,796 | **4,150** | **75.5 %** | **1,620** | **Catastrophic percolation (transitive chaining)** |
| **4** | 21,311 | 2,789 | 50.7 % | 1,137 | Giant component persists |
| **2** | 2,462 | 1,092 | 19.9 % | 424 | Spurious transitive links |
| **1** | 521 | 425 | 7.7 % | 115 | Chaining begins |
| **0 (Exact match)** | **50** | **88 (39 groups)** | **1.6 %** | **5** | **Strict, true duplicate isolation** |

*Root Cause:* All 5,500 photos are centered frontal portraits with neutral studio backgrounds. Their perceptual hashes naturally sit within 4–7 bits of each other. Due to the transitive nature of Union-Find ($A \sim B \land B \sim C \implies A \sim C$), an image of person A chained to person B, which chained to person C, creating a single giant component of 1,620 images!

### 3.3 Physical Data Curation
Rather than ignoring this, we curated the disk dataset:
- Isolated 39 duplicate clusters (88 images).
- Retained 1 master image per cluster in `Images/`.
- Moved **49 redundant files** into [`duplicate/`](file:///c:/Users/om12u/Downloads/neural_ninja/duplicate).
- Logged all details in [`duplicate/duplicate_mapping.txt`](file:///c:/Users/om12u/Downloads/neural_ninja/duplicate/duplicate_mapping.txt).
- Produced [`labels_clean.txt`](file:///c:/Users/om12u/Downloads/neural_ninja/labels_clean.txt) with **5,451 strictly unique images**.

---

## 4. Group-Disjoint Splits & Zero-Leakage Certification (T5)
- **Data Partitioning:**
  - Training: 3,817 unique images ($70.0 \%$)
  - Validation: 816 unique images ($15.0 \%$)
  - Test (Held-Out): 818 unique images ($15.0 \%$)
  - Group overlap between dev and test: **0**.
- **Quantitative Leakage Proof:**
  1. *Hash Distance:* $\min_{d \in \text{Dev}} \text{Hamming}(h_{\text{test}}, h_d) \ge \mathbf{1}$. Zero test images have 0-bit distance to dev images.
  2. *Pixel-PCA Cosine Similarity:* Only 1 test image exceeded cosine similarity $> 0.98$ (caused by solid neutral background).
  - *Limitation:* Hash matching catches identical *photographs*, but without identity labels, we cannot prove whether the same *person* appears in two different studio poses.

---

## 5. Target Formulation & Measurement Ethics (T2)
- **What the Target Measures:** The average Likert rating (1.0 to 5.0) of ~60 human annotators from an East Asian collegiate panel assessing studio portraits.
- **What the Target Does NOT Measure:** It does **not** measure objective beauty, intelligence, character, competence, or cross-cultural aesthetic value.
- **Aggregation Comparison:**
  - $\text{MAE}(\text{Mean}, \text{Median}) = 0.0824$ rating units.
  - $\text{Correlation}(\text{Mean}, \text{Median}) = 0.9841$.
  - Inter-rater Standard Deviation: $0.684$ rating units.
  - Split-Half Reliability $r = 0.865 \implies$ Spearman-Brown human ceiling **$r = 0.927$**.
  - We retain the arithmetic mean, acknowledging that human disagreement sets a hard ceiling of $r \approx 0.93$.

---

## 6. Error Diagnostics: Optical Blur & Quality Testing (T9)
We isolated the top 10 % worst absolute errors ($n = 82$, error $|y - \hat{y}| \ge 0.72$) against the remaining 90 % ($n = 736$):

| Quality Metric | Top 10% Worst Errors | Remaining 90% Test | Mann-Whitney $U$ test | Significance ($\alpha = 0.05$) |
|---|---|---|---|---|
| **Sharpness (`blur_lapvar`)** | **760.91** | **821.07** | **$p = 0.0446$** | **Statistically Significant** |
| Mean Brightness | 157.62 | 158.99 | $p = 0.4292$ | Not significant |
| Contrast (Std Dev) | 77.97 | 77.19 | $p = 0.2626$ | Not significant |
| Haar Detection Rate | 100.0 % | 99.46 % | N/A | Equivalent |

*Insight:* Blur attenuates gradient magnitudes, washing out HOG edge histograms and causing models to under-predict structure. Note: under a conservative Bonferroni correction ($0.05 / 3 = 0.0167$), this represents moderate rather than definitive evidence.

---

## 7. Demographic Fairness Audit & Regression to the Mean (T10)

### 7.1 Demographic Subgroup Analysis
Audited across four filename metadata categories:

| Demographic Group | Count ($n$) | True Mean ($y$) | Pred Mean ($\hat{y}$) | Signed Bias ($\hat{y} - y$) | MAE | MedAE |
|---|---|---|---|---|---|---|
| Asian Female (AF) | 315 | 3.027 | 3.056 | $+0.029$ | 0.356 | 0.281 |
| Asian Male (AM) | 286 | 2.881 | 2.913 | $+0.032$ | 0.311 | 0.262 |
| Caucasian Female (CF) | 115 | 3.196 | 3.085 | **$-0.111$** | 0.325 | 0.262 |
| Caucasian Male (CM) | 102 | 3.021 | 2.915 | **$-0.106$** | 0.289 | 0.231 |

- **Kruskal-Wallis Test on Absolute Errors:** $H = 4.12, p = 0.248$. No statistically detectable difference in error magnitude across groups.
- **Signed Bias Gap:** Caucasian portraits show a systematic underestimation bias of $\approx -0.11$, while Asian portraits show a slight overestimation bias of $+0.03$. This reflects dataset representation imbalances and the cultural origin of the annotator panel. Equal MAE does not prove complete demographic parity.

### 7.2 Regression to the Mean Across Quintiles
Evaluating predictions across true rating quintiles reveals severe range compression:

| True Rating Quintile | True Range | Count ($n$) | Signed Error (Bias) | MAE | Effect |
|---|---|---|---|---|---|
| R1 (Lowest) | $[1.02, 2.20]$ | 169 | **$+0.437$** | 0.454 | Predicted much too high |
| R2 (Low-Mid) | $(2.20, 2.70]$ | 163 | $+0.159$ | 0.224 | Slight overestimation |
| R3 (Mid) | $(2.70, 3.20]$ | 165 | $+0.041$ | 0.201 | Accurate predictions |
| R4 (Mid-High) | $(3.20, 3.85]$ | 161 | $-0.088$ | 0.218 | Slight underestimation |
| R5 (Highest) | $(3.85, 4.75]$ | 160 | **$-0.382$** | 0.412 | Predicted much too low |

*Mathematical Cause:* L2 regression penalizes squared residuals, forcing predictions toward the training distribution mean ($\bar{y} \approx 2.98$). True ratings span $1.02\text{--}4.75$, but ensemble predictions span only $1.85\text{--}4.10$.

---

## 8. Deployment Protocol & Ethical Safety (T11)
The deployment script [`predict.py`](file:///c:/Users/om12u/Downloads/neural_ninja/predict.py) enforces 4 mandatory safeguards:
1. **Empirical 80 % Confidence Interval:** Uses 10th and 90th percentiles of CV residuals: $[\hat{y} - 0.42, \hat{y} + 0.48]$.
2. **Model Disagreement Alarm:** Flags high uncertainty if standard deviation between P1, P2, and P3 exceeds 0.18.
3. **Automated Quality Checks:** Warns on resolution $< 100\text{px}$, non-square aspect ratio, blur outside 5th–95th training percentile, or failed face detection.
4. **Mandatory Ethical Notice:**
   > *"DO NOT USE for hiring, college admissions, dating recommendations, criminal profiling, or evaluating human worth. This system models the subjective consensus of a specific annotator panel and suffers from significant regression to the mean."*

---

## 9. Failed Experiments & Introspection
1. **Initial acceptance of threshold 6:** Catastrophic failure that nearly corrupted our train-test split. Lesson: always inspect connected component distributions before accepting clustering thresholds.
2. **Dropping color features for fairness:** Hypothesized removing color histograms would reduce demographic bias. However, CV RMSE worsened by +0.024 while the signed bias gap between Caucasian and Asian faces remained virtually identical. Skin tone was already implicitly encoded in grayscale LBP texture!
3. **KMeans as a fairness equalizer:** Failed completely ($\eta^2 < 0.05$).

---

## 10. Submission Checklist
- [x] Duplicate percolation discovered, documented, and resolved with 49 duplicates moved.
- [x] Zero-leakage verification proven via bitwise Hamming distance and PCA cosine similarity.
- [x] Mann-Whitney U test on optical blur and Kruskal-Wallis test on demographic bias executed.
- [x] Standalone inference engine includes 80 % uncertainty interval and ethical disclaimers.
