# Estimating Human Ratings from Face Images — Classical ML Study
**Report focus: Student 2 – P2 – Geometry & Structure (Facial Ratios + Edge Grid → HistGB)**  |  Name: ________  Roll No.: ________  |  Submitted to Dr. Shibna Hussain
**Neural Ninjas (Batch A3) · AIP301 Machine Learning Lab · IIIT Kota · Nov 2026**
Dataset: SCUT-FBP5500 (5,500 faces, mean rating 1–5, ~60 raters/image). Constraint: classical ML + OpenCV only (no CNN, no pretrained/transfer models).

---

## 1. Verdict (what the data supports)
Supervised **regression**. On an unseen, group-disjoint test set (n = 818) the 3-pipeline ensemble reaches **MAE 0.3272, Pearson r 0.8033, R² 0.6272** (86.6 % of the rater-agreement ceiling r = 0.927). 
My geometry pipeline (P2) proves that **classical neoclassical canons and facial symmetry contain real predictive signal** without deep learning. Using only **174 handcrafted structural features** (47× fewer than P1), P2 achieves **CV RMSE 0.4752** (−30.9 % vs mean baseline) and **Pearson r = 0.725**. Crucially, P2 provides interpretable non-linear feature importances that explain *why* ratings vary, though Haar cascade landmark detectors fail heavily when images are compressed below 100 px.

---

## 2. Shared sequential backbone (identical for every pipeline)
Every pipeline builds on the exact same verified, group-disjoint data foundation:

| Stage | What we did | Key numbers | Reason |
|---|---|---|---|
| Audit (T1) | Full OpenCV integrity scan | 5,500 images, 0 corrupt, uniform 350×350 px; rating 1.017–4.750, mean 2.981, median 2.900, SD 0.688 | Catch missing files and scale anomalies before modelling |
| Duplicates (T1) | 9×8 gray → 64-bit dHash → Hamming (XOR) → Disjoint Sets | **thr 6: 4,150 imgs (75.5 %) chained into one giant component**; **thr 0: 39 groups, 88 imgs** → 49 moved to `duplicate/` → **5,451 unique** | Graph percolation at thr 6 causes catastrophic train-test leakage; thr 0 enforces strict duplicate isolation |
| Target (T2) | mean vs median vs 10 % trimmed | MAE(mean,median) 0.082; r(mean,median) 0.984; rater SD 0.684; split-half r 0.865 → Spearman-Brown ceiling **0.927** | Mean kept for label compatibility; ceiling proves no model can exceed r ≈ 0.93 |
| Features (T3) | 5 blocks × 2 variants | **low_rgb** 128 px: 8,458 d total (GEO 46 d, 35.4 ms/img, 9.3 % nan); **high_gray** 48 px: 1,244 d (GEO 40 d, 3.5 ms/img, **27.5 % nan**) | Low resolution destroys Haar landmark detection; 128 px preserves anatomical ratios |
| Split (T5) | group-disjoint 70/15/15 + repeated group CV | train 3,817 / val 816 / test 818; dev∩test groups = **0**; min test Hamming dist to dev = **1 bit** | Disjoint sets prevent identical duplicate leakage across folds |

All feature transformations, median imputers, and scalers are fitted strictly on training folds. The test set is evaluated once at the very end.

---

## 3. P2 – Geometry & Structure (Facial Ratios + Edge Grid → HistGB) – deep dive (owner: Student 2)
Repeated group CV: 3 repeats × 5 folds on dev (n = 4,633); mean ± SD across 15 folds. Owner rows in bold.

| Model | Features | CV MAE | CV RMSE | MedAE | r | RMSE vs B0 | Fit Time | Model Size |
|---|---|---|---|---|---|---|---|---|
| B0 train mean | – | 0.568 | 0.6875 | 0.505 | 0.000 | −0 % | 0.001 s | 1 KB |
| B1 raw 32×32 ridge | 1,024 | 0.449 | 0.5745 | 0.358 | 0.549 | −16.4 % | 1.65 s | 30 KB |
| P1 Appearance low_rgb | 8,284 | 0.344 | 0.4477 | 0.272 | 0.759 | −34.9 % | 7.96 s | 233 KB |
| **P2 Geometry low_rgb** | **174** | **0.368** | **0.4752** | **0.301** | **0.725** | **−30.9 %** | **2.73 s** | **3.68 MB** |
| P3 From-scratch low_rgb | 8,458→100 | 0.356 | 0.4640 | 0.284 | 0.739 | −32.5 % | 1.16 s | 185 KB |
| P1 high_gray | 1,076 | 0.375 | 0.4814 | 0.301 | 0.714 | −30.0 % | 1.77 s | 31 KB |
| **P2 Geometry high_gray** | **168** | **0.390** | **0.5009** | **0.321** | **0.687** | **−27.1 %** | **2.54 s** | **3.52 MB** |
| P3 high_gray | 1,244→100 | 0.377 | 0.4861 | 0.304 | 0.712 | −29.3 % | 1.12 s | 180 KB |

### 3.1 Idea and reasoning
*Core Hypothesis:* Human perception of facial attractiveness relies heavily on **geometric proportions (neoclassical canons)**, bilateral horizontal symmetry, and clean jawline definition rather than raw pixel intensities alone.
*Why this design:* 
1. **Compactness:** Rather than 8,000+ dense pixel gradients, geometry condenses facial structure into 46 geometric ratios and 128 regional edge densities ($p = 174$).
2. **Non-linear interactions:** Facial ratios do not act linearly. For example, a wide jaw might look balanced only if inter-pupillary distance is proportionally large. A linear Ridge regression cannot model such cross-feature interactions without combinatorial polynomial expansion. Hence, I chose **`HistGradientBoostingRegressor`**, which naturally discovers decision splits, threshold non-linearities, and feature interactions.
3. **Missing value resilience:** Classical Haar cascades occasionally miss an eye or mouth landmark due to tilt or bangs. HistGB handles missing values (`np.nan`) natively during tree construction by routing missing values to whichever child node minimizes training variance.

### 3.2 Algorithm, step by step
1. **Landmark Detection:** On 128×128 image, run OpenCV Haar Cascades (`haarcascade_frontalface_default`, `haarcascade_eye`, `haarcascade_smile`) to detect bounding boxes for the face, left eye, right eye, and mouth.
2. **Neoclassical Ratio Computation (GEO, 46 d):**
   - *Face Aspect Ratio:* $w_{\text{face}} / h_{\text{face}}$.
   - *Inter-ocular Ratio:* $\Delta x_{\text{eyes}} / w_{\text{face}}$ (checks the rule of fifths).
   - *Vertical Proportions:* eye-to-nose vs nose-to-chin vertical distances (classical rule of thirds).
   - *Bilateral Symmetry:* Pearson correlation $r$ between pixel intensities of the left half and horizontally flipped right half of the cropped face.
   - *Forehead Occlusion Ratio:* fraction of dark pixels in upper facial third (proxy for bangs/hair obstruction).
   - *Fallback:* If a cascade fails, record `np.nan` and set indicator `face_found = 0`.
3. **Edge Density Grid (EDGE, 128 d):**
   - Apply Canny edge detection ($\sigma = 1.0$) and Sobel horizontal/vertical gradient magnitudes.
   - Partition into an $8 \times 8$ anatomical grid (64 spatial zones) and compute mean edge count and mean Sobel intensity per zone ($64 \times 2 = 128$ features).
4. **Model Architecture:**
   - Concatenate GEO (46) + EDGE (128) = 174 features.
   - `HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, early_stopping=True, random_state=42)`.

### 3.3 Results
- **CV Error:** MAE $0.3681 \pm 0.009$, RMSE $0.4752 \pm 0.0127$, Pearson $r = 0.7250$.
- **Gain over Baselines:** Beats dummy baseline B0 by **$30.9\%$** in RMSE and beats raw pixel Ridge B1 by **$17.3\%$** in RMSE.
- **Top Feature Importances (from tree split gain):**
  1. *Bilateral Facial Symmetry ($r$):* $18.4\%$ relative importance (higher symmetry correlates positively with consensus rating).
  2. *Inter-pupillary to Face Width Ratio:* $14.1\%$ relative importance (deviation from golden ratio ~0.46 penalizes score).
  3. *Lower Jawline Canny Edge Density:* $11.7\%$ relative importance (sharp jawline contrast correlates positively).
  4. *Forehead Hair Occlusion Proxy:* $8.9\%$ relative importance.
  5. *Lower Third to Total Height Ratio:* $7.6\%$ relative importance.
- **Resolution Impact (128 px vs 48 px):**
  At 48 px, landmark detection failure explodes from **$9.3\%$ to $27.5\%$**. CV RMSE worsens from $0.4752 \to 0.5009$ (+5.4 % error). Classical Viola-Jones features require minimum pixel contrast that disappears at 48×48.

### 3.4 What worked, errors and limits
- **What worked:** Extremely fast inference; high explainability; beats raw pixels convincingly with just 174 features; handles missing landmarks without imputation crashes.
- **Errors & Disagreements:** P2 struggles on non-frontal poses (slight yaw/pitch drops face detection confidence) and thick glasses frames (which trigger false eye-box coordinates). On such images, P2 and P1 disagree by $> 0.5$ rating units.
- **Limits:** Haar cascades produce loose rectangular bounding boxes, not 68 dense facial fiducial points. True 3D depth and head pose cannot be extracted with 2D box cascades.

### 3.5 Debate & Comparison with P1 and P3
- *Vs. P1 (Appearance):* P1 achieves lower RMSE (0.4477 vs 0.4752) because 8,100 HOG bins capture skin micro-contrast and lighting nuances that 46 ratios miss. However, P1 is a black-box linear combination of 8,284 weights, whereas P2's 174 features can be audited and explained to a human panel.
- *Vs. P3 (From-Scratch MLP):* P3 reaches RMSE 0.4640 using 100 PCA components. In our final ensemble, combining P2 with P1 and P3 pushes test correlation to **$r = 0.8033$**, demonstrating that geometric canons provide orthogonal, non-redundant signal.

---

## 4. Facial Ratio & Compression Studies (T3, T6)
**Impact of Image Compression on Landmark Availability:**

| Feature Block | Resolution | Dimensions | Extraction Latency | Missing Rate (`nan_frac`) | Analytical Verdict |
|---|---|---|---|---|---|
| GEO (Landmarks) | 128×128 (low_rgb) | 46 | 35.37 ms / img | **9.30 %** | Usable; 90%+ faces fully parsed |
| GEO (Landmarks) | 48×48 (high_gray) | 40 | 3.51 ms / img | **27.50 %** | **Failure**: Over 1 in 4 faces missing key ratios |
| EDGE (Grid) | 128×128 (low_rgb) | 128 | 1.39 ms / img | 0.00 % | Sharp localized contours |
| EDGE (Grid) | 48×48 (high_gray) | 128 | 0.82 ms / img | 0.00 % | Diffuse, blurred edge histograms |

*Conclusion:* For structural geometry, compressing images to 48 px is an unacceptable engineering compromise. We must maintain at least 128×128 spatial resolution for reliable landmark localisation.

---

## 5. Error Diagnostics & Qualitative Failure Cases (T9)
Audit of top 10% worst test errors ($n = 82$, $|y - \hat{y}| \ge 0.72$):
1. **Eyeglasses & Occlusions:** Thick dark frames distort the `haarcascade_eye` detector, placing the perceived eye center 10–15 pixels lower than the true pupil. This falsely inflates the eye-to-nose ratio, causing P2 to underestimate human rating by up to 0.85 points.
2. **Hair Bangs / Fringe:** Heavy dark bangs obscure the forehead, causing the symmetry calculation to compare hair strands rather than facial contours.
3. **Head Tilt (Yaw > 15°):** Classical 2D aspect ratios assume frontal parallel plane. A tilted head foreshortens one half of the face, triggering a false asymmetry penalty.

---

## 6. Failed Experiments & Engineering Decisions
1. **Experiment: Pure Linear Ridge on 46 Geometric Ratios.**
   - *Outcome:* CV RMSE was 0.5312 (barely better than raw pixels B1: 0.5745).
   - *Reason:* Neoclassical canons are non-linear; deviation from an ideal ratio in either direction is penalized. Linear models cannot capture quadratic penalties without manual polynomial features.
   - *Decision:* Switched to `HistGradientBoostingRegressor`, which dropped RMSE from 0.5312 to 0.4752.
2. **Experiment: Estimating Eye Pupils at 48×48 Resolution.**
   - *Outcome:* $27.5\%$ missing rate; eye cascade frequently triggered on eyebrows or nostrils.
   - *Decision:* Rejected 48 px for geometric extraction; strictly used 128 px `low_rgb`.
3. **Experiment: Unsupervised KMeans Clustering on Facial Ratios.**
   - *Outcome:* Cluster separation on ratings was negligible ($\eta^2 < 0.04$). KMeans clustered head tilt and image brightness rather than attractive facial geometry.
   - *Decision:* Abandoned KMeans preprocessing; relied on supervised gradient boosting.

---

## 7. Submission Checklist & Reflection
- [x] Full group cross-validation evaluated across 15 folds (3 repeats × 5 folds).
- [x] Code strictly classical (OpenCV Haar cascades + NumPy + HistGB, zero CNNs).
- [x] All imputers and models fitted strictly inside training folds (zero test leakage).
- [x] Script `03_train_models_and_cv.py` runs P2 cleanly and saves model weights.
