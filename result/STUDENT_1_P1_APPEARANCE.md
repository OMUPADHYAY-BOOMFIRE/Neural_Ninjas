# Estimating Human Ratings from Face Images — Classical ML Study
**Report focus: Student 1 – P1 – Appearance (HOG + LBP + colour → Ridge)**  |  Name: ________  Roll No.: ________  |  Submitted to Dr. Shibna Hussain
**Neural Ninjas (Batch A3) · AIP301 Machine Learning Lab · IIIT Kota · Nov 2026**
Dataset: SCUT-FBP5500 (5,500 faces, mean rating 1–5, ~60 raters/image). Constraint: classical ML + OpenCV only (no CNN, no pretrained/transfer models).

---

## 1. Verdict (what the data supports)
Supervised **regression**. On an unseen, group-disjoint test set (n = 818) the 3-pipeline ensemble reaches **MAE 0.3272, Pearson r 0.8033, R² 0.6272**, i.e. **86.6 %** of the rater-agreement ceiling (r = 0.927). Predictions are useful for the middle of the scale, **regress to the mean at the extremes**, and are only weakly affected by blur. The score is an estimate of *one rater panel's average opinion*, not objective beauty.

## 2. Shared sequential backbone (identical for every pipeline)
Fig. 1 (`pipeline_graph_p1.png`, red border = this report's pipeline) shows the graph: audit → features → dedup → target → split → leakage check → PCA/KMeans → baselines → **3 parallel pipelines** → CV join → studies → one test run → error audit → inference bundle.

| Stage | What we did | Key numbers | Reason |
|---|---|---|---|
| Audit (T1) | OpenCV read of every file | 5,500 images, 0 corrupt, all 350×350; rating 1.017–4.750, mean 2.981, median 2.900, SD 0.688, skew +0.24 | catch silent data faults early |
| Duplicates (T1) | 9×8 gray → 64-bit dHash → Hamming (XOR) → Union-Find | **thr 6: 4,150 imgs (75.5 %) chained into one cluster (max 1,620)**; **thr 0: 39 groups, 88 imgs, max 5** → 49 files moved → **5,451 unique** | a loose threshold chains faces transitively (percolation); duplicates across train/test = memorisation, inflated score |
| Target (T2) | mean vs median vs 10 % trimmed | MAE(mean,median) 0.082; r(mean,median) 0.984; rater SD 0.684; split-half r 0.865 → Spearman-Brown ceiling **0.927** | mean kept (≈ median); no model can beat rater self-agreement |
| Features (T3) | 5 blocks × 2 variants | **low_rgb** 128 px: HOG 8,100 + LBP 160 + COL 24 + EDGE 128 + GEO 46 = **8,458 d, 46.8 ms/img**; **high_gray** 48 px: **1,244 d, 6.3 ms/img** (7.4× faster); GEO missing 9.3 % vs 27.5 % | HOG = outline/contrast, LBP = skin/hair texture, COL = tone, EDGE = Canny/Sobel zones, GEO = Haar-based ratios |
| Split (T5) | group-disjoint 70/15/15 + repeated group CV | train 3,817 / val 816 / test 818; dev∩test groups = **0**; every test image ≥ 1 bit from all dev hashes; 1 test image with cosine > 0.98 | same-photo leakage ruled out; same-*person* leakage cannot be excluded (no identity labels) |

All scalers, PCA, imputers and selectors are fit on training rows only; the test set is used once.

<div style="page-break-after: always"></div>
## 3. P1 – Appearance (HOG + LBP + colour → Ridge) – deep dive (owner: Student 1)
Repeated group CV: 3 repeats × 5 folds on dev (n = 4,633); mean ± SD. Owner rows in bold.

| Model | Features | CV MAE | CV RMSE | MedAE | r | RMSE vs B0 | Fit | Size |
|---|---|---|---|---|---|---|---|---|
| B0 train mean | – | 0.568 | 0.6875 | 0.505 | 0.000 | −0 % | 0.001 s | 1 KB |
| B1 raw 32×32 ridge | 1,024 | 0.449 | 0.5745 | 0.358 | 0.549 | −16.4 % | 1.65 s | 30 KB |
| **P1 Appearance low_rgb** | **8,284** | **0.344** | **0.4477** | **0.272** | **0.759** | **−34.9 %** | **7.96 s** | **233 KB** |
| P2 Geometry low_rgb | 174 | 0.368 | 0.4752 | 0.301 | 0.725 | −30.9 % | 2.73 s | 3.68 MB |
| P3 From-scratch low_rgb | 8,458→100 | 0.356 | 0.4640 | 0.284 | 0.739 | −32.5 % | 1.16 s | 185 KB |
| **P1 high_gray** | **1,076** | **0.375** | **0.4814** | **0.301** | **0.714** | **−30.0 %** | **1.77 s** | **31 KB** |
| P2 high_gray | 168 | 0.390 | 0.5009 | 0.321 | 0.687 | −27.1 % | 2.54 s | 3.52 MB |
| P3 high_gray | 1,244→100 | 0.377 | 0.4861 | 0.304 | 0.712 | −29.3 % | 1.12 s | 180 KB |

Extraction cost (low_rgb, derived from per-block timings): P1 10.0 ms/img, P2 36.8 ms, P3 46.8 ms. All three pipelines use the same features and the same folds, so differences come from the model, not the data.

### 3.1 Idea and reasoning
*Belief:* people read attractiveness from **local appearance** – contours (jaw, nose ridge, eye outline), skin micro-texture and overall tone – so we describe the face by what its pixels look like and let a regularised linear model weigh them.
*Why this design:* HOG, LBP and colour are cheap and classical; a linear model keeps every weight inspectable; and Ridge is the standard tool when p exceeds n (here p = 8,284 vs ≈ 3,700 training rows per fold, p/n ≈ 2.2).

### 3.2 Algorithm, step by step
1. Resize to 128×128 (low_rgb) or 48×48 gray (high_gray).
2. **HOG:** 8×8-pixel cells, 9 unsigned orientation bins, 2×2-cell blocks, L2-Hys → 8,100 values (outline and contrast).
3. **LBP:** 8 neighbours, radius 1, uniform patterns, 4×4 grid × 10 bins → 160 values (skin and hair texture).
4. **Colour:** 8 bins each of H, S, V → 24 values (tone, lighting); 16-bin intensity histogram for gray.
5. Concatenate → **8,284** features; **StandardScaler fitted on training rows** (HOG, LBP and colour live on different scales and the Ridge penalty is scale-sensitive).
6. **RidgeCV:** w = (XᵀX + αI)⁻¹Xᵀy; α picked from 10¹…10⁵ inside each training fold. Larger α shrinks effective degrees of freedom df(α) = Σ σᵢ²/(σᵢ² + α), which tames the p > n instability.

### 3.3 Results
| Check | Number |
|---|---|
| CV (3×5) low_rgb | MAE 0.3436 ± 0.008 · RMSE 0.4477 ± 0.0104 · MedAE 0.272 · r 0.759 |
| Test (n = 818) | r = 0.7869 |
| vs raw-pixel B1 | RMSE −22.1 %, MAE −23.4 %, MedAE −24.0 % (r 0.549 → 0.759) |
| vs train-mean B0 | RMSE −34.9 % |
| Compression 128 → 48 px | RMSE 0.4477 → 0.4814 (+0.034, +7.5 %); features 8,284 → 1,076 (7.7× fewer); fit 7.96 → 1.77 s (4.5×); model 233 → 31 KB (7.5×) |
| Extraction (low_rgb) | 10.0 ms/img (HOG 6.35 + LBP 3.48 + COL 0.18); feature storage ≈ 181 MB |
| Dimension study (fine HOG) | weak Ridge peaks at p = n (val RMSE 0.842 ± 0.08); **tuned Ridge keeps 0.448 ± 0.009 at p = 3n** – the same mechanism that makes P1 work |
| Selection on P1 features | L1 keeps 300 of 8,284 features: RMSE 0.4546 vs 0.4520 (+0.0026), 5.5× faster; SelectKBest 0.5062 (stable, J = 0.81); top-variance 0.5423 (worst) |

### 3.4 What worked, errors and limits
- **Worked:** best of the three on every CV metric and the lowest fold-to-fold spread (RMSE SD 0.010); fastest extraction; tiny model; most of the gain over raw pixels comes from HOG + LBP.
- **Errors:** sensitive to blur and strong lighting; on glasses and harsh lighting P1 and P2 disagree by > 0.5 rating units; like every model it compresses the rating range (bias +0.44 at the lowest quintile, −0.38 at the highest).
- **Limits:** features are not face-specific (background and lighting enter the HOG grid – variance filtering failed for this reason); the colour block carries skin-tone information (dropping it costs +0.024 RMSE, so we kept it and audited group bias); 8,284 weights are inspectable but hard to read feature by feature.

### 3.5 Against the other two pipelines
P2 is 6.1 % worse in RMSE but explainable and uses 174 features; P3 is 3.6 % worse but fits 6.9× faster. The ensemble of all three reaches test r = 0.8033 versus 0.7869 for P1 alone, so P1 is the strongest single model and still gains from the other two views.



## 4. Dimension and selection studies (T6, T7)
**Dimension study** (fine HOG, n_train ≈ 3,080; random column order; weak Ridge α = 0.01 vs tuned RidgeCV):

| p (× n) | Result |
|---|---|
| 154 (0.05) | val RMSE 0.521, train 0.485 → underfit |
| 1,540 (0.5) | val 0.456, train 0.410 → stable |
| 3,080 (1.0) | weak Ridge val **0.842 ± 0.08** → interpolation peak, unstable across folds |
| 9,240 (3.0) | weak Ridge train 0.002, val 0.815 → memorises noise; **tuned Ridge val 0.448 ± 0.009** |

*Conclusion:* p < n is the safe regime; p ≈ n is unstable (ill-conditioned XᵀX); p > n works only with strong regularisation. This is why P1 is tuned and P2/P3 stay small.

**Selection / reduction on P1 features** (inside training folds only):

| Method | p kept | CV RMSE | MAE | Fit | Jaccard stability |
|---|---|---|---|---|---|
| None | 8,284 | 0.4520 | 0.3457 | 7.89 s | – |
| **L1 / Lasso (≤ 300)** | 300 | **0.4546** | 0.3514 | **1.44 s** | 0.40 |
| PCA 100 | 100 | 0.4847 | 0.3763 | 2.13 s | – |
| SelectKBest f-reg | 300 | 0.5062 | 0.3912 | 1.00 s | **0.81** |
| Top-20 % variance | 1,672 | 0.5423 | 0.4232 | 4.23 s | 0.24 |

*Conclusion:* L1 removes 96.4 % of features, trains 5.5× faster, costs only +0.0026 RMSE. Variance thresholding fails (high-variance pixels track lighting, not faces). Univariate selection is stable but weaker; Lasso picks correlated substitutes (lower Jaccard, higher accuracy).

## 5. Error inspection and fairness (T9, T10)
- **Blur (top 10 % errors, n = 82, |err| ≥ 0.72, vs n = 736):** sharpness 760.9 vs 821.1, Mann-Whitney **p = 0.0446**; brightness p = 0.43; contrast p = 0.26; face found 100 % vs 99.5 %. *Weak evidence* only: one of three tests, and it would not survive a Bonferroni correction (0.05/3 = 0.017).
- **Gallery (20 anonymous test images):** overestimates → well-lit, smiling, symmetric faces that raters scored lower; underestimates → head tilt / dark bangs where Haar misses the forehead and eyes; high disagreement → glasses and strong lighting.
- **Regression to the mean (quintiles of true rating):** bias R1 **+0.437**, R2 +0.159, R3 +0.041, R4 −0.088, R5 **−0.382** (MAE 0.454, 0.224, 0.201, 0.218, 0.412). Predictions span 1.85–4.10; truth spans 1.02–4.75. Cause: squared-error loss pulls extremes towards the mean (ȳ ≈ 2.98).
- **Groups (from filename metadata only; never predicted from faces):**

| Group | n | Mean true | Mean pred | Bias | MAE |
|---|---|---|---|---|---|
| Asian female | 315 | 3.027 | 3.056 | +0.029 | 0.356 |
| Asian male | 286 | 2.881 | 2.913 | +0.032 | 0.311 |
| Caucasian female | 115 | 3.196 | 3.085 | −0.111 | 0.325 |
| Caucasian male | 102 | 3.021 | 2.915 | −0.106 | 0.289 |

Kruskal-Wallis on |error|: H = 4.12, **p = 0.248** (no detectable difference in error size). The Caucasian groups show a −0.11 signed bias but have only ~100 images each; the cause is **not established** (training imbalance and rater-panel effects are hypotheses). Equal MAE does not prove fairness.




## 6. Deployment (T11)
`python predict.py face.jpg --bundle out/bundle.joblib` returns the mean of P1–P3, an **80 % interval [ŷ − 0.42, ŷ + 0.48]** (from 10th/90th percentile CV residuals), and warnings for: unreadable file, resolution < 100 px, non-square image, blur/brightness/contrast outside the training 5–95th percentile, no frontal face, > 30 % missing ratios, estimate outside the training range, or model spread > 0.18. A standing notice forbids use for hiring, admissions, dating filters, credit screening or judging real people.
**Failure cases:** side pose, occlusion (mask, glasses, hair), blur, tiny images, non-faces, other age/camera/crop styles. **Do not use** for any decision about a person, or on populations unlike the training panel.

## 7. Failed experiments and decisions
| # | Experiment | Outcome | Decision |
|---|---|---|---|
| 1 | Hamming threshold 6 | 75.5 % of data chained into duplicates | use threshold 0; check cluster sizes first |
| 2 | Haar geometry at 48 px | 27.5 % missing ratios; P2 weakest | keep 128 px for geometry |
| 3 | KMeans as preprocessing | η² < 0.05; Ridge RMSE 0.582 vs 0.464 with PCA | rejected; clusters track lighting |
| 4 | Drop colour features | RMSE +0.024 | kept, with bias audit |
| 5 | Mean vs median target | r = 0.984 | mean kept; ceiling documented |
| 6 | Range compression | extremes biased ±0.4 | unresolved; try isotonic calibration |

## 8. Limitations
One rater panel and one photo style; bounded 1–5 scale (regression to the mean); hash checks catch repeated *photos*, not repeated *people* (no identity labels); Haar detectors are fragile; four coarse groups with small test counts; the model's ceiling is rater agreement (r ≈ 0.93). Seeds, split manifest, scripts and `requirements.txt` are submitted for reproduction.
