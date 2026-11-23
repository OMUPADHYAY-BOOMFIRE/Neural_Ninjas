Implementation Plan: Estimating Human Ratings from Face Images
Team: Neural Ninjas (Batch A3) | Course: Machine Learning Lab (AIP301) | Approach: classical ML only (no CNN, no transfer learning)
Status note: the code (run_study.py, fbp_core.py, predict.py) was tested end to end on synthetic images only. No result from the real 5,500-image dataset exists yet. Every expectation below is a hypothesis to be tested, not a finding.
0. Guiding principles
1.  The grade rewards reasoning, not the best score. Every choice must be justified, compared with an alternative, and recorded, including the failures.
2.  The target is the average opinion of one rater panel. It is not objective beauty. This sentence goes in the report.
3.  Nothing that touches the test set may influence any decision. This covers feature ratios, scalers, imputation values, PCA, cluster centres, hyperparameters and the choice of final model.
4.  Same face in train and test is the biggest threat. It creates a falsely high score, so leakage control comes before modelling.
5.  Supervised regression is the core. Unsupervised methods (PCA, KMeans, hashing) are helpers for preprocessing, compression and duplicate detection. Semi-supervised learning is not used; state why in the report (labels exist for every image, so there is nothing unlabeled to exploit).
1. Problem framing (what is being asked)
1.  Input: a face photograph. Output: one continuous number, the predicted mean rating. This is regression, so accuracy, precision and a true confusion matrix do not apply.
2.  A confusion matrix is still drawn on low/mid/high thirds of the rating. It is labelled as a derived view for visual comparison, never as the main metric.
3.  No ready-made features exist, so we design them ourselves from pixels using classical computer vision.
4.  The brief asks us to decide whether the data supports reliable prediction. A well-argued "only partly" is a valid answer.
2. Team roles (mixed thinking styles on purpose)
Role
Owns
Strength
Blind spot to watch
Skeptic (auditor)
Tasks 1, 2, 5
Finds leakage and label problems early
Can over-check and slow the team
Classicist
P1, P2, Task 7
Interpretable, cheap, explainable
May dismiss complex models too early
Mathematician
P3 (from-scratch code), Task 6
Understands why models behave as they do
May over-engineer
Engineer
Pipeline, caching, predict.py, timing and size logs
Reproducibility
May ignore interpretation
Ethicist
Tasks 9, 10, 11, limitations
Honesty about fairness and misuse
Less focus on raw accuracy
A 15-minute debate is held at each decision point (section 14). The outcome and the losing argument are written down, because the report asks for failed experiments.
3. Environment and reproducibility
1.  Python 3.10+. Install with pip install -r requirements.txt.
2.  OpenCV must be 4.x. OpenCV 5 cannot parse the Haar cascade XML files that P2 and P3 depend on. The requirements file pins opencv-python-headless<5.
3.  Fix the random seed (--seed 42) and log package versions in the report.
4.  The split manifest (split_manifest.csv) stores file name, duplicate group, train/val/test role and CV fold for every image. It is frozen once created and submitted with the report.
5.  Features are extracted once in one parallel disk pass. All later experiments use the cached arrays, so repeated experiments are cheap.
6.  Run a smoke test first: python run_study.py --data_dir IMAGES --labels labels.txt --out out_test --limit 500 --fast. Then the full run: python run_study.py --data_dir IMAGES --labels labels.txt --ratings All_Ratings.xlsx --out out.
4. Stage 1: Data source and audit (Task 1)
Procedure
1.  Verify the licence, source and annotation method of the Kaggle copy. Write these in the report.
2.  Confirm the dataset has about 5,500 images and about 60 raters per image (check, do not assume).
3.  Parse the label file (name score). Parse the filename prefix into metadata: first letter A/C (ethnicity group), second letter M/F (gender). Never infer these from the face.
4.  Count images, labels without files, files without labels, missing scores, score min and max.
5.  Detect corrupt files. cv2.imread returning nothing marks a file corrupt, and it is excluded and counted.
6.  Record image sizes. Training images are expected to be uniform, but verify.
7.  Plot the target distribution and the score distribution per metadata group.
Reasoning
• Silent data problems (unreadable files, missing labels, a skewed or truncated rating range) are cheaper to catch now than to explain later.
• Image sharpness is plotted too, because blur is one of the error explanations used in Task 9.
Outputs: task1_audit.json, fig01_target_distribution.png, task3_feature_table.csv.
5. Stage 2: Duplicate and leakage control (Tasks 1 and 5)
Why duplicates mislead: if the same photo (or the same person) appears in both train and test, the model can memorise it and the test score becomes falsely high. The reported accuracy would then describe memory, not generalisation.
Procedure
1.  Compress each image to a tiny 9x8 grayscale matrix and turn it into a 64-bit difference hash (dHash). Compressing first cuts comparison cost and ignores small brightness or resizing changes.
2.  Compare hashes by Hamming distance. Two images with distance at or below the threshold (default 6 of 64) are treated as duplicates.
3.  Split the comparison into 4 buckets by filename prefix and run the buckets in parallel on the CPU. Each bucket is much smaller than the full set. (GPU is not needed: about 15 million 64-bit comparisons are fast on a CPU.)
4.  Add a cross-bucket exact-match check, so a duplicate stored under a different prefix is still caught.
5.  Merge duplicate pairs into groups with union-find. Every group gets one group ID.
6.  If more than 5% of images land in duplicate groups, the script warns that the threshold is too loose. Lower --dup_thr and rerun.
7.  After the split, run a second, independent check: for every test image find the nearest training image by Hamming distance and by cosine similarity in a pixel-PCA space (PCA fit on training data only). Plot both histograms.
Limitations to state honestly
• A hash catches the same photo, not the same person in a different photo. True subject-disjoint splitting needs identity labels, which are absent.
• Bucketing by prefix assumes the prefix labels are correct.
• What we can claim: no duplicate or near-duplicate photo crosses the train/test boundary at the chosen thresholds. What we cannot claim: no person appears on both sides.
Outputs: task5_leakage_check.json, fig05_leakage_check.png, split_manifest.csv.
6. Stage 3: Splits (Task 5)
1.  Hold out the test set (15%) by duplicate group, before any modelling. It is touched exactly once, at the end.
2.  From the remaining data, hold out a fixed validation set (about 15% of the total) by group. It is used for learning curves and quick checks.
3.  The remaining data (train plus validation, called "dev") is used for repeated group-aware K-fold cross-validation (default 3 repeats x 5 folds). All images of a duplicate group stay in the same fold.
4.  Also report a plain random split for comparison (brief Task 5 asks for "a normal random split" too). Expect it to look slightly better if any leakage remains. A large gap between random and group-aware results is itself evidence of leakage.
5.  Rule for every later step: anything computed from data (means, medians, scalers, PCA, cluster centres, selected features, ratio thresholds) is fit on training rows only and then applied to validation or test rows.
7. Stage 4: Target definition (Task 2)
Procedure
1.  If a per-rater file exists (e.g. All_Ratings.xlsx), compute per image: mean, median, 10% trimmed mean, standard deviation, inter-quartile range, rater count.
2.  Compare the definitions: mean absolute gap between mean and median, correlation between them, and the average rater standard deviation.
3.  Compute split-half reliability: randomly split raters into two halves, correlate the half-means, and apply Spearman-Brown. This gives a ceiling for any model, because a model cannot agree with the raters better than the raters agree with themselves.
4.  Choose the target with a stated rule. Default: mean, because it is the panel's average opinion and is what the label file already stores. Switch to median or trimmed mean (--target median|trim) if the data show skew or outlying raters.
5.  If only the mean is available, say so. Median and disagreement cannot then be computed, and the high-disagreement part of Task 9 uses disagreement between our own pipelines as a proxy (clearly labelled).
What the target does and does not measure
• Does: the average judgement of this rater panel, on a 1 to 5 scale, at one time, for one set of photos.
• Does not: objective beauty, a person's worth, attractiveness to other cultures or ages, or anything about the person other than the photo.
• A bounded 1 to 5 scale makes predictions shrink toward the mean. The residual-versus-true plots will show it.
Outputs: task2_target_comparison.csv, fig02_rater_disagreement.png.
8. Stage 5: Image representations (Task 3)
Two compression variants run in parallel on every image, so compression is a measured experiment and not a guess.
Variant
Size
Colour
Purpose
low_rgb (low compression)
128x128
RGB/HSV
More detail, higher cost
high_gray (high compression)
48x48
grayscale
Fast, small, loses detail
Five feature blocks are computed per variant. Dimension, extraction time per image and storage are logged in task3_feature_table.csv.
1.  HOG (histogram of oriented gradients). Edge and contrast direction in small cells. Captures face outline, nose, eyes, mouth and hair/forehead boundaries without learning. Large dimension (about 8,100 at 128px, about 900 at 48px).
2.  LBP (local binary patterns, 4x4 grid). Micro-texture such as skin and hair texture, fairly robust to lighting changes.
3.  Colour histograms. HSV (RGB variant) or intensity (gray variant). Captures overall tone and lighting.
4.  cv2 edge grid. Canny edge density and Sobel gradient magnitude on an 8x8 grid. This is the "OpenCV edge detection" side of the HOG-vs-OpenCV comparison.
5.  Geometry ratios (the hand-designed features). Found with Haar cascades (classical detectors) and Canny:
    ◦ Face width/height, face area fraction, horizontal offset.
    ◦ Eye separation, width, height, aspect ratio, vertical position, area, tilt, eye count.
    ◦ Mouth width, height, vertical position, bright-pixel fraction as a teeth-visibility proxy.
    ◦ 3x3 zone edge density and zone brightness over a canonical 64x64 face crop.
    ◦ Left/right symmetry (pixel and edge), jaw width at two heights.
    ◦ Forehead darkness fraction and forehead-to-cheek brightness ratio (hair-coverage proxy).
    ◦ Cheek and forehead colour statistics (RGB variant only).
    ◦ A face-found flag.
Why these may carry information: edges and proportions capture structure (symmetry, feature spacing), texture captures skin and hair, colour captures tone and lighting. Raters respond to all of these, plus things we cannot measure (expression, grooming, photo style).
Cautions
• Haar detectors fail on some faces and at 48px. Missing ratios are left as NaN, imputed using training-set statistics only, and the failure rate is reported. Failure at 48px is a real experimental finding, not a bug.
• Skin-tone and hair-colour statistics correlate with ethnicity. They can make the model rate groups differently. This must be discussed in the fairness section, and an ablation without colour features is a good follow-up.
• Images are resized to a square. If originals are not square, ratios are distorted (check the audit).
9. Stage 6: PCA versus KMeans at the start (unsupervised preprocessing)
The aim is to decide which unsupervised helper is more useful, using dev data only.
1.  PCA (own implementation) on standardised 32x32 grayscale pixels. Plot cumulative explained variance. Compression value: how few components keep most variance.
2.  KMeans for k = 2 to 8 on the first 30 PCA components. Plot silhouette score (cluster quality) and eta-squared of the rating (how much of the rating variance the clusters explain).
3.  Cluster the data into 4 groups. Show each cluster's size and mean rating, and a cluster-by-metadata-group heat map. If clusters simply rediscover the metadata groups, they are not finding "attractiveness"; they are finding photo style or group differences.
4.  Regression comparison: Ridge on PCA features (10, 30, 100 components) versus Ridge on KMeans-distance features (k = 4, 16, 32), same folds. The lower RMSE is the better unsupervised preprocessing for regression.
5.  Compare both with the predict-the-mean RMSE line. If neither beats it, say so.
Expected trade-off (hypothesis): PCA keeps continuous information and should help regression more. KMeans gives interpretable groups but throws away within-cluster variation.
Output: fig_pca_vs_kmeans.png, task_pca_vs_kmeans.csv.
10. Stage 7: Ratio study: average versus median (the "iterate to find the best ratios" idea)
1.  For every ratio feature (dev data only), compute Spearman correlation with the rating.
2.  Define "high-rated" as the top 25% of dev ratings. A fixed 2.5 cut is available through --hi_thr, but most images in the sample sit above 2.5, so it separates little.
3.  For each ratio, compare the median (and mean) in high-rated and other images, expressed as an effect size.
4.  For missing ratios, decide between mean and median imputation by repeated CV on dev data, separately for each compression variant. The median is expected to win if ratios are skewed or contain detector outliers.
5.  Do not train only on high-rated people. That truncates the target range and biases the model. The high-rated group is for profiling ratios only.
Output: fig_ratio_study.png, task_ratio_study.csv.
11. Stage 8: Baselines and the three parallel pipelines (Task 4)
Baselines
• B0: predict the training mean. Any model that cannot beat it has learned nothing.
• B1: Ridge on raw 32x32 grayscale pixels. This shows how much the engineered features add over raw pixels.
The three pipelines
Pipeline
Features
Model
Pros
Cons
P1 Appearance
HOG + LBP + colour
Standard scaler, Ridge (alpha chosen by internal CV)
Fast, stable, weight vector is inspectable
Not face-specific, sensitive to pose and lighting, high dimension
P2 Geometry + cv2 edges
Geometry ratios + Canny/Sobel grid (about 170 features)
Histogram gradient boosting
Small p (below n), explainable ratios, handles NaN
Detector failures, ignores texture, lower ceiling
P3 Fusion, from scratch
All five blocks
Own NumPy PCA (randomised SVD), own NumPy MLP (backprop, Adam, L2, early stopping on a group-aware inner split)
Satisfies "substantial original code", shows the combination of feature groups
Slowest, easiest to get wrong, less accurate than libraries
How features are combined and reduced: P1 and P2 each use their own blocks. P3 concatenates all blocks, standardises them, and reduces with PCA to 100 whitened components before the network. Ridge assigns each feature a weight w = (XᵀX + αI)⁻¹Xᵀy; larger alpha shrinks weights. This is the same weight-vector idea as a linear SVM.
From-scratch code checklist (P3)
1.  Numerical gradient check of backprop (the script prints the maximum relative error; it should be about 1e-5 or smaller).
2.  Loss curves per epoch for training and validation, with the early-stopping epoch marked.
3.  Report where early stopping triggered and what the network does if it is removed.
Optional bonus: the mean of the three pipelines (an ensemble). The spread (standard deviation) between pipelines is used as an uncertainty signal in predict.py. Check that the spread correlates with the actual error. If correlation is weak, say so.
12. Stage 9: Evaluation protocol (Task 8)
1.  Metrics: MAE, RMSE, Pearson, Spearman, R squared, and median absolute error (the robust measure that a few bad images cannot move much).
2.  Each metric is reported as mean ± standard deviation over 3 repeats x 5 group-aware folds (15 scores).
3.  Costs are reported next to accuracy: number of features, extraction time per image, storage, fit time, prediction time per image, model size (pickled MB).
4.  Hyperparameters are tuned inside training folds only (RidgeCV, boosting early stopping, MLP early stopping).
5.  The final model is chosen by cross-validation on dev data, never by test performance.
6.  Final test evaluation happens once, after refitting on dev data. Bootstrap 95% intervals (300 resamples) are reported for MAE and RMSE.
7.  Learning curves: error versus number of training images, for train and validation. A widening gap means overfitting; a flat gap means more data is unlikely to help.
Outputs: cv_all_folds.csv, task8_cv_summary.csv, test_results.csv, fig08_metric_comparison.png, fig_scoreboard_and_cost.png, fig_learning_curves.png, fig_confusion_matrices.png, fig_residuals.png.
13. Stage 10: Feature dimension and selection studies (Tasks 6 and 7)
Dimension study (Task 6)
1.  Build a very high-dimensional representation (fine HOG with 6-pixel cells, about 14,000 features).
2.  Choose a random, target-blind order of columns. Use the first p columns for p = 5%, 25%, 50%, 90%, 100%, 110%, 200% and 300% of the training-set size.
3.  Fit a weakly regularised Ridge (near-unregularised) and a tuned Ridge at each p, using 3-fold group CV.
4.  Plot validation RMSE (with fold standard deviation) and training RMSE against p, with a vertical line at p = n.
5.  Hypotheses to test: below n, stable but may underfit; around n, validation error and fold-to-fold variation peak for the weak model; above n, the weak model interpolates the training data (training error near zero) and generalises poorly, while the tuned model recovers.
6.  Conclusion to defend in the report: with limited, noisy labels, p well below n is the safe choice, and p above n needs strong regularisation or reduction. This is a result to verify on real data.
Selection and reduction study (Task 7) on the P1 features, all selection done inside each training fold:
Method
Idea
Expected strength
Expected weakness
None
Use all
Simple baseline
Large model, slow
Top-20% variance
Keep the most variable features
Fast, unsupervised
Variance is not relevance
SelectKBest (f_regression, k = 300)
Keep features most correlated with the rating
Fast, target-aware
Misses interactions
L1 / Lasso (up to 300)
Sparse linear selection
Interpretable
Slow, can be unstable
PCA (100)
Reduce, not select
Compact, stable
Components are hard to interpret
Measured for each: CV RMSE ± std, MAE, number of features, fit time, model size, and selection stability (average Jaccard overlap of the selected feature sets across folds). PCA has no selected set, so stability is not applicable. Test data are never used.
Outputs: task6_dimension_study.csv, fig_performance_vs_dimension.png, task7_feature_selection.csv, fig_feature_selection.png.
14. Decision debates to hold and record
Decision
Option A
Option B
How we settle it
Target
Mean (standard, matches labels)
Median / trimmed (robust to rogue raters)
Skewness, mean-median gap, rater outliers (Stage 4)
Unsupervised start
PCA (compression, continuous)
KMeans (groups, interpretable)
RMSE of Ridge on each, plus cluster meaning (Stage 6)
Compression
128px colour (detail)
48px gray (speed, size)
CV error against extraction time and size, including detector failure rate
Missing ratios
Mean imputation
Median imputation
Repeated CV on dev data
Duplicate threshold
Strict (misses near-duplicates)
Loose (removes legitimate images)
Inspect pairs near the threshold by eye (as aggregate counts; do not name people)
Final model
Highest accuracy
Best accuracy per cost and explainability
Scoreboard plus cost heat map
Colour features
Keep (may improve accuracy)
Drop (proxy for ethnicity)
Ablation of accuracy versus group error gap
15. Stage 11: Error analysis (Task 9)
1.  On the test set, take the ensemble prediction and compute the residual (prediction minus true).
2.  Build a gallery of 20 test images: 7 largest overestimates, 7 largest underestimates, 6 highest-disagreement images (rater standard deviation if available, otherwise pipeline disagreement).
3.  Label images with an anonymous index, the true value, the prediction and the uncertainty figure. Do not use filenames or identity codes in labels or text. Use the gallery only inside the project; do not publish the faces.
4.  Test explanations with numbers, not impressions: compare image blur, brightness, contrast and face-detected rate between the worst 10% of errors and the rest (Mann-Whitney test, task9_error_vs_image_quality.csv).
5.  Describe patterns in neutral terms such as lighting, pose, blur, occlusion or low rater agreement. Do not describe individuals.
6.  Expect that high rater disagreement and mid-range ratings produce wide errors. Check, do not assume.
16. Stage 12: Group and rating-range analysis (Task 10)
Group labels are available from filenames, so fairness analysis is possible with those caveats.
1.  For each of the 4 metadata groups report n, mean true rating, mean predicted rating, mean signed error (bias), MAE and median absolute error. Add 95% bootstrap intervals for bias.
2.  Run a Kruskal-Wallis test of absolute error across groups. A small test-set group makes this low-powered; say so.
3.  Split the true rating into 5 quantile ranges and report bias and MAE per range. This shows regression to the mean: low ratings overestimated, high ratings underestimated.
4.  The question "the model predicted well but the panel rated differently" is covered by the group comparison of the mean true rating against the mean predicted rating, and by the residual per group. Do not train any model to predict ethnicity or gender from faces. That is forbidden by the brief and adds nothing.
5.  State limits: only 4 coarse groups, the labels come from the dataset, and the rater panel's own demographics are unknown. Equal error across groups does not prove fairness. It is one check only.
Outputs: task10_group_errors.csv, task10_range_errors.csv, fig_group_and_range_errors.png.
17. Stage 13: Inference script and failure analysis (Task 11)
predict.py IMAGE --bundle out/bundle.joblib returns:
1.  The estimate from each pipeline and their mean.
2.  An 80% interval taken from the 10th and 90th percentile of cross-validation residuals.
3.  The pipeline disagreement, compared with a threshold set from cross-validation.
4.  Warnings for: unreadable file, low resolution, non-square image, blur or brightness or contrast outside the training range, no face found, many geometry ratios missing, estimate outside the training range, pipelines disagreeing.
5.  A standing notice against use for hiring, admissions, dating filters, screening or judging people.
Failure cases to test on purpose and document: no face, two faces, side pose, heavy shadow, sunglasses or mask, very blurry image, very small image, cartoon or non-face image, a face unlike the training population (different age, camera, crop style), image with a different aspect ratio.
Conditions where the system should not be used: any decision about a real person, any setting that needs fairness guarantees, any population different from the training data, any use that treats the number as a truth about attractiveness.
18. Failed and negative experiments to record
Keep a running log with date, hypothesis, setting, result and decision. Likely entries:
1.  Detector failure rate at 48px, and what it does to P2.
2.  Whether the colour features raise the group error gap.
3.  Whether KMeans-distance features beat predict-the-mean at all.
4.  Whether the from-scratch network overfits, and where early stopping triggers.
5.  Whether near-duplicate thresholds remove too many legitimate images.
6.  Any pipeline that does not beat the Ridge-on-raw-pixels baseline.
7.  Whether the pipeline-disagreement signal predicts the real error.
19. Risks and mitigations
Risk
Mitigation
Duplicate threshold too loose or too tight
Warning in script, inspect borderline pairs, report counts
Leakage through preprocessing
Fit everything on training rows only; review the code path for every fit
Haar detectors fail
NaN plus missing-indicator, face-found flag, failure rate reported
Small group sizes in the test set
Report n and bootstrap intervals, avoid strong claims
Run time too long
Use --fast, cached features, fewer repeats during development
Memory (full image set)
Features stored as float32, 128px cap, parallel single disk pass
Results look too good
Check for leakage first: compare random split with group split
Ethical misuse
Standing notice in predict.py, limits written in the report
20. Timeline suggestion
1.  Day 1: environment, smoke test, licence and dataset checks, audit, target definition.
2.  Day 2: feature extraction on the full set, duplicate groups, splits, leakage check.
3.  Day 3: PCA vs KMeans, ratio study, baselines, repeated CV of P1, P2, P3.
4.  Day 4: dimension study, selection study, learning curves, first debate notes.
5.  Day 5: single test evaluation, residuals, gallery, group analysis, inference tests.
6.  Day 6: write the 4 to 6 page typed report; each member writes the hand-written reflection.
7.  Day 7: reproduce from scratch on a clean environment, final check of the checklist.
21. Task-to-output checklist
Task
Where it is covered
1 Audit and duplicates
task1_audit.json, fig01, fig05, section 5
2 Target
task2_target_comparison.csv, fig02, section 7
3 Representations
task3_feature_table.csv, section 8
4 Baseline and candidates
task8_cv_summary.csv, from-scratch P3, section 11
5 Splits and leakage
split_manifest.csv, task5_leakage_check.json, sections 5 and 6
6 Feature dimension
task6_dimension_study.csv, fig_performance_vs_dimension.png
7 Selection or reduction
task7_feature_selection.csv, fig_feature_selection.png
8 Metrics and variation
cv_all_folds.csv, fig08, fig_scoreboard_and_cost.png
9 Error inspection
fig_error_gallery.png, task9_error_vs_image_quality.csv
10 Groups and ranges
task10_*.csv, fig_group_and_range_errors.png
11 Inference and failure cases
predict.py, section 17
22. Report outline (4 to 6 pages, typed)
1.  Problem, dataset, licence and what the target means (half page).
2.  Audit, duplicates, split design and leakage checks (one page).
3.  Features, compression variants and costs (one page).
4.  Models, baselines, evaluation results with variation (one and a half pages).
5.  Dimension, selection and error studies (one page).
6.  Group analysis, failure cases, limitations and ethics (one page).
7.  Failed experiments and decisions from the debates (half page).
Hand-written reflection (one per member): what did I learn, what worked well, what was most challenging, what will I do differently next time. Use specific examples from the project log.
23. Honest limitations to state up front
1.  The rating reflects one rater panel and the culture and context of that dataset.
2.  Hash-based checks cannot prove that no person appears in both train and test.
3.  Geometry features depend on classical detectors that fail on some faces.
4.  Only four coarse metadata groups exist, so fairness cannot be fully established.
5.  Predictions shrink toward the mean because the scale is bounded.
6.  A model can reach at best the agreement the raters have with themselves (split-half ceiling).
7.  The system is a coursework study. It is not a tool for judging people.