# Estimating Human Ratings from Face Images — Classical ML Study
**Report focus: Student 3 – P3 – From-Scratch Fusion (Custom PCA + Backprop MLP + Adam)**  |  Name: ________  Roll No.: ________  |  Submitted to Dr. Shibna Hussain
**Neural Ninjas (Batch A3) · AIP301 Machine Learning Lab · IIIT Kota · Nov 2026**
Dataset: SCUT-FBP5500 (5,500 faces, mean rating 1–5, ~60 raters/image). Constraint: classical ML + OpenCV only (no CNN, no pretrained/transfer models).

---

## 1. Verdict (what the data supports)
Supervised **regression**. On an unseen, group-disjoint test set (n = 818) the 3-pipeline ensemble reaches **MAE 0.3272, Pearson r 0.8033, R² 0.6272** (86.6 % of the rater-agreement ceiling r = 0.927).
My implementation (P3) fulfills the assignment requirement for **substantial original code written from scratch** without high-level library estimators:
1. **Custom Randomized SVD PCA** in pure NumPy (Halko et al., 2011) compressing 8,458 features down to 100 whitened components (98.8 % compression).
2. **Custom Feedforward Neural Network (MLP)** with analytical backpropagation, L2 weight decay, and the Adam optimizer in raw NumPy.
3. **Rigorous Numerical Gradient Check:** backpropagation verified against two-sided finite differences with relative error **$4.05 \times 10^{-9}$** (exceeding standard $< 10^{-7}$ verification criteria).
P3 achieves **CV RMSE 0.4640** (−32.5 % vs baseline B0) and trains in just **1.156 s** (6.9× faster than P1 Ridge).

---

## 2. Shared sequential backbone (identical for every pipeline)

| Stage | What we did | Key numbers | Reason |
|---|---|---|---|
| Audit (T1) | File read & array validation | 5,500 images, 0 corrupt, uniform 350×350; rating 1.017–4.750, mean 2.981, median 2.900, SD 0.688 | Confirm clean data before mathematical derivation |
| Duplicates (T1) | 9×8 gray → 64-bit dHash → Hamming (XOR) → Disjoint Sets | **thr 6: 4,150 imgs (75.5 %) chained into one cluster**; **thr 0: 39 groups, 88 imgs** → 49 moved to `duplicate/` → **5,451 unique** | Graph percolation at thr 6 causes catastrophic train-test leakage; thr 0 enforces strict duplicate isolation |
| Target (T2) | mean vs median vs 10 % trimmed | MAE(mean,median) 0.082; r(mean,median) 0.984; rater SD 0.684; split-half r 0.865 → Spearman-Brown ceiling **0.927** | Mean kept for label compatibility; ceiling proves no model can exceed r ≈ 0.93 |
| Features (T3) | 5 blocks × 2 variants | **low_rgb** 128 px: 8,458 d, 46.8 ms/img; **high_gray** 48 px: 1,244 d, 6.3 ms/img (7.4× faster) | Multi-modal feature fusion across appearance, texture, color, edges, and ratios |
| Split (T5) | group-disjoint 70/15/15 + repeated group CV | train 3,817 / val 816 / test 818; dev∩test groups = **0**; min test Hamming dist to dev = **1 bit** | Disjoint sets prevent identical duplicate leakage across folds |

All scalers, PCA projection matrices, and network weights are derived strictly from training folds. Test evaluation occurs once at the end.

---

## 3. P3 – From-Scratch Fusion (Custom PCA + Backprop MLP) – deep dive (owner: Student 3)
Repeated group CV: 3 repeats × 5 folds on dev (n = 4,633); mean ± SD across 15 folds. Owner rows in bold.

| Model | Features | CV MAE | CV RMSE | MedAE | r | RMSE vs B0 | Fit Time | Model Size |
|---|---|---|---|---|---|---|---|---|
| B0 train mean | – | 0.568 | 0.6875 | 0.505 | 0.000 | −0 % | 0.001 s | 1 KB |
| B1 raw 32×32 ridge | 1,024 | 0.449 | 0.5745 | 0.358 | 0.549 | −16.4 % | 1.65 s | 30 KB |
| P1 Appearance low_rgb | 8,284 | 0.344 | 0.4477 | 0.272 | 0.759 | −34.9 % | 7.96 s | 233 KB |
| P2 Geometry low_rgb | 174 | 0.368 | 0.4752 | 0.301 | 0.725 | −30.9 % | 2.73 s | 3.68 MB |
| **P3 From-scratch low_rgb** | **8,458→100** | **0.356** | **0.4640** | **0.284** | **0.739** | **−32.5 %** | **1.16 s** | **185 KB** |
| P1 high_gray | 1,076 | 0.375 | 0.4814 | 0.301 | 0.714 | −30.0 % | 1.77 s | 31 KB |
| P2 high_gray | 168 | 0.390 | 0.5009 | 0.321 | 0.687 | −27.1 % | 2.54 s | 3.52 MB |
| **P3 high_gray** | **1,244→100** | **0.377** | **0.4861** | **0.304** | **0.712** | **−29.3 %** | **1.12 s** | **180 KB** |

### 3.1 Mathematical Architecture & Algorithm (Written from Scratch)
P3 concatenates all 5 feature blocks (HOG 8,100 + LBP 160 + COL 24 + EDGE 128 + GEO 46 = 8,458 dimensions) and processes them through two mathematical stages:

#### 1. From-Scratch Whitened Randomized SVD PCA (`NumpyPCA`)
Given centered training matrix $\mathbf{X}_c \in \mathbb{R}^{n \times p}$:
1. Draw random Gaussian matrix $\boldsymbol{\Omega} \in \mathbb{R}^{p \times (k + s)}$ with oversampling parameter $s = 10$.
2. Form sample matrix $\mathbf{Y} = \mathbf{X}_c \boldsymbol{\Omega}$, and apply $q = 2$ power iterations: $\mathbf{Y} \leftarrow \mathbf{X}_c (\mathbf{X}_c^T \mathbf{Y})$.
3. Compute orthonormal basis $\mathbf{Q} \in \mathbb{R}^{n \times (k+s)}$ via thin QR factorization: $\mathbf{Q}, \mathbf{R} = \text{qr}(\mathbf{Y})$.
4. Project to small matrix $\mathbf{B} = \mathbf{Q}^T \mathbf{X}_c \in \mathbb{R}^{(k+s) \times p}$ and compute standard SVD: $\tilde{\mathbf{U}}, \mathbf{S}, \mathbf{V}^T = \text{svd}(\mathbf{B})$.
5. Retain top $k = 100$ components: projection matrix $\mathbf{W}_{\text{pca}} = \mathbf{V}[:, :k] \in \mathbb{R}^{p \times 100}$.
6. Whitened projection: $\mathbf{Z} = \mathbf{X}_c \mathbf{W}_{\text{pca}} \text{diag}(\mathbf{S}[:k] / \sqrt{n-1} + \epsilon)^{-1}$.
*Why 100 components is optimal:* 100 components compresses 8,458 dimensions down by **98.8 %**, retains $> 88 \%$ of total variance, decorrelates features, and eliminates covariance matrix ill-conditioning.

#### 2. From-Scratch 2-Layer Feedforward Network (`ScratchPCAMLP`)
- **Forward Pass:**
  $$\mathbf{h}_1 = \mathbf{Z} \mathbf{W}_1 + \mathbf{b}_1 \in \mathbb{R}^{B \times 64}, \quad \mathbf{a}_1 = \max(0, \mathbf{h}_1) \quad (\text{ReLU})$$
  $$\hat{\mathbf{y}} = \mathbf{a}_1 \mathbf{W}_2 + b_2 \in \mathbb{R}^{B \times 1}$$
- **Objective Function (MSE with L2 Weight Decay):**
  $$\mathcal{L}(\theta) = \frac{1}{2B} \sum_{i=1}^B (\hat{y}_i - y_i)^2 + \frac{\lambda}{2} \left( \|\mathbf{W}_1\|_F^2 + \|\mathbf{W}_2\|_F^2 \right)$$
- **Analytical Gradients (Manual Backpropagation):**
  $$\boldsymbol{\delta}_2 = \frac{1}{B} (\hat{\mathbf{y}} - \mathbf{y})$$
  $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_2} = \mathbf{a}_1^T \boldsymbol{\delta}_2 + \lambda \mathbf{W}_2, \quad \frac{\partial \mathcal{L}}{\partial b_2} = \sum_{i=1}^B \delta_{2,i}$$
  $$\boldsymbol{\delta}_1 = (\boldsymbol{\delta}_2 \mathbf{W}_2^T) \odot \mathbb{I}(\mathbf{h}_1 > 0)$$
  $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \mathbf{Z}^T \boldsymbol{\delta}_1 + \lambda \mathbf{W}_1, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}_1} = \sum_{i=1}^B \boldsymbol{\delta}_{1,i}$$
- **Adam Optimizer Updates:**
  $$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
  $$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}, \quad \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
  with $\eta = 2 \times 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\lambda = 0.01$.

### 3.2 Verification: Analytical vs Numerical Gradient Check
To ensure our backpropagation equations were mathematically flawless, we implemented a central finite-difference gradient check on synthetic mini-batches:
$$g_{\text{num}}(\theta_j) = \frac{\mathcal{L}(\theta_j + \epsilon) - \mathcal{L}(\theta_j - \epsilon)}{2\epsilon}, \quad \epsilon = 10^{-6}$$
$$\text{Relative Error} = \frac{\|g_{\text{backprop}} - g_{\text{num}}\|_2}{\|g_{\text{backprop}}\|_2 + \|g_{\text{num}}\|_2}$$
- **Empirical Result:** Relative error = **$4.05 \times 10^{-9}$**.
- This completely rules out silent derivative implementation errors and confirms gradient descent steps are exact.

### 3.3 Training Dynamics & Early Stopping
- Mini-batch size: $B = 128$. Max epochs: 200.
- Inner validation split: $15 \%$ of training fold held out as inner validation monitor.
- Patience: 20 epochs without validation loss improvement restores best weights.
- Empirical convergence: Typically converges within 45–60 epochs, preventing over-fitting to noisy annotator labels.

---

## 4. Empirical Dimensionality Study (Task 6: $p < n, p \approx n, p > n$)
Using fine-cell HOG features ($p_{\max} = 14,000$) on fixed training fold size $n_{\text{train}} \approx 3,080$:

| Dimension Regime | Features ($p$) | Ratio $p/n$ | Train RMSE (Weak Ridge) | Val RMSE (Weak Ridge $\alpha=0.01$) | Val RMSE (Tuned RidgeCV) | Mathematical Insight |
|---|---|---|---|---|---|---|
| Underfitting | 154 | $0.05$ | 0.485 | 0.521 | 0.518 | High bias: too few degrees of freedom |
| Classical Safe | 1,540 | $0.50$ | 0.410 | 0.456 | 0.452 | Stable regime: well-conditioned $\mathbf{X}^T\mathbf{X}$ |
| **Interpolation Peak** | **3,080** | **$1.00$** | **0.298** | **0.842 ± 0.08** | **0.449 ± 0.01** | $\sigma_{\min}(\mathbf{X}^T\mathbf{X}) \to 0$: condition number explodes |
| Over-parameterized | 9,240 | $3.00$ | 0.002 | 0.815 ± 0.07 | **0.448 ± 0.009** | Weak Ridge memorizes noise; Tuned Ridge scales $\alpha \propto p$ |

*Theoretical Analysis:*
1. **The $p = n$ Singularity:** When $p = n$, the empirical Gram matrix $\mathbf{X}^T\mathbf{X}$ becomes rank-deficient. For weak regularization ($\alpha \to 0$), the inversion $(\mathbf{X}^T\mathbf{X} + \alpha \mathbf{I})^{-1}$ amplifies high-frequency noise, creating a massive spike in validation variance (RMSE $0.842$).
2. **Why Tuned Ridge & P3 Succeed:** Tuned Ridge shrinks the effective degrees of freedom $\text{df}(\alpha) = \sum_{i=1}^n \frac{\sigma_i^2}{\sigma_i^2 + \alpha}$, neutralizing the spike. Similarly, P3 projects 8,458 features onto $k = 100$ principal components, guaranteeing $k \ll n$ ($100 \ll 3,080$) so the covariance matrix is always strictly well-conditioned.

---

## 5. Preprocessing Comparison: PCA vs KMeans (Task 5 / Task 7)
We empirically evaluated whether unsupervised KMeans clustering could serve as a non-linear feature extractor by computing Euclidean distance to $k$ cluster centroids:

| Method | Dimension ($p$) | CV RMSE | CV MAE | Target Variance Explained ($\eta^2$) | Verdict |
|---|---|---|---|---|---|
| **PCA (Linear Subspace)** | 100 | **0.4640** | **0.3561** | **> 88 % (Reconstruction)** | **Adopted: Preserves continuous manifold** |
| KMeans Cluster Distances | $k = 4$ | 0.5824 | 0.4592 | $\eta^2 = 0.038$ (< 4 %) | Rejected: Clusters track background lighting |
| KMeans Cluster Distances | $k = 8$ | 0.5791 | 0.4551 | $\eta^2 = 0.046$ (< 5 %) | Rejected: Near baseline performance |

*Mathematical Reason:* Faces lie on a continuous smooth manifold of varying geometry and texture. Discrete Voronoi cells from KMeans shatter this manifold into discrete step functions, discarding subtle continuous gradations.

---

## 6. Team Debates & Losing Hypotheses
1. **Debate: PCA Component Count ($k = 20$ vs $k = 100$ vs $k = 300$).**
   - *Student 1:* Wanted $k = 20$ to maximize speed.
   - *Student 3 (My Position):* $k = 20$ explained only 62% of variance, worsening RMSE to 0.504. $k = 100$ captured 88.4% variance with RMSE 0.464. $k = 300$ gave no further RMSE gain (0.463) while tripling matrix multiplication cost.
   - *Decision:* Adopted $k = 100$.
2. **Debate: Deep MLP (3 layers) vs Shallow (1 hidden layer).**
   - *Outcome:* 3-hidden-layer network ($100 \to 128 \to 64 \to 32 \to 1$) took 14 seconds to train and suffered from severe validation loss plateaus without improving test MAE (0.359 vs 0.356).
   - *Decision:* Kept elegant, fast 1-hidden-layer architecture with Adam optimizer.

---

## 7. Submission Checklist
- [x] Substantial original code implemented in pure NumPy (Randomized SVD + Feedforward MLP + Adam).
- [x] Numerical gradient check confirms backprop error $< 10^{-8}$ ($4.05 \times 10^{-9}$).
- [x] Empirical study of $p < n, p \approx n, p > n$ rigorously documented.
- [x] Zero leakage: PCA basis and network scalers fit strictly on training folds.
