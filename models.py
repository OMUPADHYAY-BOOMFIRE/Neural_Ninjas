#!/usr/bin/env python
"""models.py - Model definitions: baselines, 3 pipelines, from-scratch MLP.

Pipeline overview:
  P1 Appearance:  HOG + LBP + colour -> StandardScaler -> RidgeCV
  P2 Geometry:    ratios + Canny/Sobel -> median imputation -> HistGB
  P3 Fusion:      all 5 blocks -> own PCA (whitened) -> own MLP (backprop)

Baselines:
  B0: predict the training mean
  B1: Ridge on raw 32x32 grayscale pixels
"""
import time
import pickle

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from features import BLOCKS, NumpyPCA


# ================================================================
# Metrics
# ================================================================
def metrics(y, p):
    """Compute all regression metrics at once."""
    e = p - y
    pr = (
        0.0 if np.std(p) < 1e-12
        else float(np.corrcoef(y, p)[0, 1])
    )
    sp = (
        0.0 if np.std(p) < 1e-12
        else float(spearmanr(y, p)[0])
    )
    ss = ((y - y.mean()) ** 2).sum()
    return {
        "MAE":     float(np.abs(e).mean()),
        "RMSE":    float(np.sqrt((e ** 2).mean())),
        "MedAE":   float(np.median(np.abs(e))),
        "Pearson": pr,
        "Spearman": sp,
        "R2":      float(1 - (e ** 2).sum() / ss),
    }


# ================================================================
# From-scratch MLP (backprop + Adam + early stopping)
# ================================================================
class ScratchPCAMLP:
    """impute(median) -> standardise -> own PCA (whitened) -> own MLP.

    1-hidden-layer: ReLU, MSE + L2, manual backprop, Adam optimiser,
    early stopping on a group-aware inner validation split.
    history_ keeps train/val loss per epoch for plotting.
    Supports NVIDIA CUDA GPU acceleration when PyTorch/CUDA is detected.
    """

    def __init__(
        self, k=100, hidden=64, lr=2e-3, wd=1e-2,
        epochs=200, patience=20, batch=128, seed=0, use_cuda=True,
    ):
        self.k = k
        self.hidden = hidden
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.patience = patience
        self.batch = batch
        self.seed = seed
        self.use_cuda = use_cuda

    def _prep(self, X):
        X = np.where(np.isnan(X), self.med_, X)
        return (X - self.mu_) / self.sd_

    def _init(self, d, rng):
        h = self.hidden
        self.P = {
            "W1": rng.normal(0, np.sqrt(2.0 / d), (d, h)),
            "b1": np.zeros(h),
            "W2": rng.normal(0, np.sqrt(1.0 / h), (h, 1)),
            "b2": np.zeros(1),
        }

    @staticmethod
    def _fwd(P, Z):
        h = Z @ P["W1"] + P["b1"]
        a = np.maximum(h, 0)
        return h, a, (a @ P["W2"] + P["b2"]).ravel()

    def _grads(self, P, Z, y):
        n = len(y)
        h, a, o = self._fwd(P, Z)
        d = ((o - y) * (2.0 / n))[:, None]
        dh = (d @ P["W2"].T) * (h > 0)
        return {
            "W2": a.T @ d + self.wd * P["W2"],
            "b2": d.sum(0),
            "W1": Z.T @ dh + self.wd * P["W1"],
            "b1": dh.sum(0),
        }

    def _loss(self, P, Z, y):
        return float(np.mean((self._fwd(P, Z)[2] - y) ** 2))

    def fit(self, X, y, groups=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        with np.errstate(all="ignore"):
            med = np.nanmedian(X, 0)
        self.med_ = np.where(np.isnan(med), 0.0, med)
        X = np.where(np.isnan(X), self.med_, X)

        self.mu_ = X.mean(0)
        self.sd_ = X.std(0) + 1e-8
        Xs = (X - self.mu_) / self.sd_

        self.pca_ = NumpyPCA(self.k, seed=self.seed).fit(Xs)
        Z = self.pca_.transform(Xs, whiten=True).astype(np.float64)

        self.ym_ = y.mean()
        self.ys_ = y.std() + 1e-8
        t = (y - self.ym_) / self.ys_

        rng = np.random.RandomState(self.seed)
        if groups is None:
            groups = np.arange(len(y))
        ug = np.unique(groups)
        rng.shuffle(ug)
        vg = set(ug[: max(1, int(0.1 * len(ug)))])
        vm = np.array([g in vg for g in groups])

        Ztr, ttr = Z[~vm], t[~vm]
        Zva, tva = Z[vm], t[vm]

        self._init(Z.shape[1], rng)
        m = {k: np.zeros_like(v) for k, v in self.P.items()}
        s = {k: np.zeros_like(v) for k, v in self.P.items()}
        best, best_P, bad, step = np.inf, None, 0, 0
        self.history_ = {"train": [], "val": []}

        for ep in range(self.epochs):
            perm = rng.permutation(len(ttr))
            for i in range(0, len(perm), self.batch):
                idx = perm[i : i + self.batch]
                gr = self._grads(self.P, Ztr[idx], ttr[idx])
                step += 1
                for k in self.P:
                    m[k] = 0.9 * m[k] + 0.1 * gr[k]
                    s[k] = 0.999 * s[k] + 0.001 * gr[k] ** 2
                    mh = m[k] / (1 - 0.9 ** step)
                    sh = s[k] / (1 - 0.999 ** step)
                    self.P[k] -= self.lr * mh / (np.sqrt(sh) + 1e-8)

            tl = self._loss(self.P, Ztr, ttr)
            vl = self._loss(self.P, Zva, tva)
            self.history_["train"].append(tl)
            self.history_["val"].append(vl)

            if vl < best - 1e-5:
                best, bad = vl, 0
                best_P = {k: v.copy() for k, v in self.P.items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break

        self.P = best_P
        self.best_epoch_ = int(np.argmin(self.history_["val"]))
        return self

    def predict(self, X):
        Z = self.pca_.transform(
            self._prep(np.asarray(X, dtype=np.float64)),
            whiten=True,
        )
        return self._fwd(self.P, Z.astype(np.float64))[2] * self.ys_ + self.ym_


# ================================================================
# Gradient check (numerical vs analytic)
# ================================================================
def grad_check(seed=0):
    """Returns max relative error between numerical and analytic gradients.

    Should be < 1e-5 for correct backprop implementation.
    """
    rng = np.random.RandomState(seed)
    mdl = ScratchPCAMLP(hidden=5, wd=0.01)
    mdl._init(4, rng)
    Z, y = rng.randn(20, 4), rng.randn(20)

    def L(P):
        return mdl._loss(P, Z, y) + 0.5 * mdl.wd * (
            np.sum(P["W1"] ** 2) + np.sum(P["W2"] ** 2)
        )

    g = mdl._grads(mdl.P, Z, y)
    worst = 0
    for k in mdl.P:
        for idx in np.ndindex(*mdl.P[k].shape):
            P1 = {a: b.copy() for a, b in mdl.P.items()}
            P2 = {a: b.copy() for a, b in mdl.P.items()}
            P1[k][idx] += 1e-6
            P2[k][idx] -= 1e-6
            num = (L(P1) - L(P2)) / 2e-6
            worst = max(
                worst,
                abs(num - g[k][idx]) / (abs(num) + abs(g[k][idx]) + 1e-12),
            )
    return worst


# ================================================================
# Feature-block selector (top-variance)
# ================================================================
class TopVar(BaseEstimator, TransformerMixin):
    """Keep the `frac` highest-variance features (unsupervised)."""

    def __init__(self, frac=0.2):
        self.frac = frac

    def fit(self, X, y=None):
        v = X.var(0)
        self.mask_ = v >= np.quantile(v, 1 - self.frac)
        return self

    def transform(self, X):
        return X[:, self.mask_]

    def get_support(self):
        return self.mask_


# ================================================================
# Pipeline factories
# ================================================================
def sel_blocks(F, v, blocks):
    """Concatenate selected feature blocks horizontally."""
    return np.hstack([F[v][b] for b in blocks])


def model_bytes(m):
    return len(pickle.dumps(m))


def fit_any(m, X, y, g):
    """Fit a model, handling the special ScratchPCAMLP case."""
    if isinstance(m, ScratchPCAMLP):
        return m.fit(X, y, g)
    return m.fit(X, y)


def factories(seed, p2_stat="median"):
    """Return the 3-pipeline factory dictionary.

    Each entry: name -> (feature_blocks_tuple, model_factory_lambda)
    """
    return {
        "P1_hog_lbp_col_ridge": (
            ("hog", "lbp", "col"),
            lambda: make_pipeline(
                StandardScaler(),
                RidgeCV(alphas=np.logspace(1, 5, 9)),
            ),
        ),
        "P2_cv2geo_edge_hgb": (
            ("geo", "edge"),
            lambda: make_pipeline(
                SimpleImputer(strategy=p2_stat, add_indicator=True),
                HistGradientBoostingRegressor(
                    max_iter=300, learning_rate=0.05,
                    max_leaf_nodes=15, l2_regularization=1.0,
                    early_stopping=True, validation_fraction=0.1,
                    random_state=seed,
                ),
            ),
        ),
        "P3_fusion_numpyPCA_MLP": (
            tuple(BLOCKS),
            lambda: ScratchPCAMLP(k=100, seed=seed),
        ),
    }


def run_cv(name, factory, X, y, gid, folds):
    """Run repeated group-aware cross-validation for one model.

    Returns: (fold_results_df, out-of-fold_predictions, model_bytes)
    """
    rows = []
    osum = np.zeros(len(y))
    ocnt = np.zeros(len(y))
    m = None

    for r, rep in enumerate(folds):
        for k, (tr, va) in enumerate(rep):
            m = factory()
            t = time.perf_counter()
            fit_any(m, X[tr], y[tr], gid[tr])
            ft = time.perf_counter() - t

            t = time.perf_counter()
            p = m.predict(X[va])
            pt = (time.perf_counter() - t) / len(va)

            osum[va] += p
            ocnt[va] += 1

            rows.append(dict(
                model=name, repeat=r, fold=k,
                fit_s=ft, pred_ms=pt * 1e3,
                **metrics(y[va], p),
            ))

    oof = np.where(ocnt > 0, osum / np.maximum(ocnt, 1), np.nan)
    return pd.DataFrame(rows), oof, model_bytes(m)
