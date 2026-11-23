#!/usr/bin/env python
"""run_study.py - full reproducible study (classical ML only, no CNN / no transfer learning).

python run_study.py --data_dir IMAGES --labels labels.txt [--ratings All_Ratings.xlsx] --out out
Add --limit 400 --fast for a smoke test.
"""
import argparse, json, pickle, re, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from joblib import Parallel, delayed, dump
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge, RidgeCV
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fbp_core import *

warnings.filterwarnings("ignore")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ================================================================ helpers
def read_labels(p):
    rows = []
    for line in open(p):
        s = line.strip().split()
        if len(s) >= 2:
            try:
                rows.append((s[0], float(s[1])))
            except ValueError:
                pass
    df = pd.DataFrame(rows, columns=["name", "score"])
    m = df.name.str.extract(r"^([AaCc])([MmFf])")
    df["ethnicity"] = m[0].str.upper().map({"A": "Asian", "C": "Caucasian"})
    df["gender"] = m[1].str.upper().map({"M": "Male", "F": "Female"})
    df["group"] = df.ethnicity.fillna("?") + " " + df.gender.fillna("?")
    return df


def load_ratings(path):
    """long (Rater, Filename, Rating) or wide (filename, r1..rN) -> per-image stats."""
    df = pd.read_excel(path) if str(path).endswith(("xlsx", "xls")) else pd.read_csv(path)
    low = {c.lower(): c for c in df.columns}
    rc = [c for l, c in low.items() if "rating" in l or "score" in l]
    fc = [c for l, c in low.items() if "file" in l or "image" in l or "name" in l]
    if fc and rc and any("rater" in l for l in low):
        long = df[[fc[0], rc[0]]].rename(columns={fc[0]: "name", rc[0]: "r"})
        wide = long.groupby("name")["r"].apply(list)
    else:
        wide = df.set_index(df.columns[0]).apply(lambda r: [x for x in r if pd.notna(x)], axis=1)
    out = pd.DataFrame({"name": wide.index})
    A = wide.values
    out["n_raters"] = [len(a) for a in A]
    out["r_mean"] = [np.mean(a) for a in A]
    out["r_median"] = [np.median(a) for a in A]
    out["r_trim"] = [stats.trim_mean(a, .1) for a in A]
    out["r_std"] = [np.std(a, ddof=1) if len(a) > 1 else np.nan for a in A]
    out["r_iqr"] = [stats.iqr(a) for a in A]
    # split-half reliability of the mean (ceiling for any model)
    rng = np.random.RandomState(0)
    h1, h2 = [], []
    for a in A:
        a = np.array(a); rng.shuffle(a)
        h1.append(a[:len(a) // 2].mean()); h2.append(a[len(a) // 2:].mean())
    out["h1"], out["h2"] = h1, h2
    return out


def savefig(fig, out, name):
    fig.tight_layout()
    fig.savefig(Path(out) / name, dpi=130)
    plt.close(fig)


def model_bytes(m):
    return len(pickle.dumps(m))


class TopVar(BaseEstimator, TransformerMixin):
    """keep the `frac` highest-variance features (unsupervised selector)."""

    def __init__(self, frac=.2):
        self.frac = frac

    def fit(self, X, y=None):
        v = X.var(0)
        self.mask_ = v >= np.quantile(v, 1 - self.frac)
        return self

    def transform(self, X):
        return X[:, self.mask_]

    def get_support(self):
        return self.mask_


def fit_any(m, X, y, g):
    if isinstance(m, ScratchPCAMLP):
        return m.fit(X, y, g)
    return m.fit(X, y)


def run_cv(name, factory, X, y, gid, folds):
    rows, osum, ocnt = [], np.zeros(len(y)), np.zeros(len(y))
    m = None
    for r, rep in enumerate(folds):
        for k, (tr, va) in enumerate(rep):
            m = factory()
            t = time.perf_counter(); fit_any(m, X[tr], y[tr], gid[tr]); ft = time.perf_counter() - t
            t = time.perf_counter(); p = m.predict(X[va]); pt = (time.perf_counter() - t) / len(va)
            osum[va] += p; ocnt[va] += 1
            rows.append(dict(model=name, repeat=r, fold=k, fit_s=ft, pred_ms=pt * 1e3,
                             **metrics(y[va], p)))
    oof = np.where(ocnt > 0, osum / np.maximum(ocnt, 1), np.nan)
    return pd.DataFrame(rows), oof, model_bytes(m)


def sel_blocks(F, v, blocks):
    return np.hstack([F[v][b] for b in blocks])


def factories(seed, p2_stat="median"):
    return {
        "P1_hog_lbp_col_ridge": (("hog", "lbp", "col"),
                                 lambda: make_pipeline(StandardScaler(),
                                                       RidgeCV(alphas=np.logspace(1, 5, 9)))),
        "P2_cv2geo_edge_hgb": (("geo", "edge"),
                               lambda: make_pipeline(
                                   SimpleImputer(strategy=p2_stat, add_indicator=True),
                                   HistGradientBoostingRegressor(
                                       max_iter=300, learning_rate=.05, max_leaf_nodes=15,
                                       l2_regularization=1., early_stopping=True,
                                       validation_fraction=.1, random_state=seed))),
        "P3_fusion_numpyPCA_MLP": (tuple(BLOCKS), lambda: ScratchPCAMLP(k=100, seed=seed)),
    }


# ================================================================ pipeline diagram
def draw_pipeline(out):
    fig, ax = plt.subplots(figsize=(17, 9)); ax.axis("off"); ax.set_xlim(0, 17); ax.set_ylim(0, 9)

    def box(x, y, w, h, t, c="#dbeafe"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.05", fc=c, ec="k", lw=1))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8)
        return (x, y, w, h)

    def arr(a, b):
        ax.add_patch(FancyArrowPatch((a[0] + a[2], a[1] + a[3] / 2), (b[0], b[1] + b[3] / 2),
                                     arrowstyle="->", mutation_scale=12, lw=1))
    seq = [("1 Load + audit\n(corrupt, sizes,\nmissing, range)", "#e5e7eb"),
           ("2 Hash dedup\n(dHash, 4 buckets in\nparallel + union-find)", "#fde68a"),
           ("3 Group split\ntest 15% | val 15%\n| train (by dup-group)", "#fde68a"),
           ("4 Target: mean vs\nmedian vs trimmed;\nrater std", "#e5e7eb"),
           ("5 PCA vs KMeans\n(dev only)", "#e5e7eb"),
           ("6 Baselines\nB0 mean | B1 raw32 ridge", "#fecaca")]
    prev = None
    for i, (t, c) in enumerate(seq):
        b = box(.2 + i * 2.8, 7.4, 2.5, 1.3, t, c)
        if prev: arr(prev, b)
        prev = b
    ax.text(.2, 9.0, "SEQUENTIAL stage (fit on train/dev only)", fontsize=10, weight="bold")
    ax.text(.2, 6.7, "PARALLEL stage - one feature pass, two compression variants "
            "(low_rgb 128px colour | high_gray 48px gray)", fontsize=10, weight="bold")
    P = [("P1  Appearance\nHOG + LBP + colour hist\n-> scaler -> Ridge (weights w)", 4.6, "#bbf7d0"),
         ("P2  Geometry + cv2 edges\nHaar boxes, Canny/Sobel zones,\nratios -> HistGB", 2.9, "#bbf7d0"),
         ("P3  Fusion, from-scratch\nall blocks -> own PCA -> own MLP\n(backprop, Adam, early stop)", 1.2, "#bbf7d0")]
    src = box(.2, 3.6, 2.6, 1.3, "Feature store\n(hog, lbp, col,\nedge, geo)", "#e0e7ff")
    outs = []
    for t, y, c in P:
        b = box(4.0, y, 4.4, 1.3, t, c)
        arr(src, b); outs.append(b)
    ev = box(9.6, 3.6, 2.4, 1.3, "Repeated group CV\n(R x K folds)\nOOF predictions", "#ddd6fe")
    for b in outs: arr(b, ev)
    last = ev
    for i, t in enumerate(["Dimension study\np<n, p~n, p>n", "Feature selection\nstability / size / time",
                           "Test ONCE: metrics,\nconf. matrix, residuals"]):
        b = box(12.4, 5.2 - i * 1.6, 2.3, 1.3, t, "#fbcfe8")
        arr(ev, b)
    fin = box(15.0, 3.6, 1.8, 1.6, "Ensemble +\nerror gallery,\ngroup audit,\npredict.py", "#fed7aa")
    ax.text(12.4, 6.65, "SEQUENTIAL analysis", fontsize=9, weight="bold")
    ax.add_patch(FancyArrowPatch((14.7, 4.4), (15.0, 4.4), arrowstyle="->", mutation_scale=12))
    ax.set_title("Pipeline: sequential preprocessing -> 3 parallel pipelines -> sequential evaluation",
                 fontsize=12)
    savefig(fig, out, "fig00_pipeline_diagram.png")


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--ratings", default=None)
    ap.add_argument("--out", default="out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dup_thr", type=int, default=6, help="max Hamming distance (of 64) = duplicate")
    ap.add_argument("--hi_thr", type=float, default=None, help="'high-rated' cut; default = train 75th pct")
    ap.add_argument("--target", default="mean", choices=["mean", "median", "trim"])
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    seed = a.seed
    K, R = (3, 1) if a.fast else (5, 3)
    plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": .3})
    draw_pipeline(out)
    log("grad-check of from-scratch MLP, max rel. error =", f"{grad_check():.2e}")

    # ------------------------------------------------ 1 audit
    lab = read_labels(a.labels)
    files = {}
    for ext in ("jpg", "jpeg", "png", "bmp"):
        for p in Path(a.data_dir).rglob(f"*.{ext}"):
            files[p.name] = p
    audit = {"n_label_rows": int(len(lab)), "n_image_files": len(files),
             "labels_without_file": int((~lab.name.isin(files)).sum()),
             "files_without_label": int(len(set(files) - set(lab.name))),
             "missing_scores": int(lab.score.isna().sum()),
             "score_min": float(lab.score.min()), "score_max": float(lab.score.max()),
             "unparsed_group": int(lab.ethnicity.isna().sum() + lab.gender.isna().sum())}
    lab = lab[lab.name.isin(files)].drop_duplicates("name").reset_index(drop=True)
    if a.limit:
        lab = lab.sample(min(a.limit, len(lab)), random_state=seed).reset_index(drop=True)
    log(f"extracting features for {len(lab)} images (one disk pass, parallel) ...")
    res = Parallel(n_jobs=a.jobs, batch_size=16)(delayed(extract_one)(files[n]) for n in lab.name)
    ok = np.array([r is not None for r in res])
    audit["corrupt_files"] = int((~ok).sum())
    lab = lab[ok].reset_index(drop=True)
    res = [r for r in res if r is not None]
    N = len(lab)
    audit["n_valid"] = N
    shapes = pd.Series([str(r["shape"]) for r in res]).value_counts()
    audit["image_sizes_top"] = shapes.head(5).to_dict()
    y_given = lab.score.values.astype(float)

    F = {v: {b: np.stack([r[v][0][b] for r in res]) for b in BLOCKS} for v in VARIANTS}
    raw32 = np.stack([r["raw32"] for r in res])
    Q = np.stack([r["quality"] for r in res])
    G128 = np.stack([r["g128"] for r in res])
    hashes = np.array([r["dhash"] for r in res], dtype=np.uint64)
    tm = {v: {b: float(np.mean([r[v][1][b] for r in res])) for b in BLOCKS} for v in VARIANTS}
    feat_tab = pd.DataFrame([dict(variant=v, block=b, dim=F[v][b].shape[1],
                                  extract_ms_per_img=tm[v][b] * 1e3,
                                  storage_MB=F[v][b].nbytes / 1e6,
                                  nan_frac=float(np.isnan(F[v][b]).mean()))
                             for v in VARIANTS for b in BLOCKS])
    feat_tab.to_csv(out / "task3_feature_table.csv", index=False)
    log("\n" + feat_tab.round(3).to_string(index=False))

    # ------------------------------------------------ 2 duplicates (compressed-hash set matching)
    prefix = lab.group.values
    def bucket_pairs(idx):
        h = hashes[idx]; pr = []
        for i in range(len(idx) - 1):
            d = hamming_row(h[i], h[i + 1:])
            pr += [(idx[i], idx[i + 1 + j], int(d[j])) for j in np.where(d <= a.dup_thr)[0]]
        return pr
    buckets = [np.where(prefix == p)[0] for p in np.unique(prefix)]
    pairs = sum(Parallel(n_jobs=min(4, len(buckets)))(delayed(bucket_pairs)(b) for b in buckets), [])
    _, inv, cnt = np.unique(hashes, return_inverse=True, return_counts=True)   # cross-bucket exact
    for u in np.where(cnt > 1)[0]:
        m = np.where(inv == u)[0]
        pairs += [(m[0], j, 0) for j in m[1:]]
    par = np.arange(N)
    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for i, j, _ in pairs:
        par[find(i)] = find(j)
    gid = np.array([find(i) for i in range(N)])
    audit["duplicate_pairs"] = len(pairs)
    audit["images_in_dup_groups"] = int((pd.Series(gid).map(pd.Series(gid).value_counts()) > 1).sum())
    audit["n_groups"] = int(len(np.unique(gid)))
    if audit["images_in_dup_groups"] > .05 * N:
        log(f"WARNING: {audit['images_in_dup_groups']} images sit in duplicate groups - --dup_thr {a.dup_thr} may be too loose; lower it")
    json.dump(audit, open(out / "task1_audit.json", "w"), indent=2, default=str)
    log("AUDIT", json.dumps(audit, default=str))

    # ------------------------------------------------ 3 target (mean vs median vs trimmed)
    y = y_given.copy(); std_img = np.full(N, np.nan); have_r = False
    if a.ratings:
        rt = load_ratings(a.ratings).drop_duplicates("name").set_index("name")
        have_r = lab.name.isin(rt.index).mean() > .9
    if have_r:
        rr = rt.reindex(lab.name).reset_index()
        std_img = rr.r_std.values
        col = {"mean": "r_mean", "median": "r_median", "trim": "r_trim"}[a.target]
        y = rr[col].values
        tgt = pd.DataFrame({"mean_vs_median_MAE": [np.abs(rr.r_mean - rr.r_median).mean()],
                            "mean_vs_trim_MAE": [np.abs(rr.r_mean - rr.r_trim).mean()],
                            "corr_mean_median": [np.corrcoef(rr.r_mean, rr.r_median)[0, 1]],
                            "mean_rater_std": [np.nanmean(std_img)],
                            "split_half_r": [np.corrcoef(rr.h1, rr.h2)[0, 1]]})
        r_ = tgt.split_half_r[0]
        tgt["spearman_brown_ceiling_r"] = 2 * r_ / (1 + r_)
        tgt.to_csv(out / "task2_target_comparison.csv", index=False); log("TARGET\n" + tgt.round(4).to_string())
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
        ax[0].hist(rr.r_mean, 40, alpha=.6, label="mean"); ax[0].hist(rr.r_median, 40, alpha=.6, label="median")
        ax[0].hist(rr.r_trim, 40, alpha=.4, label="10% trimmed"); ax[0].legend(); ax[0].set_title("Target definitions")
        ax[1].hist(std_img, 40, color="C3"); ax[1].set_title("Rater disagreement: std per image")
        ax[2].scatter(rr.r_mean, std_img, s=4, alpha=.4); ax[2].set_xlabel("mean rating"); ax[2].set_ylabel("rater std")
        ax[2].set_title("Disagreement vs rating")
        savefig(fig, out, "fig02_rater_disagreement.png")
    else:
        log("NOTE: no per-rater file -> median/disagreement cannot be computed; target = given mean score")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].hist(y, 40, color="C0"); ax[0].axvline(np.mean(y), c="r", label="mean"); ax[0].axvline(np.median(y), c="g", ls="--", label="median")
    ax[0].legend(); ax[0].set_title(f"Target distribution (skew={stats.skew(y):.2f})")
    lab.assign(y=y).boxplot(column="y", by="group", ax=ax[1]); ax[1].set_title("Target by metadata group"); plt.sca(ax[1]); plt.xticks(rotation=20)
    ax[2].hist(Q[:, 0], 40); ax[2].set_title("Image sharpness (Laplacian var)")
    savefig(fig, out, "fig01_target_distribution.png")

    # ------------------------------------------------ 4 splits (group-disjoint) + manifest
    idx = np.arange(N)
    dev, test = next(GroupShuffleSplit(1, test_size=.15, random_state=seed).split(idx, groups=gid))
    tr, va = next(GroupShuffleSplit(1, test_size=.176, random_state=seed).split(dev, groups=gid[dev]))
    train, val = dev[tr], dev[va]
    man = pd.DataFrame({"name": lab.name, "dup_group": gid, "split": "train"})
    man.loc[val, "split"] = "val"; man.loc[test, "split"] = "test"
    folds = make_folds(dev, gid, K, R, seed)
    for r_, rep in enumerate(folds):
        man[f"cv_rep{r_}_fold"] = -1
        for k_, (_, v_) in enumerate(rep):
            man.loc[v_, f"cv_rep{r_}_fold"] = k_
    man.to_csv(out / "split_manifest.csv", index=False)
    log(f"split sizes train/val/test = {len(train)}/{len(val)}/{len(test)}; "
        f"group overlap dev-test = {len(set(gid[dev]) & set(gid[test]))}")

    # leakage check: nearest dev neighbour of every test image (hash + pixel-PCA cosine)
    pc = NumpyPCA(50, seed=seed).fit(raw32[dev])
    Zd, Zt = pc.transform(raw32[dev]), pc.transform(raw32[test])
    Zd /= np.linalg.norm(Zd, axis=1, keepdims=True) + 1e-9; Zt /= np.linalg.norm(Zt, axis=1, keepdims=True) + 1e-9
    cos_nn = (Zt @ Zd.T).max(1)
    ham_nn = np.array([hamming_row(hashes[t], hashes[dev]).min() for t in test])
    leak = {"test_with_hamming<=%d" % a.dup_thr: int((ham_nn <= a.dup_thr).sum()),
            "test_with_cos>0.98": int((cos_nn > .98).sum()), "n_test": len(test)}
    json.dump(leak, open(out / "task5_leakage_check.json", "w"), indent=2); log("LEAK CHECK", leak)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(ham_nn, 30); ax[0].axvline(a.dup_thr, c="r"); ax[0].set_title("test->train min Hamming (dHash 64 bit)")
    ax[1].hist(cos_nn, 30); ax[1].set_title("test->train max cosine (pixel-PCA)")
    savefig(fig, out, "fig05_leakage_check.png")

    # ------------------------------------------------ 5 PCA vs KMeans (dev only)
    Xs = StandardScaler().fit(raw32[dev]).transform(raw32[dev])
    pc2 = NumpyPCA(min(100, len(dev) - 2), seed=seed).fit(Xs)
    Zp = pc2.transform(Xs); ydev = y[dev]
    sil, eta = {}, {}
    for k in range(2, 9):
        km = KMeans(k, n_init=5, random_state=seed).fit(Zp[:, :30])
        sil[k] = silhouette_score(Zp[:, :30], km.labels_, sample_size=min(2000, len(dev)), random_state=seed)
        cm = pd.Series(ydev).groupby(km.labels_).transform("mean").values
        eta[k] = ((cm - ydev.mean()) ** 2).sum() / ((ydev - ydev.mean()) ** 2).sum()
    km4 = KMeans(4, n_init=10, random_state=seed).fit(Zp[:, :30])
    cl = pd.DataFrame({"c": km4.labels_, "y": ydev, "g": lab.group.values[dev]})
    # regression comparison inside dev with the repeated folds
    def cv_feat(Xf):
        s = []
        for tr_, va_ in folds[0]:
            pos = {v: i for i, v in enumerate(dev)}
            ti, vi = [pos[i] for i in tr_], [pos[i] for i in va_]
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 11))).fit(Xf[ti], ydev[ti])
            s.append(np.sqrt(np.mean((m.predict(Xf[vi]) - ydev[vi]) ** 2)))
        return np.mean(s), np.std(s)
    comp = {f"PCA{k}": cv_feat(Zp[:, :min(k, Zp.shape[1])]) for k in (10, 30, 100)}
    for k in (4, 16, 32):
        comp[f"KMeans{k}-dist"] = cv_feat(KMeans(k, n_init=3, random_state=seed).fit(Zp[:, :30]).transform(Zp[:, :30]))
    pd.DataFrame(comp, index=["RMSE_mean", "RMSE_std"]).T.to_csv(out / "task_pca_vs_kmeans.csv")
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    ax[0, 0].plot(np.cumsum(pc2.evr_), "o-"); ax[0, 0].set_title("PCA cumulative explained variance (raw 32x32)")
    ax[0, 1].plot(list(sil), list(sil.values()), "o-", label="silhouette"); ax[0, 1].plot(list(eta), list(eta.values()), "s-", label="eta^2 of rating")
    ax[0, 1].legend(); ax[0, 1].set_title("KMeans: k vs cluster quality / rating separation")
    cl.groupby("c").y.agg(["mean", "std", "count"]).reset_index().plot.bar(x="c", y="mean", yerr="std", ax=ax[0, 2], legend=False)
    ax[0, 2].set_title("k=4 clusters: mean rating (n on bars)")
    for i, n_ in enumerate(cl.groupby("c").y.count().values): ax[0, 2].text(i, 0.1, str(n_), ha="center", color="w")
    ct = pd.crosstab(cl.c, cl.g, normalize="index")
    im = ax[1, 0].imshow(ct.values, cmap="viridis"); ax[1, 0].set_xticks(range(ct.shape[1])); ax[1, 0].set_xticklabels(ct.columns, rotation=30, fontsize=7)
    ax[1, 0].set_title("cluster composition by metadata group\n(do clusters just rediscover the group?)"); ax[1, 0].grid(False); plt.colorbar(im, ax=ax[1, 0])
    ax[1, 1].scatter(Zp[:, 0], Zp[:, 1], c=km4.labels_, s=5, cmap="tab10"); ax[1, 1].set_title("PCA-2D coloured by KMeans(4)")
    cdf = pd.DataFrame(comp, index=["m", "s"]).T
    ax[1, 2].bar(cdf.index, cdf.m, yerr=cdf.s); ax[1, 2].tick_params(axis="x", rotation=40)
    ax[1, 2].axhline(np.std(ydev), c="r", ls="--", label="predict-mean RMSE"); ax[1, 2].legend(); ax[1, 2].set_title("Ridge CV RMSE: PCA features vs KMeans-distance features")
    savefig(fig, out, "fig_pca_vs_kmeans.png")

    # ------------------------------------------------ 6 ratio study (train-only statistics)
    Ftr = F["low_rgb"]["geo"][dev]
    gn = geo_names(True)
    hi = a.hi_thr if a.hi_thr is not None else float(np.quantile(ydev, .75))
    hm = ydev >= hi
    log(f"high-rated cut = {hi:.3f}; share >=2.5 in dev = {(ydev >= 2.5).mean():.2%}")
    rows = []
    for j, n_ in enumerate(gn):
        c = Ftr[:, j]; m = ~np.isnan(c)
        if m.sum() < 30 or np.std(c[m]) == 0: continue
        rho = stats.spearmanr(c[m], ydev[m])[0]
        mh, ml = np.median(c[m & hm]), np.median(c[m & ~hm])
        rows.append(dict(ratio=n_, missing=1 - m.mean(), spearman=rho, median_high=mh, median_low=ml,
                         mean_high=np.mean(c[m & hm]), mean_low=np.mean(c[m & ~hm]),
                         eff=(mh - ml) / (np.std(c[m]) + 1e-9)))
    rt_ = pd.DataFrame(rows).sort_values("spearman", key=np.abs, ascending=False)
    rt_.to_csv(out / "task_ratio_study.csv", index=False)
    top = rt_.head(15)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].barh(top.ratio[::-1], top.spearman[::-1]); ax[0].set_title("Spearman(ratio, rating) - dev only")
    ax[1].barh(top.ratio[::-1], top.eff[::-1], color="C1"); ax[1].set_title(f"(median high-rated - median rest)/sd, cut={hi:.2f}")
    savefig(fig, out, "fig_ratio_study.png")

    # ------------------------------------------------ 7 repeated CV: baselines + 3 pipelines x 2 variants
    results, oofs, sizes, dims = [], {}, {}, {}
    p2_choice = {}
    for v in VARIANTS:
        # decide mean-vs-median imputation for ratios by CV on dev (never on test)
        sc = {}
        for st in ("mean", "median"):
            fz = factories(seed, st)["P2_cv2geo_edge_hgb"]
            d_, _, _ = run_cv(f"P2_{st}", fz[1], sel_blocks(F, v, fz[0]), y, gid, folds[:1])
            sc[st] = d_.RMSE.mean()
        p2_choice[v] = min(sc, key=sc.get)
        log(f"[{v}] ratio-imputation statistic: {sc} -> {p2_choice[v]}")
        fac = factories(seed, p2_choice[v])
        fac = {"B0_train_mean": ((), lambda: DummyRegressor()),
               "B1_raw32_ridge": ("raw", lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 4, 9)))),
               **fac}
        for nm, (bl, fn) in fac.items():
            X = raw32 if bl == "raw" else (np.zeros((N, 1)) if nm.startswith("B0") else sel_blocks(F, v, bl))
            key = f"{nm}|{v}"
            d_, oof, sz = run_cv(key, fn, X, y, gid, folds)
            results.append(d_); oofs[key] = oof; sizes[key] = sz; dims[key] = X.shape[1]
            log(f"{key:42s} RMSE {d_.RMSE.mean():.4f}±{d_.RMSE.std():.4f}  MAE {d_.MAE.mean():.4f}  r {d_.Pearson.mean():.3f}")
    cvdf = pd.concat(results)
    cvdf.to_csv(out / "cv_all_folds.csv", index=False)
    mets = ["MAE", "RMSE", "MedAE", "Pearson", "Spearman", "R2"]
    summ = cvdf.groupby("model")[mets + ["fit_s", "pred_ms"]].agg(["mean", "std"])
    summ.columns = [f"{c}_{s}" for c, s in summ.columns]
    summ["n_features"] = pd.Series(dims); summ["model_MB"] = pd.Series(sizes) / 1e6
    summ["n_train_ref"] = len(dev) * (K - 1) // K
    ext = {f"{k}|{v}": sum(tm[v][b] for b in bl) * 1e3
           for v in VARIANTS for k, (bl, _) in factories(seed).items()}
    summ["extract_ms_per_img"] = pd.Series(ext)
    summ.round(4).to_csv(out / "task8_cv_summary.csv")
    log("\n" + summ[["MAE_mean", "RMSE_mean", "RMSE_std", "MedAE_mean", "Pearson_mean", "n_features", "fit_s_mean", "model_MB"]].round(3).to_string())

    # ensemble (mean of the 3 pipelines) and uncertainty from disagreement
    best_v = min(VARIANTS, key=lambda v: np.mean([cvdf[cvdf.model == f"{p}|{v}"].RMSE.mean() for p in factories(seed)]))
    log("best compression variant by CV =", best_v)
    pk = [f"{p}|{best_v}" for p in factories(seed)]
    oof_mat = np.stack([oofs[k] for k in pk], 1)
    ens_oof = oof_mat.mean(1); dis_oof = oof_mat.std(1)
    devm = ~np.isnan(ens_oof)
    log("ensemble OOF", {k: round(v_, 4) for k, v_ in metrics(y[devm], ens_oof[devm]).items()},
        "| corr(disagreement, |err|) =", round(np.corrcoef(dis_oof[devm], np.abs(ens_oof - y)[devm])[0, 1], 3))

    # ---- fig: metric bars with variation, heatmap, cost bars
    names = list(summ.index)
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    for a_, m_ in zip(ax.ravel(), mets):
        a_.bar(range(len(names)), summ[f"{m_}_mean"], yerr=summ[f"{m_}_std"], color=plt.cm.tab20(np.arange(len(names))), capsize=3)
        a_.set_xticks(range(len(names))); a_.set_xticklabels(names, rotation=75, ha="right", fontsize=7); a_.set_title(f"{m_} (mean±std over {R}x{K} folds)")
    savefig(fig, out, "fig08_metric_comparison.png")
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    H = summ[[f"{m}_mean" for m in mets]].copy()
    Hn = (H - H.min()) / (H.max() - H.min() + 1e-12)
    for m in ("Pearson", "Spearman", "R2"): Hn[f"{m}_mean"] = 1 - Hn[f"{m}_mean"]       # lower = better everywhere
    ax[0].imshow(Hn.values, cmap="RdYlGn_r", aspect="auto"); ax[0].grid(False)
    ax[0].set_yticks(range(len(names))); ax[0].set_yticklabels(names, fontsize=7); ax[0].set_xticks(range(len(mets))); ax[0].set_xticklabels(mets)
    for i in range(len(names)):
        for j, m in enumerate(mets): ax[0].text(j, i, f"{H.iloc[i, j]:.3f}", ha="center", va="center", fontsize=7)
    ax[0].set_title("Scoreboard (green = best per metric)")
    cost = summ[["fit_s_mean", "pred_ms_mean", "model_MB", "n_features"]].copy()
    cost["extract_ms_per_img"] = summ.extract_ms_per_img.fillna(0) if "extract_ms_per_img" in summ else 0
    ax[1].imshow(np.log10(cost.values.astype(float) + 1e-3), cmap="Blues", aspect="auto"); ax[1].grid(False)
    ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels(names, fontsize=7); ax[1].set_xticks(range(cost.shape[1])); ax[1].set_xticklabels(["fit s", "predict ms/img", "model MB", "#features", "extract ms/img"], rotation=15)
    for i in range(len(names)):
        for j in range(cost.shape[1]): ax[1].text(j, i, f"{cost.iloc[i, j]:.3g}", ha="center", va="center", fontsize=7)
    ax[1].set_title("Cost (log colour scale): time / size / dimension")
    savefig(fig, out, "fig_scoreboard_and_cost.png")

    # ------------------------------------------------ 8 learning curves (epoch + data-size)
    fac = factories(seed, p2_choice[best_v])
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
    m3 = ScratchPCAMLP(k=100, seed=seed).fit(sel_blocks(F, best_v, BLOCKS)[train], y[train], gid[train])
    ax[0].plot(m3.history_["train"], label="train MSE (std. units)"); ax[0].plot(m3.history_["val"], label="inner-val MSE")
    ax[0].axvline(m3.best_epoch_, c="k", ls=":", label=f"early stop @ {m3.best_epoch_}"); ax[0].legend(); ax[0].set_xlabel("epoch"); ax[0].set_title("P3 from-scratch MLP: epoch learning curve")
    rng = np.random.RandomState(seed); fr = [.2, .4, .6, .8, 1.]
    lc = {}
    for nm, (bl, fn) in fac.items():
        Xb = sel_blocks(F, best_v, bl); trm, vam = [], []
        for f in fr:
            sub = rng.permutation(train)[:max(30, int(f * len(train)))]
            m = fn(); fit_any(m, Xb[sub], y[sub], gid[sub])
            trm.append(np.sqrt(np.mean((m.predict(Xb[sub]) - y[sub]) ** 2))); vam.append(np.sqrt(np.mean((m.predict(Xb[val]) - y[val]) ** 2)))
        lc[nm] = (trm, vam)
        ax[1].plot(np.array(fr) * len(train), vam, "o-", label=nm.split("_")[0]); ax[2].plot(np.array(fr) * len(train), trm, "o--", label=nm.split("_")[0])
    ax[1].set_title("Validation RMSE vs #train images"); ax[2].set_title("Train RMSE vs #train images (gap = overfitting)")
    for a_ in ax[1:]: a_.legend(); a_.set_xlabel("train images")
    savefig(fig, out, "fig_learning_curves.png")

    # ------------------------------------------------ 9 dimension study  (p < n, p ~ n, p > n)
    log("dimension study: fine HOG (6px cells) ...")
    fine = np.stack(Parallel(n_jobs=a.jobs, batch_size=32)(delayed(f_hog)(g, 6) for g in G128)).astype(np.float32)
    colperm = np.random.RandomState(seed).permutation(fine.shape[1])      # random, target-blind column order
    dfolds = make_folds(dev, gid, 3, 1 if a.fast else 2, seed)
    n_tr = len(dev) * 2 // 3
    ps = sorted({min(int(f * n_tr), fine.shape[1]) for f in (.05, .25, .5, .9, 1., 1.1, 2., 3.)})
    drows = []
    for p in ps:
        for rep in dfolds:
            for tr_, va_ in rep:
                Xa = fine[:, colperm[:p]]; sc = StandardScaler().fit(Xa[tr_])
                A, B = sc.transform(Xa[tr_]), sc.transform(Xa[va_])
                for nm, mdl in (("weak ridge (alpha=0.01)", Ridge(alpha=.01)), ("tuned ridge (CV alpha)", RidgeCV(alphas=np.logspace(0, 5, 8)))):
                    try:
                        mdl.fit(A, y[tr_])
                        drows.append(dict(p=p, model=nm, train=np.sqrt(np.mean((mdl.predict(A) - y[tr_]) ** 2)),
                                          val=np.sqrt(np.mean((mdl.predict(B) - y[va_]) ** 2))))
                    except Exception as e:
                        log("dim-study fit failed", p, nm, e)
    dd = pd.DataFrame(drows); dd.to_csv(out / "task6_dimension_study.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for nm, g_ in dd.groupby("model"):
        s = g_.groupby("p").agg(vm=("val", "mean"), vs=("val", "std"), tm=("train", "mean"))
        ax[0].errorbar(s.index, s.vm, s.vs, fmt="o-", capsize=3, label=nm); ax[1].plot(s.index, s.tm, "o--", label=nm)
    for a_ in ax:
        a_.axvline(n_tr, c="k", ls=":", label="p = n_train"); a_.set_xscale("log"); a_.set_xlabel("#features p"); a_.legend()
    ax[0].set_ylabel("validation RMSE"); ax[0].set_title("Performance vs feature dimension (error bar = fold std)"); ax[1].set_title("Train RMSE (interpolation when p>n)")
    savefig(fig, out, "fig_performance_vs_dimension.png")

    # ------------------------------------------------ 10 feature selection / reduction study (P1 features)
    X1 = sel_blocks(F, best_v, ("hog", "lbp", "col"))
    kk = min(300, X1.shape[1] // 2)
    meths = {"none (all)": lambda: ("none", None),
             "top-20% variance": lambda: ("sel", TopVar(.2)),
             f"SelectKBest f_reg k={kk}": lambda: ("sel", SelectKBest(f_regression, k=kk)),
             "L1 / Lasso (<=300)": lambda: ("sel", SelectFromModel(Lasso(alpha=.02, max_iter=300, tol=1e-3), max_features=300, threshold=-np.inf)),
             "PCA 100 (sklearn)": lambda: ("pca", PCA(100, random_state=seed))}
    srows, supp = [], {}
    for nm, mk in meths.items():
        for r_, rep in enumerate(folds[:1]):
            for k_, (tr_, va_) in enumerate(rep):
                kind, step = mk()
                steps = [StandardScaler()] + ([] if step is None else [step]) + [RidgeCV(alphas=np.logspace(0, 5, 9))]
                m = make_pipeline(*steps)
                t = time.perf_counter(); m.fit(X1[tr_], y[tr_]); ft = time.perf_counter() - t
                p = m.predict(X1[va_])
                nf = X1.shape[1] if kind == "none" else (100 if kind == "pca" else int(m.steps[1][1].get_support().sum()))
                if kind == "sel": supp.setdefault(nm, []).append(set(np.where(m.steps[1][1].get_support())[0]))
                srows.append(dict(method=nm, fold=k_, RMSE=metrics(y[va_], p)["RMSE"], MAE=metrics(y[va_], p)["MAE"],
                                  n_features=nf, fit_s=ft, model_MB=model_bytes(m) / 1e6))
    sd = pd.DataFrame(srows)
    stab = {k: np.mean([len(a_ & b_) / max(len(a_ | b_), 1) for i, a_ in enumerate(v_) for b_ in v_[i + 1:]]) for k, v_ in supp.items()}
    ss = sd.groupby("method").agg(RMSE=("RMSE", "mean"), RMSE_std=("RMSE", "std"), MAE=("MAE", "mean"), n_features=("n_features", "mean"),
                                  fit_s=("fit_s", "mean"), model_MB=("model_MB", "mean"))
    ss["stability_jaccard"] = pd.Series(stab)
    ss.round(4).to_csv(out / "task7_feature_selection.csv"); log("\n" + ss.round(4).to_string())
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
    for a_, (c, t) in zip(ax, [("RMSE", "CV RMSE"), ("fit_s", "fit time (s)"), ("model_MB", "model size (MB)"), ("stability_jaccard", "selection stability (Jaccard)")]):
        a_.bar(range(len(ss)), ss[c].fillna(0), yerr=ss["RMSE_std"] if c == "RMSE" else None, color=plt.cm.Set2(np.arange(len(ss))))
        a_.set_xticks(range(len(ss))); a_.set_xticklabels(ss.index, rotation=35, ha="right", fontsize=7); a_.set_title(t)
    savefig(fig, out, "fig_feature_selection.png")

    # ------------------------------------------------ 11 FINAL: refit on dev, test ONCE
    log("final refit on train+val, single test evaluation")
    test_rows, tpred, final_models = [], {}, {}
    boots = np.random.RandomState(seed).randint(0, len(test), (300, len(test)))
    for v in VARIANTS:
        fac = {"B0_train_mean": ((), lambda: DummyRegressor()),
               "B1_raw32_ridge": ("raw", lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 4, 9)))),
               **factories(seed, p2_choice[v])}
        for nm, (bl, fn) in fac.items():
            X = raw32 if bl == "raw" else (np.zeros((N, 1)) if nm.startswith("B0") else sel_blocks(F, v, bl))
            m = fn(); fit_any(m, X[dev], y[dev], gid[dev]); p = m.predict(X[test]); key = f"{nm}|{v}"
            tpred[key] = p; final_models[key] = m
            mt = metrics(y[test], p)
            bm = np.array([[np.abs(p[b] - y[test][b]).mean(), np.sqrt(np.mean((p[b] - y[test][b]) ** 2))] for b in boots])
            test_rows.append(dict(model=key, **mt, MAE_ci_lo=np.percentile(bm[:, 0], 2.5), MAE_ci_hi=np.percentile(bm[:, 0], 97.5),
                                  RMSE_ci_lo=np.percentile(bm[:, 1], 2.5), RMSE_ci_hi=np.percentile(bm[:, 1], 97.5)))
    pt = np.stack([tpred[f"{p}|{best_v}"] for p in factories(seed)], 1)
    ens, dis = pt.mean(1), pt.std(1)
    tpred[f"ENSEMBLE|{best_v}"] = ens
    test_rows.append(dict(model=f"ENSEMBLE|{best_v}", **metrics(y[test], ens)))
    td = pd.DataFrame(test_rows); td.round(4).to_csv(out / "test_results.csv", index=False)
    log("\nTEST\n" + td.round(4).to_string(index=False))

    # confusion matrices on rating tertiles (derived view of a regression result)
    qs = np.quantile(y[dev], [1 / 3, 2 / 3]); binf = lambda z: np.digitize(z, qs)
    lbl = ["low", "mid", "high"]
    ck = [f"{p}|{best_v}" for p in ["B1_raw32_ridge"] + list(factories(seed))] + [f"ENSEMBLE|{best_v}"]
    fig, ax = plt.subplots(1, len(ck), figsize=(4.4 * len(ck), 4.3))
    for a_, k in zip(ax, ck):
        cm = np.zeros((3, 3), int)
        for t_, p_ in zip(binf(y[test]), binf(tpred[k])): cm[t_, p_] += 1
        a_.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1); a_.grid(False)
        for i in range(3):
            for j in range(3): a_.text(j, i, cm[i, j], ha="center", va="center", color="k")
        acc = np.trace(cm) / cm.sum(); within = (np.abs(binf(y[test]) - binf(tpred[k])) <= 1).mean()
        a_.set_xticks(range(3)); a_.set_xticklabels(lbl); a_.set_yticks(range(3)); a_.set_yticklabels(lbl)
        a_.set_xlabel("predicted tertile"); a_.set_ylabel("true tertile"); a_.set_title(f"{k.split('|')[0][:22]}\nacc {acc:.2f}, within-1 {within:.2f}", fontsize=9)
    fig.suptitle("Confusion matrices on rating tertiles (derived from regression; row-normalised colour)")
    savefig(fig, out, "fig_confusion_matrices.png")

    # residual plots
    rk = [f"{p}|{best_v}" for p in factories(seed)] + [f"ENSEMBLE|{best_v}"]
    fig, ax = plt.subplots(3, len(rk), figsize=(4.8 * len(rk), 12))
    for j, k in enumerate(rk):
        p = tpred[k]; res_ = p - y[test]
        ax[0, j].scatter(y[test], p, s=6, alpha=.5); lim = [min(y[test].min(), p.min()), max(y[test].max(), p.max())]; ax[0, j].plot(lim, lim, "r--")
        bins = np.quantile(y[test], np.linspace(0, 1, 8)); bi = np.clip(np.digitize(y[test], bins[1:-1]), 0, 6)
        ax[0, j].plot([y[test][bi == b].mean() for b in range(7)], [p[bi == b].mean() for b in range(7)], "k-o", ms=4)
        ax[0, j].set_title(k.split("|")[0][:26] + "\npred vs true (black = binned mean)"); ax[0, j].set_xlabel("true"); ax[0, j].set_ylabel("pred")
        ax[1, j].hist(res_, 30); ax[1, j].axvline(0, c="r"); ax[1, j].set_title(f"residuals  bias={res_.mean():+.3f}")
        ax[2, j].scatter(y[test], res_, s=6, alpha=.5); ax[2, j].axhline(0, c="r"); ax[2, j].set_title("residual vs true rating"); ax[2, j].set_xlabel("true")
    savefig(fig, out, "fig_residuals.png")

    # ------------------------------------------------ 12 error gallery (20 test images, anonymous)
    res_e = ens - y[test]
    unc = std_img[test] if have_r else dis
    unc_name = "rater std" if have_r else "pipeline disagreement"
    o = np.argsort(-res_e); u = np.argsort(res_e); h_ = np.argsort(-np.nan_to_num(unc))
    pick, seen = [], set()
    for cat, order, n_ in (("overestimate", o, 7), ("underestimate", u, 7), ("high " + unc_name, h_, 6)):
        c = 0
        for i in order:
            if i in seen: continue
            seen.add(i); pick.append((cat, i)); c += 1
            if c == n_: break
    fig, ax = plt.subplots(4, 5, figsize=(15, 13))
    for a_, (cat, i) in zip(ax.ravel(), pick):
        a_.imshow(G128[test[i]], cmap="gray"); a_.axis("off")
        a_.set_title(f"#{i:03d} {cat}\ntrue {y[test][i]:.2f} pred {ens[i]:+.2f}\n{unc_name} {unc[i]:.2f}", fontsize=8)
    savefig(fig, out, "fig_error_gallery.png")
    big = np.abs(res_e) >= np.quantile(np.abs(res_e), .9)
    qd = pd.DataFrame({n_: [np.mean(Q[test][big, j]), np.mean(Q[test][~big, j]), stats.mannwhitneyu(Q[test][big, j], Q[test][~big, j])[1] if big.sum() > 2 else np.nan]
                       for j, n_ in enumerate(QUAL_NAMES)}, index=["top10%_error", "rest", "mannwhitney_p"]).T
    qd.loc["face_found_rate"] = [F[best_v]["geo"][test][big, gn.index("face_found") if best_v == "low_rgb" else geo_names(False).index("face_found")].mean(),
                                 F[best_v]["geo"][test][~big, geo_names(best_v == "low_rgb").index("face_found")].mean(), np.nan]
    qd.round(4).to_csv(out / "task9_error_vs_image_quality.csv"); log("\nERROR vs IMAGE QUALITY\n" + qd.round(4).to_string())

    # ------------------------------------------------ 13 group / rating-range analysis (metadata only)
    gt = pd.DataFrame({"group": lab.group.values[test], "y": y[test], "p": ens, "e": res_e, "ae": np.abs(res_e)})
    gt["range"] = pd.qcut(gt.y, 5, labels=["R1 lowest", "R2", "R3", "R4", "R5 highest"], duplicates="drop")
    gtab = gt.groupby("group").agg(n=("e", "size"), mean_true=("y", "mean"), mean_pred=("p", "mean"), bias=("e", "mean"), MAE=("ae", "mean"), MedAE=("ae", "median"))
    rtab = gt.groupby("range").agg(n=("e", "size"), bias=("e", "mean"), MAE=("ae", "mean"), MedAE=("ae", "median"))
    gtab.round(4).to_csv(out / "task10_group_errors.csv"); rtab.round(4).to_csv(out / "task10_range_errors.csv")
    grp = [g_.ae.values for _, g_ in gt.groupby("group") if len(g_) > 5]
    kw = stats.kruskal(*grp)[1] if len(grp) > 1 else np.nan
    log(f"\nGROUP\n{gtab.round(3).to_string()}\nRANGE\n{rtab.round(3).to_string()}\nKruskal-Wallis p(|err| differs by group) = {kw:.4f}")
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))
    gt.boxplot(column="e", by="group", ax=ax[0]); ax[0].axhline(0, c="r"); ax[0].set_title("signed error by group"); plt.sca(ax[0]); plt.xticks(rotation=20)
    br = np.random.RandomState(1)
    lo, hi_b = [], []
    for g_, d_ in gt.groupby("group"):
        bs = [d_.e.values[br.randint(0, len(d_), len(d_))].mean() for _ in range(300)]; lo.append(np.percentile(bs, 2.5)); hi_b.append(np.percentile(bs, 97.5))
    ax[1].bar(gtab.index, gtab.bias, yerr=[gtab.bias - lo, np.array(hi_b) - gtab.bias], capsize=4); ax[1].set_title("mean bias with 95% bootstrap CI"); ax[1].tick_params(axis="x", rotation=20)
    ax[2].bar(gtab.index, gtab.mean_true, width=.4, label="true (rater mean)"); ax[2].bar(np.arange(len(gtab)) + .4, gtab.mean_pred, width=.4, label="predicted"); ax[2].set_xticks(np.arange(len(gtab)) + .2)
    ax[2].set_xticklabels(gtab.index, rotation=20); ax[2].legend(); ax[2].set_title("group mean: raters vs model")
    ax[3].plot(rtab.index.astype(str), rtab.MAE, "o-", label="MAE"); ax[3].plot(rtab.index.astype(str), rtab.bias, "s-", label="bias"); ax[3].axhline(0, c="k"); ax[3].legend(); ax[3].set_title("error by rating range (regression to mean)")
    savefig(fig, out, "fig_group_and_range_errors.png")

    # ------------------------------------------------ 14 bundle for predict.py
    qlo = np.quantile(ens_oof[devm] - y[devm], [.1, .9])
    bundle = dict(variant=best_v, models={p: final_models[f"{p}|{best_v}"] for p in factories(seed)},
                  blocks={p: b for p, (b, _) in factories(seed).items()},
                  y_range=(float(y[dev].min()), float(y[dev].max())),
                  dis_thr=float(np.quantile(dis_oof[devm], .75)), resid_q10_q90=[float(qlo[0]), float(qlo[1])],
                  qual_lo=np.quantile(Q[dev], .05, axis=0), qual_hi=np.quantile(Q[dev], .95, axis=0),
                  cv_rmse=float(np.sqrt(np.mean((ens_oof[devm] - y[devm]) ** 2))))
    dump(bundle, out / "bundle.joblib")
    log("DONE. Outputs in", out)


if __name__ == "__main__":
    main()
