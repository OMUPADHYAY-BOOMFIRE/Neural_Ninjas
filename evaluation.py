#!/usr/bin/env python
"""evaluation.py - Analysis stages: PCA vs KMeans, ratio study,
modelling (3 pipelines x 2 variants), dimension study,
feature selection, error gallery, group analysis.

Part of the Neural Ninjas classical ML pipeline (Batch A3).
"""
import json
import time
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, dump
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression
from sklearn.linear_model import Lasso, Ridge, RidgeCV
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import CheckpointManager, log, log_separator
from features import (
    VARIANTS, BLOCKS, QUAL_NAMES, NumpyPCA,
    f_hog, geo_names, hamming_row,
)
from models import (
    ScratchPCAMLP, TopVar, factories, fit_any,
    model_bytes, metrics, run_cv, sel_blocks,
)

warnings.filterwarnings("ignore")


def savefig(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(Path(out_dir) / name, dpi=130)
    plt.close(fig)


# ================================================================
# Stage 5: PCA vs KMeans (Unsupervised Preprocessing)
# ================================================================
def run_pca_vs_kmeans(cfg, ckpt, stage1, stage3, stage4):
    """Compare PCA and KMeans as unsupervised preprocessing helpers."""
    log_separator("STAGE 5: PCA vs KMeans")

    if ckpt.is_done("stage5_pca_kmeans"):
        log("Stage 5 already complete, skipping...")
        return

    t0 = time.time()
    out = Path(cfg.output_dir)
    raw32 = stage1["raw32"]
    dev = stage3["dev"]
    folds = stage3["folds"]
    y = stage4["y"]
    ydev = y[dev]
    lab = stage1["lab"]

    Xs = StandardScaler().fit(raw32[dev]).transform(raw32[dev])
    pc2 = NumpyPCA(min(100, len(dev) - 2), seed=cfg.seed).fit(Xs)
    Zp = pc2.transform(Xs)

    sil, eta = {}, {}
    for k in range(2, 9):
        km = KMeans(k, n_init=5, random_state=cfg.seed).fit(Zp[:, :30])
        sil[k] = silhouette_score(
            Zp[:, :30], km.labels_,
            sample_size=min(2000, len(dev)), random_state=cfg.seed,
        )
        cm = pd.Series(ydev).groupby(km.labels_).transform("mean").values
        eta[k] = ((cm - ydev.mean()) ** 2).sum() / (((ydev - ydev.mean()) ** 2).sum() + 1e-9)

    km4 = KMeans(4, n_init=10, random_state=cfg.seed).fit(Zp[:, :30])
    cl = pd.DataFrame({
        "c": km4.labels_, "y": ydev,
        "g": lab.group.values[dev],
    })

    def cv_feat(Xf):
        s = []
        for tr_, va_ in folds[0]:
            pos = {v: i for i, v in enumerate(dev)}
            ti = [pos[i] for i in tr_]
            vi = [pos[i] for i in va_]
            m = make_pipeline(
                StandardScaler(),
                RidgeCV(alphas=np.logspace(-1, 4, 11)),
            ).fit(Xf[ti], ydev[ti])
            s.append(np.sqrt(np.mean((m.predict(Xf[vi]) - ydev[vi]) ** 2)))
        return np.mean(s), np.std(s)

    comp = {
        f"PCA{k}": cv_feat(Zp[:, :min(k, Zp.shape[1])])
        for k in (10, 30, 100)
    }
    for k in (4, 16, 32):
        comp[f"KMeans{k}-dist"] = cv_feat(
            KMeans(k, n_init=3, random_state=cfg.seed)
            .fit(Zp[:, :30])
            .transform(Zp[:, :30])
        )

    pd.DataFrame(comp, index=["RMSE_mean", "RMSE_std"]).T.to_csv(
        out / "task_pca_vs_kmeans.csv"
    )

    # Plot
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    ax[0, 0].plot(np.cumsum(pc2.evr_), "o-")
    ax[0, 0].set_title("PCA cumulative explained variance (raw 32x32)")
    ax[0, 1].plot(list(sil), list(sil.values()), "o-", label="silhouette")
    ax[0, 1].plot(list(eta), list(eta.values()), "s-", label="eta^2 of rating")
    ax[0, 1].legend()
    ax[0, 1].set_title("KMeans: k vs cluster quality / rating separation")
    cl.groupby("c").y.agg(["mean", "std", "count"]).reset_index().plot.bar(
        x="c", y="mean", yerr="std", ax=ax[0, 2], legend=False
    )
    ax[0, 2].set_title("k=4 clusters: mean rating (n on bars)")
    for i, n_ in enumerate(cl.groupby("c").y.count().values):
        ax[0, 2].text(i, 0.1, str(n_), ha="center", color="w")
    ct = pd.crosstab(cl.c, cl.g, normalize="index")
    im = ax[1, 0].imshow(ct.values, cmap="viridis")
    ax[1, 0].set_xticks(range(ct.shape[1]))
    ax[1, 0].set_xticklabels(ct.columns, rotation=30, fontsize=7)
    ax[1, 0].set_title(
        "cluster composition by metadata group\n"
        "(do clusters just rediscover the group?)"
    )
    ax[1, 0].grid(False)
    plt.colorbar(im, ax=ax[1, 0])
    ax[1, 1].scatter(Zp[:, 0], Zp[:, 1], c=km4.labels_, s=5, cmap="tab10")
    ax[1, 1].set_title("PCA-2D coloured by KMeans(4)")
    cdf = pd.DataFrame(comp, index=["m", "s"]).T
    ax[1, 2].bar(cdf.index, cdf.m, yerr=cdf.s)
    ax[1, 2].tick_params(axis="x", rotation=40)
    ax[1, 2].axhline(np.std(ydev), c="r", ls="--", label="predict-mean RMSE")
    ax[1, 2].legend()
    ax[1, 2].set_title("Ridge CV RMSE: PCA vs KMeans-distance features")
    savefig(fig, out, "fig_pca_vs_kmeans.png")

    ckpt.mark_done("stage5_pca_kmeans", time.time() - t0)
    log(f"Stage 5 done in {time.time() - t0:.1f}s")


# ================================================================
# Stage 6: Ratio Study
# ================================================================
def run_ratio_study(cfg, ckpt, stage1, stage3, stage4):
    log_separator("STAGE 6: Ratio Study")

    if ckpt.is_done("stage6_ratio_study"):
        log("Stage 6 already complete, skipping...")
        return

    t0 = time.time()
    out = Path(cfg.output_dir)
    F = stage1["F"]
    dev = stage3["dev"]
    y = stage4["y"]
    ydev = y[dev]

    Ftr = F["low_rgb"]["geo"][dev]
    gn = geo_names(True)
    hi = (
        cfg.hi_threshold if cfg.hi_threshold is not None
        else float(np.quantile(ydev, 0.75))
    )
    hm = ydev >= hi
    log(f"high-rated cut = {hi:.3f}; share >=2.5 in dev = {(ydev >= 2.5).mean():.2%}")

    rows = []
    for j, n_ in enumerate(gn):
        c = Ftr[:, j]
        m = ~np.isnan(c)
        if m.sum() < 30 or np.std(c[m]) == 0:
            continue
        rho = stats.spearmanr(c[m], ydev[m])[0]
        mh = np.median(c[m & hm])
        ml = np.median(c[m & ~hm])
        rows.append(dict(
            ratio=n_, missing=1 - m.mean(), spearman=rho,
            median_high=mh, median_low=ml,
            mean_high=np.mean(c[m & hm]),
            mean_low=np.mean(c[m & ~hm]),
            eff=(mh - ml) / (np.std(c[m]) + 1e-9),
        ))

    rt_ = pd.DataFrame(rows).sort_values(
        "spearman", key=np.abs, ascending=False
    )
    rt_.to_csv(out / "task_ratio_study.csv", index=False)

    top = rt_.head(15)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].barh(top.ratio[::-1], top.spearman[::-1])
    ax[0].set_title("Spearman(ratio, rating) - dev only")
    ax[1].barh(top.ratio[::-1], top.eff[::-1], color="C1")
    ax[1].set_title(f"(median high - median rest)/sd, cut={hi:.2f}")
    savefig(fig, out, "fig_ratio_study.png")

    ckpt.mark_done("stage6_ratio_study", time.time() - t0)
    log(f"Stage 6 done in {time.time() - t0:.1f}s")


# ================================================================
# Stage 7: Baselines + 3 Pipelines x 2 Variants (repeated CV)
# ================================================================
def run_modelling(cfg, ckpt, stage1, stage3, stage4):
    log_separator("STAGE 7: Baselines + 3 Pipelines (Repeated CV)")

    if ckpt.is_done("stage7_modelling"):
        log("Stage 7 already complete, loading from checkpoint...")
        return ckpt.load("stage7_modelling")

    t0 = time.time()
    out = Path(cfg.output_dir)
    F = stage1["F"]
    raw32 = stage1["raw32"]
    tm = stage1["tm"]
    gid = ckpt.load("stage2_duplicates")["gid"]
    dev = stage3["dev"]
    folds = stage3["folds"]
    y = stage4["y"]
    N = len(y)

    K = cfg.cv_folds
    R = cfg.cv_repeats

    results, oofs, sizes, dims = [], {}, {}, {}
    p2_choice = {}

    from sklearn.dummy import DummyRegressor

    for v in VARIANTS:
        # Decide mean-vs-median imputation by CV on dev
        sc = {}
        for st in ("mean", "median"):
            fz = factories(cfg.seed, st)["P2_cv2geo_edge_hgb"]
            d_, _, _ = run_cv(
                f"P2_{st}", fz[1],
                sel_blocks(F, v, fz[0]), y, gid, folds[:1],
            )
            sc[st] = d_.RMSE.mean()
        p2_choice[v] = min(sc, key=sc.get)
        log(f"[{v}] ratio-imputation: {sc} -> {p2_choice[v]}")

        fac = factories(cfg.seed, p2_choice[v])
        fac = {
            "B0_train_mean": (
                (), lambda: DummyRegressor()
            ),
            "B1_raw32_ridge": (
                "raw",
                lambda: make_pipeline(
                    StandardScaler(),
                    RidgeCV(alphas=np.logspace(0, 4, 9)),
                ),
            ),
            **fac,
        }

        for nm, (bl, fn) in fac.items():
            if bl == "raw":
                X = raw32
            elif nm.startswith("B0"):
                X = np.zeros((N, 1))
            else:
                X = sel_blocks(F, v, bl)

            key = f"{nm}|{v}"
            d_, oof, sz = run_cv(key, fn, X, y, gid, folds)
            results.append(d_)
            oofs[key] = oof
            sizes[key] = sz
            dims[key] = X.shape[1]
            log(
                f"{key:42s} RMSE {d_.RMSE.mean():.4f}"
                f"+/-{d_.RMSE.std():.4f}  "
                f"MAE {d_.MAE.mean():.4f}  "
                f"r {d_.Pearson.mean():.3f}"
            )

        # Checkpoint after each variant
        ckpt.save(f"stage7_partial_{v}", {
            "results_so_far": results,
            "oofs_so_far": oofs,
        })
        log(f"Variant {v} checkpointed.")

    cvdf = pd.concat(results)
    cvdf.to_csv(out / "cv_all_folds.csv", index=False)

    mets = ["MAE", "RMSE", "MedAE", "Pearson", "Spearman", "R2"]
    summ = cvdf.groupby("model")[mets + ["fit_s", "pred_ms"]].agg(
        ["mean", "std"]
    )
    summ.columns = [f"{c}_{s}" for c, s in summ.columns]
    summ["n_features"] = pd.Series(dims)
    summ["model_MB"] = pd.Series(sizes) / 1e6
    summ["n_train_ref"] = len(dev) * (K - 1) // K

    ext = {
        f"{k}|{v}": sum(tm[v][b] for b in bl) * 1e3
        for v in VARIANTS
        for k, (bl, _) in factories(cfg.seed).items()
    }
    summ["extract_ms_per_img"] = pd.Series(ext)
    summ.round(4).to_csv(out / "task8_cv_summary.csv")

    log("\n" + summ[[
        "MAE_mean", "RMSE_mean", "RMSE_std", "MedAE_mean",
        "Pearson_mean", "n_features", "fit_s_mean", "model_MB",
    ]].round(3).to_string())

    # Fig 08: Metric comparison bars
    names = list(summ.index)
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    for a_, m_ in zip(ax.ravel(), mets):
        a_.bar(range(len(names)), summ[f"{m_}_mean"], yerr=summ[f"{m_}_std"],
               color=plt.cm.tab20(np.arange(len(names))), capsize=3)
        a_.set_xticks(range(len(names)))
        a_.set_xticklabels(names, rotation=75, ha="right", fontsize=7)
        a_.set_title(f"{m_} (mean+/-std over {R}x{K} folds)")
    savefig(fig, out, "fig08_metric_comparison.png")

    # Fig scoreboard + cost
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    H = summ[[f"{m}_mean" for m in mets]].copy()
    Hn = (H - H.min()) / (H.max() - H.min() + 1e-12)
    for m in ("Pearson", "Spearman", "R2"):
        Hn[f"{m}_mean"] = 1 - Hn[f"{m}_mean"]
    ax[0].imshow(Hn.values, cmap="RdYlGn_r", aspect="auto")
    ax[0].grid(False)
    ax[0].set_yticks(range(len(names)))
    ax[0].set_yticklabels(names, fontsize=7)
    ax[0].set_xticks(range(len(mets)))
    ax[0].set_xticklabels(mets)
    for i in range(len(names)):
        for j, m in enumerate(mets):
            ax[0].text(j, i, f"{H.iloc[i, j]:.3f}", ha="center", va="center", fontsize=7)
    ax[0].set_title("Scoreboard (green = best per metric)")

    cost = summ[["fit_s_mean", "pred_ms_mean", "model_MB", "n_features"]].copy()
    cost["extract_ms_per_img"] = summ.extract_ms_per_img.fillna(0) if "extract_ms_per_img" in summ else 0
    ax[1].imshow(np.log10(cost.values.astype(float) + 1e-3), cmap="Blues", aspect="auto")
    ax[1].grid(False)
    ax[1].set_yticks(range(len(names)))
    ax[1].set_yticklabels(names, fontsize=7)
    ax[1].set_xticks(range(cost.shape[1]))
    ax[1].set_xticklabels(["fit s", "predict ms/img", "model MB", "#features", "extract ms/img"], rotation=15)
    for i in range(len(names)):
        for j in range(cost.shape[1]):
            ax[1].text(j, i, f"{cost.iloc[i, j]:.3g}", ha="center", va="center", fontsize=7)
    ax[1].set_title("Cost (log colour scale): time / size / dimension")
    savefig(fig, out, "fig_scoreboard_and_cost.png")

    result = {
        "cvdf": cvdf, "oofs": oofs, "sizes": sizes, "dims": dims,
        "p2_choice": p2_choice, "summ": summ,
    }

    ckpt.save("stage7_modelling", result)
    ckpt.mark_done("stage7_modelling", time.time() - t0)
    log(f"Stage 7 done in {time.time() - t0:.1f}s")
    return result


# ================================================================
# Stage 8: Dimension Study
# ================================================================
def run_dimension_study(cfg, ckpt, stage1, stage3, stage4):
    log_separator("STAGE 8: Dimension Study (p < n, p ~ n, p > n)")

    if ckpt.is_done("stage8_dimension"):
        log("Stage 8 already complete, skipping...")
        return

    t0 = time.time()
    out = Path(cfg.output_dir)
    G128 = stage1["G128"]
    dev = stage3["dev"]
    y = stage4["y"]
    gid = ckpt.load("stage2_duplicates")["gid"]

    log("Building fine HOG (6px cells)...")
    fine = np.stack(
        Parallel(n_jobs=cfg.n_jobs, batch_size=cfg.batch_size)(
            delayed(f_hog)(g, 6) for g in G128
        )
    ).astype(np.float32)

    colperm = np.random.RandomState(cfg.seed).permutation(fine.shape[1])
    from preprocessing import make_folds
    dfolds = make_folds(dev, gid, 3, 1 if cfg.fast else 2, cfg.seed)
    n_tr = len(dev) * 2 // 3
    ps = sorted({
        min(int(f * n_tr), fine.shape[1])
        for f in (0.05, 0.25, 0.5, 0.9, 1.0, 1.1, 2.0, 3.0)
    })

    drows = []
    for p in ps:
        for rep in dfolds:
            for tr_, va_ in rep:
                Xa = fine[:, colperm[:p]]
                sc = StandardScaler().fit(Xa[tr_])
                A = sc.transform(Xa[tr_])
                B = sc.transform(Xa[va_])
                for nm, mdl in [
                    ("weak ridge (alpha=0.01)", Ridge(alpha=0.01)),
                    ("tuned ridge (CV alpha)", RidgeCV(alphas=np.logspace(0, 5, 8))),
                ]:
                    try:
                        mdl.fit(A, y[tr_])
                        drows.append(dict(
                            p=p, model=nm,
                            train=np.sqrt(np.mean((mdl.predict(A) - y[tr_]) ** 2)),
                            val=np.sqrt(np.mean((mdl.predict(B) - y[va_]) ** 2)),
                        ))
                    except Exception as e:
                        log("dim-study fit failed", p, nm, e)

    dd = pd.DataFrame(drows)
    dd.to_csv(out / "task6_dimension_study.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for nm, g_ in dd.groupby("model"):
        s = g_.groupby("p").agg(
            vm=("val", "mean"), vs=("val", "std"), tm=("train", "mean")
        )
        ax[0].errorbar(s.index, s.vm, s.vs, fmt="o-", capsize=3, label=nm)
        ax[1].plot(s.index, s.tm, "o--", label=nm)
    for a_ in ax:
        a_.axvline(n_tr, c="k", ls=":", label="p = n_train")
        a_.set_xscale("log")
        a_.set_xlabel("#features p")
        a_.legend()
    ax[0].set_ylabel("validation RMSE")
    ax[0].set_title("Performance vs feature dimension")
    ax[1].set_title("Train RMSE (interpolation when p>n)")
    savefig(fig, out, "fig_performance_vs_dimension.png")

    ckpt.mark_done("stage8_dimension", time.time() - t0)
    log(f"Stage 8 done in {time.time() - t0:.1f}s")


# ================================================================
# Stage 9: Feature Selection Study
# ================================================================
def run_feature_selection(cfg, ckpt, stage1, stage3, stage4, stage7):
    log_separator("STAGE 9: Feature Selection / Reduction")

    if ckpt.is_done("stage9_selection"):
        log("Stage 9 already complete, skipping...")
        return

    t0 = time.time()
    out = Path(cfg.output_dir)
    F = stage1["F"]
    folds = stage3["folds"]
    y = stage4["y"]

    best_v = min(
        VARIANTS,
        key=lambda v: np.mean([
            stage7["cvdf"][stage7["cvdf"].model == f"{p}|{v}"].RMSE.mean()
            for p in factories(cfg.seed)
        ]),
    )

    X1 = sel_blocks(F, best_v, ("hog", "lbp", "col"))
    kk = min(300, X1.shape[1] // 2)

    meths = {
        "none (all)": lambda: ("none", None),
        "top-20% variance": lambda: ("sel", TopVar(0.2)),
        f"SelectKBest f_reg k={kk}": lambda: (
            "sel", SelectKBest(f_regression, k=kk)
        ),
        "L1 / Lasso (<=300)": lambda: (
            "sel",
            SelectFromModel(
                Lasso(alpha=0.02, max_iter=300, tol=1e-3),
                max_features=300, threshold=-np.inf,
            ),
        ),
        "PCA 100 (sklearn)": lambda: (
            "pca", PCA(100, random_state=cfg.seed)
        ),
    }

    srows, supp = [], {}
    for nm, mk in meths.items():
        for r_, rep in enumerate(folds[:1]):
            for k_, (tr_, va_) in enumerate(rep):
                kind, step = mk()
                steps = (
                    [StandardScaler()]
                    + ([] if step is None else [step])
                    + [RidgeCV(alphas=np.logspace(0, 5, 9))]
                )
                m = make_pipeline(*steps)
                t = time.perf_counter()
                m.fit(X1[tr_], y[tr_])
                ft = time.perf_counter() - t
                p = m.predict(X1[va_])
                nf = (
                    X1.shape[1] if kind == "none"
                    else (100 if kind == "pca"
                          else int(m.steps[1][1].get_support().sum()))
                )
                if kind == "sel":
                    supp.setdefault(nm, []).append(
                        set(np.where(m.steps[1][1].get_support())[0])
                    )
                srows.append(dict(
                    method=nm, fold=k_,
                    RMSE=metrics(y[va_], p)["RMSE"],
                    MAE=metrics(y[va_], p)["MAE"],
                    n_features=nf, fit_s=ft,
                    model_MB=model_bytes(m) / 1e6,
                ))

    sd = pd.DataFrame(srows)
    stab = {
        k: np.mean([
            len(a_ & b_) / max(len(a_ | b_), 1)
            for i, a_ in enumerate(v_)
            for b_ in v_[i + 1:]
        ])
        for k, v_ in supp.items()
    }
    ss = sd.groupby("method").agg(
        RMSE=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        MAE=("MAE", "mean"), n_features=("n_features", "mean"),
        fit_s=("fit_s", "mean"), model_MB=("model_MB", "mean"),
    )
    ss["stability_jaccard"] = pd.Series(stab)
    ss.round(4).to_csv(out / "task7_feature_selection.csv")
    log("\n" + ss.round(4).to_string())

    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
    for a_, (c, t) in zip(ax, [
        ("RMSE", "CV RMSE"),
        ("fit_s", "fit time (s)"),
        ("model_MB", "model size (MB)"),
        ("stability_jaccard", "selection stability (Jaccard)"),
    ]):
        a_.bar(
            range(len(ss)), ss[c].fillna(0),
            yerr=ss["RMSE_std"] if c == "RMSE" else None,
            color=plt.cm.Set2(np.arange(len(ss))),
        )
        a_.set_xticks(range(len(ss)))
        a_.set_xticklabels(ss.index, rotation=35, ha="right", fontsize=7)
        a_.set_title(t)
    savefig(fig, out, "fig_feature_selection.png")

    ckpt.mark_done("stage9_selection", time.time() - t0)
    log(f"Stage 9 done in {time.time() - t0:.1f}s")


# ================================================================
# Stage 10: Final Test + Plots + Bundle
# ================================================================
def run_final_test(cfg, ckpt, stage1, stage3, stage4, stage7):
    """Refit on dev, test ONCE, generate all final plots + bundle."""
    log_separator("STAGE 10: Final Test Evaluation + Bundle")

    if ckpt.is_done("stage10_final"):
        log("Stage 10 already complete, skipping...")
        return

    t0 = time.time()
    out = Path(cfg.output_dir)
    F = stage1["F"]
    raw32 = stage1["raw32"]
    G128 = stage1["G128"]
    Q = stage1["Q"]
    lab = stage1["lab"]
    dev = stage3["dev"]
    test = stage3["test"]
    train = stage3["train"]
    val = stage3["val"]
    folds = stage3["folds"]
    y = stage4["y"]
    have_r = stage4["have_r"]
    std_img = stage4["std_img"]
    gid = ckpt.load("stage2_duplicates")["gid"]
    p2_choice = stage7["p2_choice"]
    oofs = stage7["oofs"]
    cvdf = stage7["cvdf"]
    N = len(y)

    best_v = min(
        VARIANTS,
        key=lambda v: np.mean([
            cvdf[cvdf.model == f"{p}|{v}"].RMSE.mean()
            for p in factories(cfg.seed)
        ]),
    )
    log("Best compression variant by CV =", best_v)

    # Ensemble OOF
    pk = [f"{p}|{best_v}" for p in factories(cfg.seed)]
    oof_mat = np.stack([oofs[k] for k in pk], 1)
    ens_oof = oof_mat.mean(1)
    dis_oof = oof_mat.std(1)
    devm = ~np.isnan(ens_oof)

    log("Ensemble OOF", {
        k: round(v_, 4)
        for k, v_ in metrics(y[devm], ens_oof[devm]).items()
    })

    # -- Learning curves
    fac = factories(cfg.seed, p2_choice[best_v])
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
    m3 = ScratchPCAMLP(k=100, seed=cfg.seed).fit(
        sel_blocks(F, best_v, BLOCKS)[train], y[train], gid[train]
    )
    ax[0].plot(m3.history_["train"], label="train MSE")
    ax[0].plot(m3.history_["val"], label="inner-val MSE")
    ax[0].axvline(m3.best_epoch_, c="k", ls=":", label=f"early stop @ {m3.best_epoch_}")
    ax[0].legend()
    ax[0].set_xlabel("epoch")
    ax[0].set_title("P3 MLP: epoch learning curve")

    rng = np.random.RandomState(cfg.seed)
    fr = [0.2, 0.4, 0.6, 0.8, 1.0]
    for nm, (bl, fn) in fac.items():
        Xb = sel_blocks(F, best_v, bl)
        trm, vam = [], []
        for f in fr:
            sub = rng.permutation(train)[:max(30, int(f * len(train)))]
            m = fn()
            fit_any(m, Xb[sub], y[sub], gid[sub])
            trm.append(np.sqrt(np.mean((m.predict(Xb[sub]) - y[sub]) ** 2)))
            vam.append(np.sqrt(np.mean((m.predict(Xb[val]) - y[val]) ** 2)))
        ax[1].plot(np.array(fr) * len(train), vam, "o-", label=nm.split("_")[0])
        ax[2].plot(np.array(fr) * len(train), trm, "o--", label=nm.split("_")[0])
    ax[1].set_title("Validation RMSE vs #train")
    ax[2].set_title("Train RMSE (gap = overfitting)")
    for a_ in ax[1:]:
        a_.legend()
        a_.set_xlabel("train images")
    savefig(fig, out, "fig_learning_curves.png")

    # -- Final refit + test
    log("Final refit on dev, single test evaluation")
    test_rows, tpred, final_models = [], {}, {}
    boots = np.random.RandomState(cfg.seed).randint(0, len(test), (300, len(test)))

    from sklearn.dummy import DummyRegressor

    for v in VARIANTS:
        full_fac = {
            "B0_train_mean": ((), lambda: DummyRegressor()),
            "B1_raw32_ridge": (
                "raw",
                lambda: make_pipeline(
                    StandardScaler(),
                    RidgeCV(alphas=np.logspace(0, 4, 9)),
                ),
            ),
            **factories(cfg.seed, p2_choice[v]),
        }
        for nm, (bl, fn) in full_fac.items():
            if bl == "raw":
                X = raw32
            elif nm.startswith("B0"):
                X = np.zeros((N, 1))
            else:
                X = sel_blocks(F, v, bl)

            key = f"{nm}|{v}"
            m = fn()
            fit_any(m, X[dev], y[dev], gid[dev])
            p = m.predict(X[test])
            tpred[key] = p
            final_models[key] = m

            mt = metrics(y[test], p)
            bm = np.array([
                [
                    np.abs(p[b] - y[test][b]).mean(),
                    np.sqrt(np.mean((p[b] - y[test][b]) ** 2)),
                ]
                for b in boots
            ])
            test_rows.append(dict(
                model=key, **mt,
                MAE_ci_lo=np.percentile(bm[:, 0], 2.5),
                MAE_ci_hi=np.percentile(bm[:, 0], 97.5),
                RMSE_ci_lo=np.percentile(bm[:, 1], 2.5),
                RMSE_ci_hi=np.percentile(bm[:, 1], 97.5),
            ))

    # Ensemble
    pt = np.stack([tpred[f"{p}|{best_v}"] for p in factories(cfg.seed)], 1)
    ens, dis = pt.mean(1), pt.std(1)
    tpred[f"ENSEMBLE|{best_v}"] = ens
    test_rows.append(dict(
        model=f"ENSEMBLE|{best_v}",
        **metrics(y[test], ens),
    ))

    td = pd.DataFrame(test_rows)
    td.round(4).to_csv(out / "test_results.csv", index=False)
    log("\nTEST\n" + td.round(4).to_string(index=False))

    # -- Confusion matrices on rating tertiles
    qs = np.quantile(y[dev], [1 / 3, 2 / 3])
    binf = lambda z: np.digitize(z, qs)
    lbl = ["low", "mid", "high"]
    ck = [f"{p}|{best_v}" for p in ["B1_raw32_ridge"] + list(factories(cfg.seed))]
    ck.append(f"ENSEMBLE|{best_v}")

    fig, ax = plt.subplots(1, len(ck), figsize=(4.4 * len(ck), 4.3))
    for a_, k in zip(ax, ck):
        cm = np.zeros((3, 3), int)
        for t_, p_ in zip(binf(y[test]), binf(tpred[k])):
            cm[t_, p_] += 1
        a_.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
        a_.grid(False)
        for i in range(3):
            for j in range(3):
                a_.text(j, i, cm[i, j], ha="center", va="center", color="k")
        acc = np.trace(cm) / cm.sum()
        within = (np.abs(binf(y[test]) - binf(tpred[k])) <= 1).mean()
        a_.set_xticks(range(3))
        a_.set_xticklabels(lbl)
        a_.set_yticks(range(3))
        a_.set_yticklabels(lbl)
        a_.set_xlabel("predicted tertile")
        a_.set_ylabel("true tertile")
        a_.set_title(
            f"{k.split('|')[0][:22]}\nacc {acc:.2f}, within-1 {within:.2f}",
            fontsize=9,
        )
    fig.suptitle("Confusion matrices on rating tertiles (derived view)")
    savefig(fig, out, "fig_confusion_matrices.png")

    # -- Residual plots
    rk = [f"{p}|{best_v}" for p in factories(cfg.seed)]
    rk.append(f"ENSEMBLE|{best_v}")
    fig, ax = plt.subplots(3, len(rk), figsize=(4.8 * len(rk), 12))
    for j, k in enumerate(rk):
        p = tpred[k]
        res_ = p - y[test]
        ax[0, j].scatter(y[test], p, s=6, alpha=0.5)
        lim = [min(y[test].min(), p.min()), max(y[test].max(), p.max())]
        ax[0, j].plot(lim, lim, "r--")
        ax[0, j].set_title(k.split("|")[0][:26] + "\npred vs true")
        ax[0, j].set_xlabel("true")
        ax[0, j].set_ylabel("pred")
        ax[1, j].hist(res_, 30)
        ax[1, j].axvline(0, c="r")
        ax[1, j].set_title(f"residuals bias={res_.mean():+.3f}")
        ax[2, j].scatter(y[test], res_, s=6, alpha=0.5)
        ax[2, j].axhline(0, c="r")
        ax[2, j].set_title("residual vs true")
        ax[2, j].set_xlabel("true")
    savefig(fig, out, "fig_residuals.png")

    # -- Error gallery (20 test images)
    res_e = ens - y[test]
    unc = std_img[test] if have_r else dis
    unc_name = "rater std" if have_r else "pipeline disagreement"

    o = np.argsort(-res_e)
    u = np.argsort(res_e)
    h_ = np.argsort(-np.nan_to_num(unc))
    pick, seen = [], set()
    for cat, order, n_ in [
        ("overestimate", o, 7),
        ("underestimate", u, 7),
        ("high " + unc_name, h_, 6),
    ]:
        c = 0
        for i in order:
            if i in seen:
                continue
            seen.add(i)
            pick.append((cat, i))
            c += 1
            if c == n_:
                break

    fig, ax = plt.subplots(4, 5, figsize=(15, 13))
    for a_, (cat, i) in zip(ax.ravel(), pick):
        a_.imshow(G128[test[i]], cmap="gray")
        a_.axis("off")
        a_.set_title(
            f"#{i:03d} {cat}\ntrue {y[test][i]:.2f} "
            f"pred {ens[i]:+.2f}\n{unc_name} {unc[i]:.2f}",
            fontsize=8,
        )
    savefig(fig, out, "fig_error_gallery.png")

    # -- Error vs image quality (Mann-Whitney U test)
    big = np.abs(res_e) >= np.quantile(np.abs(res_e), 0.9)
    gn = geo_names(best_v == "low_rgb")
    face_idx = gn.index("face_found")
    qd = pd.DataFrame({
        n_: [
            np.mean(Q[test][big, j]),
            np.mean(Q[test][~big, j]),
            (
                stats.mannwhitneyu(Q[test][big, j], Q[test][~big, j])[1]
                if big.sum() > 2 else np.nan
            ),
        ]
        for j, n_ in enumerate(QUAL_NAMES)
    }, index=["top10%_error", "rest", "mannwhitney_p"]).T
    qd.loc["face_found_rate"] = [
        F[best_v]["geo"][test][big, face_idx].mean(),
        F[best_v]["geo"][test][~big, face_idx].mean(),
        np.nan,
    ]
    qd.round(4).to_csv(out / "task9_error_vs_image_quality.csv")
    log("\nERROR vs IMAGE QUALITY\n" + qd.round(4).to_string())

    # -- Group analysis & rating range analysis
    gt = pd.DataFrame({
        "group": lab.group.values[test],
        "y": y[test], "p": ens,
        "e": res_e, "ae": np.abs(res_e),
    })
    gt["range"] = pd.qcut(
        gt.y, 5,
        labels=["R1 lowest", "R2", "R3", "R4", "R5 highest"],
        duplicates="drop",
    )
    gtab = gt.groupby("group").agg(
        n=("e", "size"), mean_true=("y", "mean"),
        mean_pred=("p", "mean"), bias=("e", "mean"),
        MAE=("ae", "mean"), MedAE=("ae", "median"),
    )
    rtab = gt.groupby("range").agg(
        n=("e", "size"), bias=("e", "mean"),
        MAE=("ae", "mean"), MedAE=("ae", "median"),
    )
    gtab.round(4).to_csv(out / "task10_group_errors.csv")
    rtab.round(4).to_csv(out / "task10_range_errors.csv")

    grp = [g_.ae.values for _, g_ in gt.groupby("group") if len(g_) > 5]
    kw = stats.kruskal(*grp)[1] if len(grp) > 1 else np.nan
    log(
        f"\nGROUP\n{gtab.round(3).to_string()}\n"
        f"RANGE\n{rtab.round(3).to_string()}\n"
        f"Kruskal-Wallis p(|err| differs by group) = {kw:.4f}"
    )

    fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))
    gt.boxplot(column="e", by="group", ax=ax[0])
    ax[0].axhline(0, c="r")
    ax[0].set_title("signed error by group")
    plt.sca(ax[0])
    plt.xticks(rotation=20)

    br = np.random.RandomState(1)
    lo_ci, hi_ci = [], []
    for g_, d_ in gt.groupby("group"):
        bs = [
            d_.e.values[br.randint(0, len(d_), len(d_))].mean()
            for _ in range(300)
        ]
        lo_ci.append(np.percentile(bs, 2.5))
        hi_ci.append(np.percentile(bs, 97.5))

    ax[1].bar(
        gtab.index, gtab.bias,
        yerr=[gtab.bias - lo_ci, np.array(hi_ci) - gtab.bias],
        capsize=4,
    )
    ax[1].set_title("mean bias with 95% bootstrap CI")
    ax[1].tick_params(axis="x", rotation=20)

    ax[2].bar(gtab.index, gtab.mean_true, width=0.4, label="true")
    ax[2].bar(
        np.arange(len(gtab)) + 0.4, gtab.mean_pred,
        width=0.4, label="predicted",
    )
    ax[2].set_xticks(np.arange(len(gtab)) + 0.2)
    ax[2].set_xticklabels(gtab.index, rotation=20)
    ax[2].legend()
    ax[2].set_title("group mean: raters vs model")

    ax[3].plot(rtab.index.astype(str), rtab.MAE, "o-", label="MAE")
    ax[3].plot(rtab.index.astype(str), rtab.bias, "s-", label="bias")
    ax[3].axhline(0, c="k")
    ax[3].legend()
    ax[3].set_title("error by rating range")
    savefig(fig, out, "fig_group_and_range_errors.png")

    # -- Bundle for predict.py
    qlo = np.quantile(ens_oof[devm] - y[devm], [0.1, 0.9])
    bundle = dict(
        variant=best_v,
        models={
            p: final_models[f"{p}|{best_v}"]
            for p in factories(cfg.seed)
        },
        blocks={
            p: b for p, (b, _) in factories(cfg.seed).items()
        },
        y_range=(float(y[dev].min()), float(y[dev].max())),
        dis_thr=float(np.quantile(dis_oof[devm], 0.75)),
        resid_q10_q90=[float(qlo[0]), float(qlo[1])],
        qual_lo=np.quantile(Q[dev], 0.05, axis=0),
        qual_hi=np.quantile(Q[dev], 0.95, axis=0),
        cv_rmse=float(np.sqrt(np.mean((ens_oof[devm] - y[devm]) ** 2))),
    )
    dump(bundle, out / "bundle.joblib")

    ckpt.mark_done("stage10_final", time.time() - t0)
    log(f"Stage 10 done in {time.time() - t0:.1f}s")
    log("DONE. Bundle saved to", out / "bundle.joblib")
