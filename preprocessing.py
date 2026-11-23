#!/usr/bin/env python
"""preprocessing.py - Data loading, audit, duplicate detection, splits.

Part of the Neural Ninjas classical ML pipeline (Batch A3).
Run standalone:
    python preprocessing.py --data_dir Images --labels labels.txt --out out

Produces:
    checkpoints/stage1_audit.pkl     (features, hashes, quality)
    checkpoints/stage2_duplicates.pkl (group IDs from union-find disjoint sets)
    checkpoints/stage3_splits.pkl    (train/val/test indices, folds)
    checkpoints/stage4_target.pkl    (target values, rater stats)
    out/task1_audit.json
    out/split_manifest.csv
    out/fig01_target_distribution.png
    out/fig05_leakage_check.png
"""
import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit

from config import (
    PipelineConfig, CheckpointManager, log, log_separator, GPU_INFO
)
from features import (
    VARIANTS, BLOCKS, extract_one, dhash, hamming_row, NumpyPCA
)

warnings.filterwarnings("ignore")


# ================================================================
# Stage 1: Load + Audit
# ================================================================
def read_labels(path):
    """Parse the labels file: 'filename score' per line."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    rows.append((parts[0], float(parts[1])))
                except ValueError:
                    pass

    df = pd.DataFrame(rows, columns=["name", "score"])
    m = df.name.str.extract(r"^([AaCc])([MmFf])")
    df["ethnicity"] = m[0].str.upper().map({"A": "Asian", "C": "Caucasian"})
    df["gender"] = m[1].str.upper().map({"M": "Male", "F": "Female"})
    df["group"] = df.ethnicity.fillna("?") + " " + df.gender.fillna("?")
    return df


def load_ratings(path):
    """Load per-rater ratings from Excel or CSV => per-image statistics."""
    df = (pd.read_excel(path)
          if str(path).endswith(("xlsx", "xls"))
          else pd.read_csv(path))

    low = {c.lower(): c for c in df.columns}
    rc = [c for l, c in low.items() if "rating" in l or "score" in l]
    fc = [c for l, c in low.items()
          if "file" in l or "image" in l or "name" in l]

    if fc and rc and any("rater" in l for l in low):
        long = df[[fc[0], rc[0]]].rename(
            columns={fc[0]: "name", rc[0]: "r"})
        wide = long.groupby("name")["r"].apply(list)
    else:
        wide = df.set_index(df.columns[0]).apply(
            lambda r: [x for x in r if pd.notna(x)], axis=1)

    out = pd.DataFrame({"name": wide.index})
    A = wide.values
    out["n_raters"] = [len(a) for a in A]
    out["r_mean"] = [np.mean(a) for a in A]
    out["r_median"] = [np.median(a) for a in A]
    out["r_trim"] = [stats.trim_mean(a, 0.1) for a in A]
    out["r_std"] = [
        np.std(a, ddof=1) if len(a) > 1 else np.nan for a in A
    ]
    out["r_iqr"] = [stats.iqr(a) for a in A]

    # Split-half reliability (ceiling for any model)
    rng = np.random.RandomState(0)
    h1, h2 = [], []
    for a in A:
        a = np.array(a)
        rng.shuffle(a)
        h1.append(a[: len(a) // 2].mean())
        h2.append(a[len(a) // 2 :].mean())
    out["h1"], out["h2"] = h1, h2
    return out


def run_audit(cfg, ckpt):
    """Stage 1: load images, extract features, detect corrupt files."""
    log_separator("STAGE 1: Data Audit + Feature Extraction")

    if ckpt.is_done("stage1_audit"):
        log("Stage 1 already complete, loading from checkpoint...")
        return ckpt.load("stage1_audit")

    t0 = time.time()
    lab = read_labels(cfg.labels_file)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover image files
    files = {}
    for ext in ("jpg", "jpeg", "png", "bmp"):
        for p in Path(cfg.data_dir).rglob(f"*.{ext}"):
            files[p.name] = p

    audit = {
        "n_label_rows": int(len(lab)),
        "n_image_files": len(files),
        "labels_without_file": int((~lab.name.isin(files)).sum()),
        "files_without_label": int(len(set(files) - set(lab.name))),
        "missing_scores": int(lab.score.isna().sum()),
        "score_min": float(lab.score.min()) if len(lab) else 0.0,
        "score_max": float(lab.score.max()) if len(lab) else 0.0,
        "unparsed_group": int(
            lab.ethnicity.isna().sum() + lab.gender.isna().sum()
        ),
        "hardware": GPU_INFO,
    }

    lab = lab[lab.name.isin(files)].drop_duplicates("name").reset_index(
        drop=True
    )

    if cfg.limit:
        lab = lab.sample(
            min(cfg.limit, len(lab)), random_state=cfg.seed
        ).reset_index(drop=True)

    log(f"Extracting features for {len(lab)} images "
        f"(parallel, batch_size={cfg.batch_size})...")

    # Parallel feature extraction
    res = Parallel(
        n_jobs=cfg.n_jobs, batch_size=cfg.batch_size, verbose=1
    )(delayed(extract_one)(files[n]) for n in lab.name)

    ok = np.array([r is not None for r in res])
    audit["corrupt_files"] = int((~ok).sum())
    lab = lab[ok].reset_index(drop=True)
    res = [r for r in res if r is not None]
    N = len(lab)
    audit["n_valid"] = N

    shapes = pd.Series([str(r["shape"]) for r in res]).value_counts()
    audit["image_sizes_top"] = shapes.head(5).to_dict()
    y_given = lab.score.values.astype(float)

    # Build feature matrices
    F = {
        v: {b: np.stack([r[v][0][b] for r in res]) for b in BLOCKS}
        for v in VARIANTS
    }
    raw32 = np.stack([r["raw32"] for r in res])
    Q = np.stack([r["quality"] for r in res])
    G128 = np.stack([r["g128"] for r in res])
    hashes = np.array([r["dhash"] for r in res], dtype=np.uint64)

    # Timing table
    tm = {
        v: {
            b: float(np.mean([r[v][1][b] for r in res]))
            for b in BLOCKS
        }
        for v in VARIANTS
    }
    feat_tab = pd.DataFrame([
        dict(
            variant=v, block=b, dim=F[v][b].shape[1],
            extract_ms_per_img=tm[v][b] * 1e3,
            storage_MB=F[v][b].nbytes / 1e6,
            nan_frac=float(np.isnan(F[v][b]).mean()),
        )
        for v in VARIANTS for b in BLOCKS
    ])
    feat_tab.to_csv(out_dir / "task3_feature_table.csv", index=False)
    log("\n" + feat_tab.round(3).to_string(index=False))

    # Save audit
    with open(out_dir / "task1_audit.json", "w") as f:
        json.dump(audit, f, indent=2, default=str)
    log("AUDIT", json.dumps(audit, default=str))

    # Plot target distribution
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].hist(y_given, 40, color="C0")
    ax[0].axvline(np.mean(y_given), c="r", label="mean")
    ax[0].axvline(np.median(y_given), c="g", ls="--", label="median")
    ax[0].legend()
    ax[0].set_title(f"Target distribution (skew={stats.skew(y_given):.2f})")
    lab.assign(y=y_given).boxplot(column="y", by="group", ax=ax[1])
    ax[1].set_title("Target by metadata group")
    plt.sca(ax[1])
    plt.xticks(rotation=20)
    ax[2].hist(Q[:, 0], 40)
    ax[2].set_title("Image sharpness (Laplacian var)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig01_target_distribution.png", dpi=130)
    plt.close(fig)

    result = {
        "lab": lab, "F": F, "raw32": raw32, "Q": Q, "G128": G128,
        "hashes": hashes, "y_given": y_given, "tm": tm, "audit": audit,
    }

    ckpt.save("stage1_audit", result)
    ckpt.mark_done("stage1_audit", time.time() - t0)
    log(f"Stage 1 done in {time.time() - t0:.1f}s")
    return result


# ================================================================
# Stage 2: Duplicate Detection (compressed-hash disjoint sets)
# ================================================================
def run_duplicates(cfg, ckpt, stage1):
    """Detect near-duplicates using dHash + union-find on 4 prefix buckets.

    Key optimisation: images are compressed to 9x8 grayscale BEFORE hashing,
    so we compare 64-bit integers, NOT full pixel matrices.
    The union-find creates disjoint sets in O(N*alpha(N)) ~ O(N).
    Cross-bucket exact-match catches duplicates with wrong prefix labels.
    """
    log_separator("STAGE 2: Duplicate Detection (dHash + Disjoint Sets)")

    if ckpt.is_done("stage2_duplicates"):
        log("Stage 2 already complete, loading from checkpoint...")
        return ckpt.load("stage2_duplicates")

    t0 = time.time()
    lab = stage1["lab"]
    hashes = stage1["hashes"]
    N = len(lab)
    prefix = lab.group.values

    def bucket_pairs(idx):
        """Find duplicate pairs within one prefix bucket."""
        h = hashes[idx]
        pr = []
        for i in range(len(idx) - 1):
            d = hamming_row(h[i], h[i + 1:])
            pr += [
                (idx[i], idx[i + 1 + j], int(d[j]))
                for j in np.where(d <= cfg.dup_threshold)[0]
            ]
        return pr

    buckets = [
        np.where(prefix == p)[0] for p in np.unique(prefix)
    ]
    log(f"Scanning {len(buckets)} prefix buckets in parallel...")

    pairs = sum(
        Parallel(n_jobs=min(4, len(buckets)))(
            delayed(bucket_pairs)(b) for b in buckets
        ),
        [],
    )

    # Cross-bucket exact-match
    _, inv, cnt = np.unique(hashes, return_inverse=True, return_counts=True)
    for u in np.where(cnt > 1)[0]:
        m = np.where(inv == u)[0]
        pairs += [(m[0], j, 0) for j in m[1:]]

    # Union-Find with path compression (Disjoint Set Union)
    par = np.arange(N)

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for i, j, _ in pairs:
        par[find(i)] = find(j)

    gid = np.array([find(i) for i in range(N)])

    dup_info = {
        "duplicate_pairs": len(pairs),
        "images_in_dup_groups": int(
            (pd.Series(gid).map(pd.Series(gid).value_counts()) > 1).sum()
        ),
        "n_groups": int(len(np.unique(gid))),
    }

    if dup_info["images_in_dup_groups"] > 0.05 * N:
        log(
            f"WARNING: {dup_info['images_in_dup_groups']} images in "
            f"duplicate groups - --dup_thr {cfg.dup_threshold} may be "
            f"too loose; lower it"
        )

    log("DUPLICATES", json.dumps(dup_info))

    result = {"gid": gid, "dup_info": dup_info}
    ckpt.save("stage2_duplicates", result)
    ckpt.mark_done("stage2_duplicates", time.time() - t0)
    log(f"Stage 2 done in {time.time() - t0:.1f}s")
    return result


# ================================================================
# Stage 3: Splits + Leakage Check
# ================================================================
def make_folds(dev_idx, gid, K, R, seed):
    """R repeats of group-aware K-fold.

    All images of a duplicate-group stay in the same fold.
    """
    out = []
    for r in range(R):
        rng = np.random.RandomState(seed + r)
        ug = np.unique(gid[dev_idx])
        rng.shuffle(ug)
        parts = np.array_split(ug, K)
        rep = []
        for p in parts:
            m = np.isin(gid[dev_idx], p)
            rep.append((dev_idx[~m], dev_idx[m]))
        out.append(rep)
    return out


def run_splits(cfg, ckpt, stage1, stage2):
    """Group-disjoint train/val/test splits + leakage verification."""
    log_separator("STAGE 3: Splits + Leakage Check")

    if ckpt.is_done("stage3_splits"):
        log("Stage 3 already complete, loading from checkpoint...")
        return ckpt.load("stage3_splits")

    t0 = time.time()
    lab = stage1["lab"]
    hashes = stage1["hashes"]
    raw32 = stage1["raw32"]
    gid = stage2["gid"]
    N = len(lab)
    out_dir = Path(cfg.output_dir)

    idx = np.arange(N)
    dev, test = next(
        GroupShuffleSplit(1, test_size=cfg.test_fraction,
                         random_state=cfg.seed).split(idx, groups=gid)
    )
    tr, va = next(
        GroupShuffleSplit(1, test_size=cfg.val_fraction,
                         random_state=cfg.seed).split(dev, groups=gid[dev])
    )
    train, val = dev[tr], dev[va]

    # Build manifest
    man = pd.DataFrame({"name": lab.name, "dup_group": gid, "split": "train"})
    man.loc[val, "split"] = "val"
    man.loc[test, "split"] = "test"

    K = cfg.cv_folds
    R = cfg.cv_repeats
    folds = make_folds(dev, gid, K, R, cfg.seed)
    for r_, rep in enumerate(folds):
        man[f"cv_rep{r_}_fold"] = -1
        for k_, (_, v_) in enumerate(rep):
            man.loc[v_, f"cv_rep{r_}_fold"] = k_

    man.to_csv(out_dir / "split_manifest.csv", index=False)
    log(
        f"Split sizes train/val/test = "
        f"{len(train)}/{len(val)}/{len(test)}; "
        f"group overlap dev-test = "
        f"{len(set(gid[dev]) & set(gid[test]))}"
    )

    # Leakage check: nearest dev neighbour of every test image
    pc = NumpyPCA(50, seed=cfg.seed).fit(raw32[dev])
    Zd = pc.transform(raw32[dev])
    Zt = pc.transform(raw32[test])
    Zd /= np.linalg.norm(Zd, axis=1, keepdims=True) + 1e-9
    Zt /= np.linalg.norm(Zt, axis=1, keepdims=True) + 1e-9
    cos_nn = (Zt @ Zd.T).max(1)
    ham_nn = np.array([
        hamming_row(hashes[t], hashes[dev]).min() for t in test
    ])

    leak = {
        f"test_with_hamming<={cfg.dup_threshold}": int(
            (ham_nn <= cfg.dup_threshold).sum()
        ),
        "test_with_cos>0.98": int((cos_nn > 0.98).sum()),
        "n_test": len(test),
    }
    with open(out_dir / "task5_leakage_check.json", "w") as f:
        json.dump(leak, f, indent=2)
    log("LEAK CHECK", leak)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(ham_nn, 30)
    ax[0].axvline(cfg.dup_threshold, c="r")
    ax[0].set_title("test->train min Hamming (dHash 64 bit)")
    ax[1].hist(cos_nn, 30)
    ax[1].set_title("test->train max cosine (pixel-PCA)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig05_leakage_check.png", dpi=130)
    plt.close(fig)

    result = {
        "dev": dev, "test": test, "train": train, "val": val,
        "folds": folds, "leak": leak,
    }

    ckpt.save("stage3_splits", result)
    ckpt.mark_done("stage3_splits", time.time() - t0)
    log(f"Stage 3 done in {time.time() - t0:.1f}s")
    return result


# ================================================================
# Stage 4: Target definition
# ================================================================
def run_target(cfg, ckpt, stage1):
    """Define the regression target: mean vs median vs trimmed mean."""
    log_separator("STAGE 4: Target Definition")

    if ckpt.is_done("stage4_target"):
        log("Stage 4 already complete, loading from checkpoint...")
        return ckpt.load("stage4_target")

    t0 = time.time()
    lab = stage1["lab"]
    y_given = stage1["y_given"]
    N = len(lab)
    out_dir = Path(cfg.output_dir)

    y = y_given.copy()
    std_img = np.full(N, np.nan)
    have_r = False

    if cfg.ratings_file:
        rt = load_ratings(cfg.ratings_file).drop_duplicates("name").set_index("name")
        have_r = lab.name.isin(rt.index).mean() > 0.9

    if have_r:
        rr = rt.reindex(lab.name).reset_index()
        std_img = rr.r_std.values
        col = {"mean": "r_mean", "median": "r_median",
               "trim": "r_trim"}[cfg.target_type]
        y = rr[col].values

        tgt = pd.DataFrame({
            "mean_vs_median_MAE": [np.abs(rr.r_mean - rr.r_median).mean()],
            "mean_vs_trim_MAE": [np.abs(rr.r_mean - rr.r_trim).mean()],
            "corr_mean_median": [
                np.corrcoef(rr.r_mean, rr.r_median)[0, 1]
            ],
            "mean_rater_std": [np.nanmean(std_img)],
            "split_half_r": [np.corrcoef(rr.h1, rr.h2)[0, 1]],
        })
        r_ = tgt.split_half_r[0]
        tgt["spearman_brown_ceiling_r"] = 2 * r_ / (1 + r_)
        tgt.to_csv(out_dir / "task2_target_comparison.csv", index=False)
        log("TARGET\n" + tgt.round(4).to_string())

        fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
        ax[0].hist(rr.r_mean, 40, alpha=0.6, label="mean")
        ax[0].hist(rr.r_median, 40, alpha=0.6, label="median")
        ax[0].hist(rr.r_trim, 40, alpha=0.4, label="10% trimmed")
        ax[0].legend()
        ax[0].set_title("Target definitions")
        ax[1].hist(std_img, 40, color="C3")
        ax[1].set_title("Rater disagreement: std per image")
        ax[2].scatter(rr.r_mean, std_img, s=4, alpha=0.4)
        ax[2].set_xlabel("mean rating")
        ax[2].set_ylabel("rater std")
        ax[2].set_title("Disagreement vs rating")
        fig.tight_layout()
        fig.savefig(out_dir / "fig02_rater_disagreement.png", dpi=130)
        plt.close(fig)
    else:
        log("NOTE: no per-rater file -> target = given mean score")

    result = {"y": y, "std_img": std_img, "have_r": have_r}
    ckpt.save("stage4_target", result)
    ckpt.mark_done("stage4_target", time.time() - t0)
    log(f"Stage 4 done in {time.time() - t0:.1f}s")
    return result


# ================================================================
# CLI entry point: run preprocessing stages only
# ================================================================
if __name__ == "__main__":
    cfg, resume = PipelineConfig.from_args()
    ckpt = CheckpointManager(cfg.checkpoint_dir)
    if not resume:
        ckpt.reset()

    s1 = run_audit(cfg, ckpt)
    s2 = run_duplicates(cfg, ckpt, s1)
    s3 = run_splits(cfg, ckpt, s1, s2)
    s4 = run_target(cfg, ckpt, s1)

    log("Preprocessing complete! Stages 1-4 saved to checkpoints/")
    log("Run 'python run_pipeline.py --resume' to continue with modelling.")
