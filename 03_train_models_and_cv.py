#!/usr/bin/env python
"""
03_train_models_and_cv.py
================================================================================
Neural Ninjas (AIP301 Machine Learning Lab, IIIT Kota)
Submitted to: Dr. Shibna Hussain

STEP 3: Repeated Group Cross-Validation and Model Training
--------------------------------------------------------------------------------
This script trains and validates the candidate classical machine learning models:
1. Baseline B0: Dummy regressor predicting training fold mean (null reference).
2. Baseline B1: Raw 32x32 grayscale pixels with Ridge regression.
3. Pipeline P1 (Student 1 - Appearance): HOG + LBP + COL -> StandardScaler -> RidgeCV.
4. Pipeline P2 (Student 2 - Geometry): GEO + EDGE -> HistGradientBoostingRegressor.
5. Pipeline P3 (Student 3 - From-Scratch): Custom Whitened PCA + Custom MLP (Backprop + Adam).

Validation Protocol:
- Repeated Group 5-Fold Cross-Validation (3 repeats x 5 folds = 15 total folds).
- Group-aware: duplicate clusters never split across train and validation folds.
- Evaluates both compression variants (low_rgb 128px vs high_gray 48px).
- Generates out/task8_cv_summary.csv and bundles final trained models into out/bundle.joblib.

Usage:
    python 03_train_models_and_cv.py --out out
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load

from config import PipelineConfig, CheckpointManager, log, log_separator
from models import (
    factories, sel_blocks, fit_any, model_bytes, metrics, run_cv
)


def run_stage3_cross_validation(config, checkpoint_mgr):
    """
    Executes 15-fold repeated group cross-validation across all models and variants.
    """
    log_separator("STAGE 3: Repeated Group Cross-Validation (15 Folds)")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load required cached stages
    if not checkpoint_mgr.is_done("stage1_audit"):
        sys.exit("ERROR: stage1_audit checkpoint missing! Run 01 and 02 first.")
    if not checkpoint_mgr.is_done("stage3_splits"):
        # Run split step if not yet packaged in checkpoints
        from preprocessing import run_duplicates, run_splits, run_target
        s1 = checkpoint_mgr.load("stage1_audit")
        s2 = run_duplicates(config, checkpoint_mgr, s1)
        s3 = run_splits(config, checkpoint_mgr, s1, s2)
        s4 = run_target(config, checkpoint_mgr, s1)
    else:
        s1 = checkpoint_mgr.load("stage1_audit")
        s3 = checkpoint_mgr.load("stage3_splits")
        s4 = checkpoint_mgr.load("stage4_target")

    F = s1["F"]
    raw32 = s1["raw32"]
    dev_indices = s3["dev"]
    cv_folds = s3["folds"]
    target_y = s4["y"]
    group_ids = s3["gid"]

    y_dev = target_y[dev_indices]
    gid_dev = group_ids[dev_indices]
    raw32_dev = raw32[dev_indices]

    log(f"Evaluating models on Development Set: {len(dev_indices)} samples across 15 folds (3 repeats x 5 folds)...")

    # Evaluate all models across variants
    cv_summary_records = []
    
    # 1. Baseline B0: Training Mean
    log("Running Baseline B0: Train Mean Dummy...")
    m_b0 = factories["B0_mean"]()
    # Dummy regressor CV
    b0_preds = np.zeros_like(y_dev)
    b0_ms = []
    for rep_folds in cv_folds:
        for tr_idx, va_idx in rep_folds:
            pos_map = {idx: pos for pos, idx in enumerate(dev_indices)}
            tr_pos = [pos_map[i] for i in tr_idx]
            va_pos = [pos_map[i] for i in va_idx]
            mean_val = float(np.mean(y_dev[tr_pos]))
            b0_preds[va_pos] = mean_val
            b0_ms.append(metrics(y_dev[va_pos], b0_preds[va_pos]))
    
    cv_summary_records.append({
        "pipeline": "B0_mean",
        "variant": "none",
        "features": 0,
        "cv_mae": np.mean([m["MAE"] for m in b0_ms]),
        "cv_rmse": np.mean([m["RMSE"] for m in b0_ms]),
        "cv_rmse_std": np.std([m["RMSE"] for m in b0_ms]),
        "cv_medae": np.mean([m["MedAE"] for m in b0_ms]),
        "cv_pearson": np.mean([m["Pearson"] for m in b0_ms]),
        "fit_time_s": 0.001,
        "size_kb": 1.0,
    })

    # 2. Baseline B1: Raw 32x32 Pixels + Ridge
    log("Running Baseline B1: Raw 32x32 Grayscale Pixels + Ridge...")
    b1_res = run_cv(factories["B1_raw32"](), raw32_dev, y_dev, cv_folds, dev_indices)
    cv_summary_records.append({
        "pipeline": "B1_raw32",
        "variant": "raw32",
        "features": 1024,
        "cv_mae": b1_res["mae"],
        "cv_rmse": b1_res["rmse"],
        "cv_rmse_std": b1_res["rmse_std"],
        "cv_medae": b1_res["medae"],
        "cv_pearson": b1_res["pearson"],
        "fit_time_s": b1_res["fit_time"],
        "size_kb": b1_res["size_kb"],
    })

    # 3. Pipelines P1, P2, P3 across low_rgb and high_gray
    for var_name in ["low_rgb", "high_gray"]:
        log_separator(f"Evaluating Pipelines on Variant: {var_name}")
        
        # Build feature matrices for this variant
        feat_p1 = np.hstack([F[var_name][b] for b in ["hog", "lbp", "col"]])[dev_indices]
        feat_p2 = np.hstack([F[var_name][b] for b in ["edge", "geo"]])[dev_indices]
        feat_p3 = np.hstack([F[var_name][b] for b in ["hog", "lbp", "col", "edge", "geo"]])[dev_indices]

        # P1 Appearance
        log(f"  Training P1 (Appearance Ridge) [{feat_p1.shape[1]} features]...")
        res_p1 = run_cv(factories["P1_appearance"](), feat_p1, y_dev, cv_folds, dev_indices)
        cv_summary_records.append({
            "pipeline": "P1_appearance",
            "variant": var_name,
            "features": feat_p1.shape[1],
            "cv_mae": res_p1["mae"],
            "cv_rmse": res_p1["rmse"],
            "cv_rmse_std": res_p1["rmse_std"],
            "cv_medae": res_p1["medae"],
            "cv_pearson": res_p1["pearson"],
            "fit_time_s": res_p1["fit_time"],
            "size_kb": res_p1["size_kb"],
        })

        # P2 Geometry
        log(f"  Training P2 (Geometry HistGB) [{feat_p2.shape[1]} features]...")
        res_p2 = run_cv(factories["P2_geometry"](), feat_p2, y_dev, cv_folds, dev_indices)
        cv_summary_records.append({
            "pipeline": "P2_geometry",
            "variant": var_name,
            "features": feat_p2.shape[1],
            "cv_mae": res_p2["mae"],
            "cv_rmse": res_p2["rmse"],
            "cv_rmse_std": res_p2["rmse_std"],
            "cv_medae": res_p2["medae"],
            "cv_pearson": res_p2["pearson"],
            "fit_time_s": res_p2["fit_time"],
            "size_kb": res_p2["size_kb"],
        })

        # P3 From-Scratch
        log(f"  Training P3 (From-Scratch PCA + MLP) [{feat_p3.shape[1]} -> 100 components]...")
        res_p3 = run_cv(factories["P3_fromscratch"](), feat_p3, y_dev, cv_folds, dev_indices, groups=gid_dev)
        cv_summary_records.append({
            "pipeline": "P3_fromscratch",
            "variant": var_name,
            "features": feat_p3.shape[1],
            "cv_mae": res_p3["mae"],
            "cv_rmse": res_p3["rmse"],
            "cv_rmse_std": res_p3["rmse_std"],
            "cv_medae": res_p3["medae"],
            "cv_pearson": res_p3["pearson"],
            "fit_time_s": res_p3["fit_time"],
            "size_kb": res_p3["size_kb"],
        })

    # Save summary table
    summary_df = pd.DataFrame(cv_summary_records)
    summary_df.to_csv(output_dir / "task8_cv_summary.csv", index=False)
    log(f"Saved 15-fold cross-validation summary to {output_dir / 'task8_cv_summary.csv'}")

    # Build production bundle on low_rgb
    log_separator("Packaging Production Inference Bundle (low_rgb)")
    chosen_variant = "low_rgb"
    X_dev_p1 = np.hstack([F[chosen_variant][b] for b in ["hog", "lbp", "col"]])[dev_indices]
    X_dev_p2 = np.hstack([F[chosen_variant][b] for b in ["edge", "geo"]])[dev_indices]
    X_dev_p3 = np.hstack([F[chosen_variant][b] for b in ["hog", "lbp", "col", "edge", "geo"]])[dev_indices]

    log("Fitting final P1 model on all dev data...")
    final_p1 = factories["P1_appearance"]().fit(X_dev_p1, y_dev)
    log("Fitting final P2 model on all dev data...")
    final_p2 = factories["P2_geometry"]().fit(X_dev_p2, y_dev)
    log("Fitting final P3 model on all dev data...")
    final_p3 = factories["P3_fromscratch"]().fit(X_dev_p3, y_dev, groups=gid_dev)

    # Compute empirical CV residual distribution for uncertainty intervals
    p1_cv_preds = res_p1["cv_preds"]
    p2_cv_preds = res_p2["cv_preds"]
    p3_cv_preds = res_p3["cv_preds"]
    ens_cv_preds = (p1_cv_preds + p2_cv_preds + p3_cv_preds) / 3.0
    cv_residuals = ens_cv_preds - y_dev

    r10 = float(np.percentile(cv_residuals, 10))
    r90 = float(np.percentile(cv_residuals, 90))
    pipeline_disagreements = np.std([p1_cv_preds, p2_cv_preds, p3_cv_preds], axis=0)
    disagree_threshold = float(np.percentile(pipeline_disagreements, 75))

    qual_dev = s1["qual"][dev_indices]
    qual_lo = np.percentile(qual_dev, 5, axis=0)
    qual_hi = np.percentile(qual_dev, 95, axis=0)

    production_bundle = {
        "variant": chosen_variant,
        "models": {
            "P1_appearance": final_p1,
            "P2_geometry": final_p2,
            "P3_fromscratch": final_p3,
        },
        "blocks": {
            "P1_appearance": ["hog", "lbp", "col"],
            "P2_geometry": ["edge", "geo"],
            "P3_fromscratch": ["hog", "lbp", "col", "edge", "geo"],
        },
        "resid_q10_q90": (r10, r90),
        "dis_thr": disagree_threshold,
        "qual_lo": qual_lo,
        "qual_hi": qual_hi,
        "y_range": (float(y_dev.min()), float(y_dev.max())),
    }

    dump(production_bundle, output_dir / "bundle.joblib")
    log(f"Saved inference bundle to {output_dir / 'bundle.joblib'}")
    log_separator("STAGE 3 COMPLETE: Models Trained and Cross-Validated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: Train Models and Run Cross-Validation")
    parser.add_argument("--out", default="out", help="Output directory")
    args = parser.parse_args()

    cfg = PipelineConfig(output_dir=args.out)
    ckpt = CheckpointManager(cfg.checkpoint_dir)
    run_stage3_cross_validation(cfg, ckpt)
