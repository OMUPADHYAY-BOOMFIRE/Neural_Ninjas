#!/usr/bin/env python
"""
04_run_experiments_and_studies.py
================================================================================
Neural Ninjas (AIP301 Machine Learning Lab, IIIT Kota)
Submitted to: Dr. Shibna Hussain

STEP 4: Research Studies, Diagnostics, Fairness Audits & Final Test Evaluation
--------------------------------------------------------------------------------
This script runs the in-depth analytical experiments required by the assignment:
1. Unsupervised Preprocessing: Compares PCA vs KMeans cluster distances.
2. Facial Ratio Analysis: Evaluates neoclassical proportions and resolution impact.
3. Empirical Dimensionality Study: Tests p < n, p ~ n, p > n with the interpolation peak.
4. Feature Selection & Sparsity: Evaluates L1/Lasso, SelectKBest, Variance, and PCA with
   Jaccard selection stability across folds.
5. Error Diagnostics & Mann-Whitney Test: Tests whether optical blur statistically
   explains top-decile prediction errors (p = 0.0446).
6. Demographic Fairness & Regression to the Mean: Audits errors across AF, AM, CF, CM
   and quantifies range compression across ground-truth quintiles.
7. Final Test Evaluation: Runs the ensemble exactly once on the held-out 818 test images.

Usage:
    python 04_run_experiments_and_studies.py --out out
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import load

from config import PipelineConfig, CheckpointManager, log, log_separator
from evaluation import (
    run_pca_vs_kmeans, run_ratio_study, run_dimension_study,
    run_feature_selection, run_final_test
)


def run_stage4_all_studies(config, checkpoint_mgr):
    """
    Executes all academic studies and final test evaluation.
    """
    log_separator("STAGE 4: Experimental Studies & Academic Diagnostics")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load required cached stages
    if not checkpoint_mgr.is_done("stage1_audit"):
        sys.exit("ERROR: stage1_audit checkpoint missing! Run 01 and 02 first.")
    if not checkpoint_mgr.is_done("stage3_splits"):
        sys.exit("ERROR: stage3_splits checkpoint missing! Run 01 first.")

    s1 = checkpoint_mgr.load("stage1_audit")
    s3 = checkpoint_mgr.load("stage3_splits")
    s4 = checkpoint_mgr.load("stage4_target")

    # 1. PCA vs KMeans
    log_separator("Study 1: PCA vs KMeans as Unsupervised Preprocessing")
    run_pca_vs_kmeans(config, checkpoint_mgr, s1, s3, s4)

    # 2. Facial Ratios & Compression Study
    log_separator("Study 2: Facial Ratios & Resolution Compression")
    run_ratio_study(config, checkpoint_mgr, s1, s3, s4)

    # 3. Dimensionality Study (p < n, p ~ n, p > n)
    log_separator("Study 3: Empirical Dimensionality & Interpolation Peak")
    run_dimension_study(config, checkpoint_mgr, s1, s3, s4)

    # 4. Feature Selection & Jaccard Stability
    log_separator("Study 4: Feature Selection, Sparsity & Jaccard Stability")
    # Needs stage 7 modelling checkpoint or runs with dev features
    if checkpoint_mgr.is_done("stage7_modelling"):
        s7 = checkpoint_mgr.load("stage7_modelling")
    else:
        # Load bundle if trained via 03
        bundle_path = output_dir / "bundle.joblib"
        if bundle_path.exists():
            bundle = load(bundle_path)
            s7 = {"bundle": bundle, "cv_summary": {}}
        else:
            from evaluation import run_modelling
            s7 = run_modelling(config, checkpoint_mgr, s1, s3, s4)

    run_feature_selection(config, checkpoint_mgr, s1, s3, s4, s7)

    # 5. Final Held-Out Test Evaluation & Fairness Audits
    log_separator("Study 5: Final Held-Out Test Evaluation (n = 818) & Fairness")
    run_final_test(config, checkpoint_mgr, s1, s3, s4, s7)

    log_separator("STAGE 4 COMPLETE: All Figures and Result Tables Generated")
    log(f"All artifacts saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: Research Studies, Diagnostics and Fairness")
    parser.add_argument("--out", default="out", help="Output directory")
    args = parser.parse_args()

    cfg = PipelineConfig(output_dir=args.out)
    ckpt = CheckpointManager(cfg.checkpoint_dir)
    run_stage4_all_studies(cfg, ckpt)
