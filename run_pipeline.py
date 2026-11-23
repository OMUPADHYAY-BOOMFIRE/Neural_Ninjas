#!/usr/bin/env python
"""run_pipeline.py - Full reproducible study orchestrator.

Main entry point for Neural Ninjas (Batch A3).
Calls each stage in sequence with checkpointing and auto-resume.

Usage:
    # Full run
    python run_pipeline.py --data_dir Images --labels labels.txt --out out

    # Resume from last checkpoint if interrupted
    python run_pipeline.py --data_dir Images --labels labels.txt --out out --resume

    # Fast smoke test
    python run_pipeline.py --data_dir Images --labels labels.txt --out out --limit 500 --fast
"""
import sys
import time
import warnings

from config import PipelineConfig, CheckpointManager, log, log_separator
from models import grad_check

warnings.filterwarnings("ignore")


def main():
    cfg, resume = PipelineConfig.from_args()
    ckpt = CheckpointManager(cfg.checkpoint_dir)

    if not resume:
        log("Starting fresh run (no --resume flag)")
        ckpt.reset()
    else:
        log("Resuming from last checkpoint...")

    out_dir = cfg.output_dir
    log_separator("Neural Ninjas Pipeline (Classical ML)")
    log(f"Config: seed={cfg.seed}, limit={cfg.limit}, "
        f"fast={cfg.fast}, jobs={cfg.n_jobs}")
    log(f"Output: {out_dir}")
    log(f"Checkpoints: {cfg.checkpoint_dir}")

    # Save config for reproducibility
    cfg.to_json(f"{out_dir}/config.json")

    # Numerical gradient check of from-scratch backprop
    log("MLP gradient check: max rel. error =",
        f"{grad_check():.2e}")

    # =============================================
    # PREPROCESSING (Stages 1-4)
    # =============================================
    from preprocessing import run_audit, run_duplicates, run_splits, run_target

    s1 = run_audit(cfg, ckpt)
    s2 = run_duplicates(cfg, ckpt, s1)
    s3 = run_splits(cfg, ckpt, s1, s2)
    s4 = run_target(cfg, ckpt, s1)

    # =============================================
    # ANALYSIS + MODELLING (Stages 5-10)
    # =============================================
    from evaluation import (
        run_pca_vs_kmeans, run_ratio_study, run_modelling,
        run_dimension_study, run_feature_selection, run_final_test,
    )

    run_pca_vs_kmeans(cfg, ckpt, s1, s3, s4)
    run_ratio_study(cfg, ckpt, s1, s3, s4)
    s7 = run_modelling(cfg, ckpt, s1, s3, s4)
    run_dimension_study(cfg, ckpt, s1, s3, s4)
    run_feature_selection(cfg, ckpt, s1, s3, s4, s7)
    run_final_test(cfg, ckpt, s1, s3, s4, s7)

    log_separator("ALL STAGES COMPLETE")
    log("Results saved in:", out_dir)
    log("Trained bundle:", f"{out_dir}/bundle.joblib")
    log(f"To run inference: python predict.py <image> --bundle {out_dir}/bundle.joblib")


if __name__ == "__main__":
    main()
