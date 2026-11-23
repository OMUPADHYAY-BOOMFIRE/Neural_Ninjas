#!/usr/bin/env python
"""predict.py - Single-image inference with uncertainty + quality warnings.

Usage:
    python predict.py <image_path> [--bundle out/bundle.joblib]

Outputs:
    - Predictions from P1 (Appearance), P2 (Geometry), P3 (Fusion)
    - Ensemble mean prediction
    - 80% confidence interval based on cross-validation residual distribution
    - Uncertainty metric: pipeline disagreement
    - Image quality checks & warnings
"""
import argparse
import sys

import numpy as np
import cv2
from joblib import load

from features import (
    VARIANTS, QUAL_NAMES, compute_blocks, quality, geo_names,
)


def main():
    ap = argparse.ArgumentParser(
        description="Predict average human rating for a face image."
    )
    ap.add_argument("image", help="Path to face image file")
    ap.add_argument(
        "--bundle", default="out/bundle.joblib",
        help="Path to trained bundle file (default: out/bundle.joblib)",
    )
    a = ap.parse_args()

    try:
        b = load(a.bundle)
    except Exception as e:
        sys.exit(f"ERROR loading model bundle '{a.bundle}': {e}\nPlease train the pipeline first.")

    img = cv2.imread(a.image, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        sys.exit(f"ERROR: Image file '{a.image}' cannot be read.")

    warns = []
    H, W = img.shape[:2]

    # Input resolution checks
    if min(H, W) < 100:
        warns.append(f"Low resolution ({W}x{H}); training images were >= 350x350")
    if abs(H / W - 1) > 0.3:
        warns.append("Non-square aspect ratio; square warping distorts facial geometry ratios")

    # Feature extraction
    feats, _ = compute_blocks(img, VARIANTS[b["variant"]])
    g128 = cv2.resize(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        (128, 128), interpolation=cv2.INTER_AREA,
    )
    q = quality(g128)

    for n_, v_, lo, hi in zip(QUAL_NAMES, q, b["qual_lo"], b["qual_hi"]):
        if v_ < lo:
            warns.append(f"{n_} unusually LOW ({v_:.1f} < train 5th pct {lo:.1f})")
        if v_ > hi:
            warns.append(f"{n_} unusually HIGH ({v_:.1f} > train 95th pct {hi:.1f})")

    names = geo_names(VARIANTS[b["variant"]]["rgb"])
    found = feats["geo"][names.index("face_found")] > 0.5
    if not found:
        warns.append("No frontal face detected by Haar cascade; geometry features are fallback estimates")
    if np.isnan(feats["geo"]).mean() > 0.3:
        warns.append(f"{np.isnan(feats['geo']).mean():.0%} of facial geometry ratios missing (eyes/mouth occluded)")

    # Predictions
    preds = {}
    for name, m in b["models"].items():
        x = np.hstack([feats[k] for k in b["blocks"][name]])[None]
        preds[name] = float(m.predict(x)[0])

    p = np.array(list(preds.values()))
    est, dis = p.mean(), p.std()

    lo_, hi_ = b["y_range"]
    if est < lo_ or est > hi_:
        warns.append(f"Prediction ({est:.2f}) falls outside dataset range [{lo_:.2f}, {hi_:.2f}]")
    if dis > b["dis_thr"]:
        warns.append(f"High pipeline disagreement (std {dis:.2f} > threshold {b['dis_thr']:.2f}): lower confidence")

    r10, r90 = b["resid_q10_q90"]

    print("=" * 64)
    print("Neural Ninjas — Face Rating Estimation (Classical ML)")
    print("=" * 64)
    for k, v in preds.items():
        print(f"  {k:32s}: {v:.3f}")
    print("-" * 64)
    print(f"  Ensemble Estimated Rating     : {est:.2f} / 5.00")
    print(f"  80% Empirical Interval (CV)   : [{est - r90:.2f}, {est - r10:.2f}]")
    print(f"  Pipeline Disagreement (std)   : {dis:.3f} (CV RMSE: {b['cv_rmse']:.2f})")
    print("=" * 64)

    if warns:
        print("INPUT & CONFIDENCE WARNINGS:")
        for w in warns:
            print(f"  [!] {w}")
    else:
        print("No input-quality warnings detected.")

    print("-" * 64)
    print("ETHICAL BOUNDARY NOTICE:")
    print("This estimate reflects the statistical mean rating of a specific rater panel.")
    print("It does NOT measure human worth, objective beauty, character, or competence.")
    print("STRICTLY PROHIBITED for hiring, dating filter, admissions, or commercial profiling.")
    print("=" * 64)


if __name__ == "__main__":
    main()
