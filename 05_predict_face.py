#!/usr/bin/env python
"""
05_predict_face.py
================================================================================
Neural Ninjas (AIP301 Machine Learning Lab, IIIT Kota)
Submitted to: Dr. Shibna Hussain

STEP 5: Production Inference CLI with Uncertainty Bounds and Quality Auditing
--------------------------------------------------------------------------------
Given an unrated facial photograph, this script estimates the expected human consensus rating:
1. Validates image readability, minimum resolution (>= 100px), and aspect ratio.
2. Audits image quality (blur via Laplacian variance, brightness, contrast) against
   training set percentiles (5th to 95th).
3. Verifies frontal face and landmark localization with classical Haar cascades.
4. Generates predictions from all 3 diverse pipelines:
   - P1 (Appearance: HOG + LBP + COL -> Ridge)
   - P2 (Geometry: Neoclassical Ratios + Edge Grid -> HistGB)
   - P3 (From-Scratch Fusion: Whitened PCA -> Backprop MLP + Adam)
5. Computes ensemble rating and empirical 80% confidence interval based on CV residuals.
6. Flags automated uncertainty warnings if models disagree (std > 0.18) or quality degrades.
7. Displays mandatory ethical non-use disclaimer.

Usage:
    python 05_predict_face.py path/to/face.jpg [--bundle out/bundle.joblib]
"""

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
from joblib import load

from features import (
    VARIANTS, QUAL_NAMES, compute_blocks, quality, geo_names
)


def predict_single_image(image_path, bundle_path="out/bundle.joblib"):
    """
    Executes end-to-end inference for a single input portrait.
    """
    bundle_file = Path(bundle_path)
    if not bundle_file.exists():
        sys.exit(f"ERROR: Model bundle '{bundle_path}' not found! Please run step 03 first.")

    try:
        bundle = load(str(bundle_file))
    except Exception as exc:
        sys.exit(f"ERROR loading model bundle '{bundle_path}': {exc}")

    # Read image
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None or image_bgr.size == 0:
        sys.exit(f"ERROR: Failed to read image file '{image_path}'. Ensure file exists and is a valid image.")

    warnings = []
    img_height, img_width = image_bgr.shape[:2]

    # 1. Image Resolution & Geometry Checks
    if min(img_height, img_width) < 100:
        warnings.append(f"Low resolution ({img_width}x{img_height} px). Training images were >= 350x350 px.")
    if abs(img_height / img_width - 1.0) > 0.3:
        warnings.append("Non-square aspect ratio. Warping to square may distort neoclassical facial ratios.")

    # 2. Extract Features
    chosen_variant = bundle["variant"]
    feature_blocks, _ = compute_blocks(image_bgr, VARIANTS[chosen_variant])

    # 3. Quality Audits
    gray_128 = cv2.resize(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), (128, 128), interpolation=cv2.INTER_AREA)
    img_quality = quality(gray_128)

    for metric_name, value, q_low, q_high in zip(QUAL_NAMES, img_quality, bundle["qual_lo"], bundle["qual_hi"]):
        if value < q_low:
            warnings.append(f"{metric_name} unusually LOW ({value:.1f} < training 5th percentile {q_low:.1f}).")
        elif value > q_high:
            warnings.append(f"{metric_name} unusually HIGH ({value:.1f} > training 95th percentile {q_high:.1f}).")

    # 4. Landmark Verification
    names = geo_names(VARIANTS[chosen_variant]["rgb"])
    face_detected = feature_blocks["geo"][names.index("face_found")] > 0.5
    if not face_detected:
        warnings.append("Haar cascade failed to detect frontal face. Geometry features are estimated fallbacks.")
    
    missing_geo_frac = float(np.isnan(feature_blocks["geo"]).mean())
    if missing_geo_frac > 0.25:
        warnings.append(f"{missing_geo_frac:.0%} of facial landmark ratios missing (eyes/mouth occluded).")

    # 5. Pipeline Predictions
    pipeline_preds = {}
    for model_name, model_estimator in bundle["models"].items():
        feature_vector = np.hstack([feature_blocks[b] for b in bundle["blocks"][model_name]])[None]
        pipeline_preds[model_name] = float(model_estimator.predict(feature_vector)[0])

    preds_array = np.array(list(pipeline_preds.values()))
    ensemble_estimate = float(preds_array.mean())
    model_disagreement = float(preds_array.std())

    # Range and Disagreement Warnings
    y_min, y_max = bundle["y_range"]
    if ensemble_estimate < y_min or ensemble_estimate > y_max:
        warnings.append(f"Prediction ({ensemble_estimate:.2f}) falls outside dataset range [{y_min:.2f}, {y_max:.2f}].")
    if model_disagreement > bundle["dis_thr"]:
        warnings.append(f"High pipeline disagreement (std {model_disagreement:.2f} > threshold {bundle['dis_thr']:.2f}): confidence is reduced.")

    # 80% Empirical Interval from CV residuals
    resid_q10, resid_q90 = bundle["resid_q10_q90"]
    lower_bound = max(1.0, ensemble_estimate - resid_q90)
    upper_bound = min(5.0, ensemble_estimate - resid_q10)

    # Output Presentation
    print("\n" + "=" * 70)
    print("  NEURAL NINJAS — FACIAL RATING ESTIMATION (CLASSICAL ML)")
    print("=" * 70)
    print(f"  Input Image File           : {image_path}")
    print(f"  Image Dimensions           : {img_width} x {img_height} pixels")
    print("-" * 70)
    print("  Individual Pipeline Predictions:")
    for model_key, score_val in pipeline_preds.items():
        # Format friendly label
        if "P1" in model_key:
            label = "P1 Appearance (Ridge)"
        elif "P2" in model_key:
            label = "P2 Geometry (HistGB)"
        elif "P3" in model_key:
            label = "P3 From-Scratch (MLP)"
        else:
            label = model_key
        print(f"    {label:28s}: {score_val:.3f}")
    print("-" * 70)
    print(f"  Ensemble Estimated Rating  : {ensemble_estimate:.2f} / 5.00")
    print(f"  80% Confidence Interval    : [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  Pipeline Disagreement (SD) : {model_disagreement:.3f}")

    if warnings:
        print("\n  [!] SYSTEM DIAGNOSTIC WARNINGS:")
        for w in warnings:
            print(f"      • {w}")
    else:
        print("\n  [✓] Image quality checks passed (no optical anomalies detected).")

    print("\n" + "-" * 70)
    print("  MANDATORY ETHICAL DISCLAIMER:")
    print("  This prediction reflects the statistical consensus of an experimental")
    print("  annotator panel under laboratory studio lighting.")
    print("  DO NOT USE for hiring, admissions, dating filters, social profiling,")
    print("  or evaluating human worth. Predictions suffer from regression to the mean.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5: Estimate human face rating with uncertainty bounds")
    parser.add_argument("image", help="Path to input face image file")
    parser.add_argument("--bundle", default="out/bundle.joblib", help="Path to trained bundle file")
    args = parser.parse_args()

    predict_single_image(args.image, args.bundle)
