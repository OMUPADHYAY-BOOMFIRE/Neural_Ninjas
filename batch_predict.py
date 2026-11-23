#!/usr/bin/env python
"""
batch_predict.py
================================================================================
Neural Ninjas (AIP301 Machine Learning Lab, IIIT Kota)

Batch Image Prediction Script — Input Folder → Output Results
--------------------------------------------------------------------------------
How to use:
  1. Drop any face photos (jpg, jpeg, png, bmp, webp) into the  input/  folder.
  2. Run:   python batch_predict.py
  3. Find results in  input-output/results.txt  and one  input-output/<name>.txt
     per image.

What you get per image:
  - P1 (Appearance Ridge), P2 (Geometry HistGB), P3 (From-Scratch MLP) scores
  - Ensemble mean rating (1.0 to 5.0 scale)
  - 80% empirical confidence interval
  - Rating tier label  (Very Low / Low / Average / High / Very High)
  - Quality warnings (blur, brightness, face detection, resolution)
  - Pipeline disagreement flag

The bundle used is  out/bundle.joblib  — the model trained on the full pipeline.
Run  python run_pipeline.py  first if that file does not exist yet.
"""

import sys
import os
import time
import shutil
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from joblib import load

from features import VARIANTS, QUAL_NAMES, compute_blocks, quality, geo_names


# ---------------------------------------------------------------------------
# Rating tier labels — maps the numeric ensemble score to a descriptive level
# ---------------------------------------------------------------------------
def rating_tier(score):
    if score < 2.0:
        return "Very Low  (bottom tier)"
    elif score < 2.6:
        return "Low       (below average)"
    elif score < 3.4:
        return "Average   (mid-range consensus)"
    elif score < 4.1:
        return "High      (above average)"
    else:
        return "Very High (top tier)"


# ---------------------------------------------------------------------------
# Single image prediction — returns a dict with all metrics
# ---------------------------------------------------------------------------
def predict_one(image_path, bundle):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return {"error": f"Cannot read file: {image_path}"}

    warnings = []
    h, w = img.shape[:2]

    if min(h, w) < 50:
        warnings.append(f"Very low resolution ({w}x{h} px) — accuracy may suffer")
    if abs(h / w - 1.0) > 0.4:
        warnings.append("Non-square aspect ratio — geometric ratios may be distorted")

    # Feature extraction
    var_cfg = VARIANTS[bundle["variant"]]
    feature_blocks, _ = compute_blocks(img, var_cfg)

    # Image quality metrics
    gray128 = cv2.resize(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        (128, 128), interpolation=cv2.INTER_AREA
    )
    img_quality = quality(gray128)
    for q_name, q_val, q_lo, q_hi in zip(
        QUAL_NAMES, img_quality, bundle["qual_lo"], bundle["qual_hi"]
    ):
        if q_val < q_lo:
            warnings.append(f"{q_name} unusually LOW ({q_val:.1f} < train 5th pct {q_lo:.1f})")
        elif q_val > q_hi:
            warnings.append(f"{q_name} unusually HIGH ({q_val:.1f} > train 95th pct {q_hi:.1f})")

    # Face landmark check
    names = geo_names(var_cfg["rgb"])
    face_found = feature_blocks["geo"][names.index("face_found")] > 0.5
    if not face_found:
        warnings.append("Haar cascade: no frontal face detected — geometry is estimated")
    missing_geo = float(np.isnan(feature_blocks["geo"]).mean())
    if missing_geo > 0.25:
        warnings.append(f"{missing_geo:.0%} of facial geometry ratios missing")

    # Pipeline predictions
    pipeline_scores = {}
    for model_name, estimator in bundle["models"].items():
        feat_vec = np.hstack(
            [feature_blocks[b] for b in bundle["blocks"][model_name]]
        )[None]
        pipeline_scores[model_name] = float(estimator.predict(feat_vec)[0])

    scores = np.array(list(pipeline_scores.values()))
    ensemble = float(scores.mean())
    disagreement = float(scores.std())

    r10, r90 = bundle["resid_q10_q90"]
    ci_lo = max(1.0, ensemble - r90)
    ci_hi = min(5.0, ensemble - r10)

    y_min, y_max = bundle["y_range"]
    if ensemble < y_min or ensemble > y_max:
        warnings.append(
            f"Prediction {ensemble:.2f} outside training range [{y_min:.2f}, {y_max:.2f}]"
        )
    if disagreement > bundle["dis_thr"]:
        warnings.append(
            f"High pipeline disagreement (SD {disagreement:.3f} > thr {bundle['dis_thr']:.3f})"
        )

    return {
        "image": image_path.name,
        "resolution": f"{w}x{h}",
        "face_detected": face_found,
        "quality": {n: round(float(v), 2) for n, v in zip(QUAL_NAMES, img_quality)},
        "pipeline_scores": {k: round(v, 4) for k, v in pipeline_scores.items()},
        "ensemble_rating": round(ensemble, 4),
        "rating_tier": rating_tier(ensemble),
        "ci_80": (round(ci_lo, 3), round(ci_hi, 3)),
        "pipeline_disagreement_sd": round(disagreement, 4),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Format one result dict into a human-readable text block
# ---------------------------------------------------------------------------
def format_result(r):
    lines = []
    sep = "=" * 68
    thin = "-" * 68
    lines.append(sep)
    lines.append(f"  IMAGE : {r['image']}")
    lines.append(thin)

    if "error" in r:
        lines.append(f"  ERROR : {r['error']}")
        lines.append(sep)
        return "\n".join(lines)

    lines.append(f"  Resolution      : {r['resolution']} px")
    lines.append(f"  Face detected   : {'Yes' if r['face_detected'] else 'No (fallback geometry used)'}")

    q = r["quality"]
    lines.append(f"  Sharpness       : {q.get('blur_lapvar', 'N/A'):.1f}  "
                 f"(higher = sharper)")
    lines.append(f"  Brightness      : {q.get('brightness', 'N/A'):.1f}  "
                 f"Contrast: {q.get('contrast', 'N/A'):.1f}")
    lines.append(thin)

    lines.append("  Pipeline Predictions:")
    for model_key, score in r["pipeline_scores"].items():
        label = (
            "P1 Appearance (HOG+LBP+Ridge)" if "P1" in model_key else
            "P2 Geometry   (Ratios+HistGB)" if "P2" in model_key else
            "P3 From-Scratch (PCA + MLP)  "
        )
        lines.append(f"    {label}: {score:.4f}")

    lines.append(thin)
    lines.append(f"  ENSEMBLE RATING : {r['ensemble_rating']:.4f} / 5.0000")
    lines.append(f"  RATING TIER     : {r['rating_tier']}")
    lines.append(f"  80% CI          : [{r['ci_80'][0]:.3f}, {r['ci_80'][1]:.3f}]")
    lines.append(f"  Pipeline SD     : {r['pipeline_disagreement_sd']:.4f}")

    if r["warnings"]:
        lines.append(thin)
        lines.append("  WARNINGS:")
        for w in r["warnings"]:
            lines.append(f"    [!] {w}")
    else:
        lines.append(thin)
        lines.append("  [OK] All quality checks passed.")

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main — scan input/, predict all images, write input-output/
# ---------------------------------------------------------------------------
def main():
    input_dir = Path("input")
    output_dir = Path("input-output")
    bundle_path = Path("out/bundle.joblib")

    output_dir.mkdir(parents=True, exist_ok=True)

    if not bundle_path.exists():
        print(
            "[X] ERROR: out/bundle.joblib not found!\n"
            "    Please train the pipeline first:\n"
            "        python run_pipeline.py --data_dir Images --labels labels.txt --out out"
        )
        sys.exit(1)

    print("=" * 68)
    print("  NEURAL NINJAS — BATCH FACE RATING PREDICTION")
    print("=" * 68)

    bundle = load(str(bundle_path))
    print(f"[+] Bundle loaded: variant={bundle['variant']}")

    # Collect supported image files from input/
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in supported_exts
    ])

    if not image_files:
        print(f"\n[!] No image files found in  {input_dir}/")
        print("    Paste jpg / jpeg / png / bmp / webp images into  input/  and re-run.")
        sys.exit(0)

    print(f"[+] Found {len(image_files)} image(s) in {input_dir}/")
    print(f"[+] Results will be saved to  {output_dir}/\n")

    all_results = []
    t_start = time.time()

    for img_path in image_files:
        print(f"  Processing: {img_path.name} ...", end=" ", flush=True)
        t0 = time.time()
        result = predict_one(img_path, bundle)
        elapsed = time.time() - t0

        if "error" in result:
            print(f"ERROR ({result['error']})")
        else:
            print(
                f"done  |  Rating: {result['ensemble_rating']:.3f}  "
                f"({result['rating_tier'].split('(')[0].strip()})  [{elapsed*1000:.0f} ms]"
            )

        all_results.append(result)

        # Write individual per-image file
        stem = img_path.stem
        per_file = output_dir / f"{stem}_result.txt"
        with open(per_file, "w", encoding="utf-8") as fh:
            fh.write(format_result(result) + "\n")
            fh.write(
                "\n[DISCLAIMER] This score is a statistical estimate from a specific "
                "rater panel.\nDO NOT use for hiring, admissions, or judging human worth.\n"
            )

    total_elapsed = time.time() - t_start

    # Write combined summary results.txt
    summary_path = output_dir / "results.txt"
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("NEURAL NINJAS — BATCH PREDICTION RESULTS\n")
        fh.write(f"Run timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"Images processed : {len(all_results)}\n")
        fh.write(f"Model bundle : out/bundle.joblib  (variant: {bundle['variant']})\n")
        fh.write("=" * 68 + "\n\n")

        # Quick score table at the top
        fh.write("SUMMARY TABLE\n")
        fh.write("-" * 68 + "\n")
        fh.write(f"{'Image':<35} {'Rating':>8}  {'Tier':<30} Warnings\n")
        fh.write("-" * 68 + "\n")
        for r in all_results:
            if "error" in r:
                fh.write(f"{r['image']:<35} {'ERROR':>8}  {r['error']}\n")
            else:
                tier_short = r["rating_tier"].split("(")[0].strip()
                warn_flag = "!" if r["warnings"] else " "
                fh.write(
                    f"{r['image']:<35} {r['ensemble_rating']:>8.4f}  "
                    f"{tier_short:<30} [{warn_flag}]\n"
                )
        fh.write("\n")

        # Full detail blocks
        for r in all_results:
            fh.write(format_result(r) + "\n\n")

        fh.write("=" * 68 + "\n")
        fh.write(
            f"Total processing time: {total_elapsed:.2f}s  "
            f"({total_elapsed / max(len(all_results), 1) * 1000:.0f} ms/image avg)\n"
        )
        fh.write(
            "\n[DISCLAIMER] Predictions are statistical estimates from a specific "
            "annotator panel.\nDO NOT use for hiring, admissions, dating filters, "
            "or evaluating human worth.\n"
        )

    print(f"\n{'='*68}")
    print(f"  DONE — {len(all_results)} image(s) processed in {total_elapsed:.2f}s")
    print(f"  Results summary : {summary_path}")
    print(f"  Per-image files : {output_dir}/<name>_result.txt")
    print("=" * 68)


if __name__ == "__main__":
    main()
