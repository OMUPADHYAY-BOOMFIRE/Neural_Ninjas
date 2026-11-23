#!/usr/bin/env python
"""
02_extract_features.py
================================================================================
Neural Ninjas (AIP301 Machine Learning Lab, IIIT Kota)
Submitted to: Dr. Shibna Hussain

STEP 2: Classical Computer Vision Feature Extraction
--------------------------------------------------------------------------------
This script extracts handcrafted classical visual descriptors without deep neural nets:
1. HOG (Histogram of Oriented Gradients): captures edge orientations and contour contrast.
2. LBP (Local Binary Patterns): captures skin micro-texture and grain.
3. COL (Color Histograms): HSV distributions (low_rgb) and intensity histograms (high_gray).
4. EDGE (Canny Edge Grid + Sobel Gradients): spatial distribution of contours across 64 zones.
5. GEO (Facial Geometry Ratios): neoclassical canons, aspect ratios, eye distances,
   bilateral symmetry correlation, and forehead hair occlusion from Haar cascades.

Extracts two compression variants in parallel:
- low_rgb   (128x128 pixels, 3 channels): high detail, 8,458 dimensions total.
- high_gray (48x48 pixels, 1 channel): fast and compact, 1,244 dimensions total.

Outputs:
- checkpoints/stage1_audit.pkl (feature matrices, image quality metrics, hashes)
- out/task3_feature_table.csv  (feature economics: dimensionality, extraction latency, RAM)

Usage:
    python 02_extract_features.py --data_dir Images --labels labels.txt --out out
"""

import os
import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from config import PipelineConfig, CheckpointManager, log, log_separator
from features import (
    VARIANTS, BLOCKS, QUAL_NAMES, extract_one, quality, dhash
)


def extract_single_image_features(image_info, data_dir):
    """
    Worker function to process one image: computes both variants (low_rgb, high_gray),
    extracts all 5 feature blocks, computes image quality metrics, and computes dHash.
    """
    image_name, rating_score = image_info
    image_file = Path(data_dir) / image_name
    
    # If file was moved to duplicate directory, read from there
    if not image_file.exists():
        image_file = Path("duplicate") / image_name

    image_bgr = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
    if image_bgr is None or image_bgr.size == 0:
        return None

    # Call feature extraction engine
    result = extract_one(image_bgr, VARIANTS)
    result["name"] = image_name
    result["score"] = rating_score
    return result


def run_stage2_feature_extraction(config, checkpoint_mgr):
    """
    Coordinates multi-core feature extraction across the verified dataset.
    """
    log_separator("STAGE 2: Multi-Modal Feature Extraction")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if stage is already cached
    if checkpoint_mgr.is_done("stage1_audit"):
        log("Features already cached in checkpoints/stage1_audit.pkl. Loading...")
        cached_data = checkpoint_mgr.load("stage1_audit")
        log("Cached features loaded successfully.")
        return cached_data

    # Read label entries
    label_path = Path(config.labels_file)
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip().split() for line in f if len(line.strip().split()) >= 2]
    image_items = [(p[0], float(p[1])) for p in lines]

    if config.limit > 0:
        image_items = image_items[:config.limit]
        log(f"Operating in limited smoke-test mode: {len(image_items)} images.")
    else:
        log(f"Extracting features across all {len(image_items)} verified images...")

    start_time = time.time()
    
    # Parallel processing across all available CPU threads
    extracted_records = Parallel(n_jobs=config.n_jobs, batch_size=config.batch_size)(
        delayed(extract_single_image_features)(item, config.data_dir)
        for item in image_items
    )

    # Filter any unreadable results
    extracted_records = [r for r in extracted_records if r is not None]
    total_extracted = len(extracted_records)
    elapsed_time = time.time() - start_time
    log(f"Extracted features for {total_extracted} images in {elapsed_time:.1f} s ({elapsed_time/total_extracted*1000:.2f} ms/image).")

    # Organize features into numpy arrays per variant and block
    feature_store = {}
    for var_name in VARIANTS:
        feature_store[var_name] = {}
        for block_name in BLOCKS:
            block_matrix = np.vstack([r["F"][var_name][block_name] for r in extracted_records]).astype(np.float32)
            feature_store[var_name][block_name] = block_matrix

    raw32_matrix = np.vstack([r["raw32"] for r in extracted_records]).astype(np.float32)
    quality_matrix = np.vstack([r["qual"] for r in extracted_records]).astype(np.float32)
    hashes_array = np.array([r["hash"] for r in extracted_records], dtype=np.uint64)
    names_list = [r["name"] for r in extracted_records]
    scores_array = np.array([r["score"] for r in extracted_records], dtype=np.float32)

    # Build Feature Economics Summary Table
    economics_rows = []
    for var_name, var_info in VARIANTS.items():
        for block_name in BLOCKS:
            mat = feature_store[var_name][block_name]
            dim = mat.shape[1]
            ram_mb = mat.nbytes / (1024 * 1024)
            nan_rate = float(np.isnan(mat).mean())
            
            economics_rows.append({
                "variant": var_name,
                "block": block_name,
                "dimension": dim,
                "storage_MB": round(ram_mb, 2),
                "nan_fraction": round(nan_rate, 4),
            })

    economics_df = pd.DataFrame(economics_rows)
    economics_df.to_csv(output_dir / "task3_feature_table.csv", index=False)
    log(f"Saved feature economics table to {output_dir / 'task3_feature_table.csv'}")

    # Build dataframe for downstream modules
    labels_df = pd.DataFrame({"name": names_list, "score": scores_array})
    prefix_match = labels_df["name"].str.extract(r"^([AaCc])([MmFf])")
    labels_df["ethnicity"] = prefix_match[0].str.upper().map({"A": "Asian", "C": "Caucasian"})
    labels_df["gender"] = prefix_match[1].str.upper().map({"M": "Male", "F": "Female"})
    labels_df["group"] = labels_df["ethnicity"].fillna("Unknown") + " " + labels_df["gender"].fillna("Unknown")

    result_payload = {
        "F": feature_store,
        "raw32": raw32_matrix,
        "qual": quality_matrix,
        "hashes": hashes_array,
        "lab": labels_df,
        "names": names_list,
    }

    # Save to checkpoint
    checkpoint_mgr.save("stage1_audit", result_payload)
    log("Saved feature cache to checkpoints/stage1_audit.pkl")
    log_separator("STAGE 2 COMPLETE: Ready for Cross-Validation and Modelling")
    return result_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Classical Visual Feature Extraction")
    parser.add_argument("--data_dir", default="Images", help="Directory containing facial images")
    parser.add_argument("--labels", default="labels.txt", help="Path to labels text file")
    parser.add_argument("--out", default="out", help="Output directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images for testing (0 = all)")
    args = parser.parse_args()

    cfg = PipelineConfig(data_dir=args.data_dir, labels_file=args.labels, output_dir=args.out, limit=args.limit)
    ckpt = CheckpointManager(cfg.checkpoint_dir)
    run_stage2_feature_extraction(cfg, ckpt)
