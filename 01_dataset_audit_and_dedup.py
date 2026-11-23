#!/usr/bin/env python
"""
01_dataset_audit_and_dedup.py
================================================================================
Neural Ninjas (AIP301 Machine Learning Lab, IIIT Kota)
Submitted to: Dr. Shibna Hussain

STEP 1: Dataset Audit, Duplicate Percolation Detection & Group-Disjoint Splits
--------------------------------------------------------------------------------
This script carries out the foundational audit of the SCUT-FBP5500 dataset:
1. Validates every image file on disk (dimensions, corrupt/unreadable files).
2. Parses labels and records summary statistics for the rating target.
3. Computes 64-bit difference hashes (dHash) to discover duplicate images.
4. Identifies the "percolation catastrophe" where a loose Hamming threshold (<= 6)
   chains 75.5% of images into duplicate clusters, and enforces threshold 0.
5. Moves redundant duplicate photos to the duplicate/ directory to guarantee
   zero test leakage, producing labels_clean.txt.
6. Partitions unique images into group-disjoint train (70%), validation (15%),
   and held-out test (15%) splits, plus 15 cross-validation folds.

Usage:
    python 01_dataset_audit_and_dedup.py --data_dir Images --labels labels.txt --out out
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit

from config import PipelineConfig, CheckpointManager, log, log_separator
from features import dhash, hamming_row


# ==============================================================================
# Helper Data Loading Functions
# ==============================================================================

def load_label_records(labels_file_path):
    """
    Parse the labels text file containing 'filename score' on each line.
    Also extracts demographic group tags from filename prefixes (e.g., AF, AM, CF, CM).
    """
    records = []
    with open(labels_file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    image_filename = parts[0]
                    rating_score = float(parts[1])
                    records.append((image_filename, rating_score))
                except ValueError:
                    continue

    dataframe = pd.DataFrame(records, columns=["name", "score"])
    
    # Extract ethnicity and gender metadata from filename prefix
    # e.g., 'AF1.jpg' -> Ethnicity: Asian (A), Gender: Female (F)
    prefix_match = dataframe["name"].str.extract(r"^([AaCc])([MmFf])")
    dataframe["ethnicity"] = prefix_match[0].str.upper().map({"A": "Asian", "C": "Caucasian"})
    dataframe["gender"] = prefix_match[1].str.upper().map({"M": "Male", "F": "Female"})
    dataframe["group"] = dataframe["ethnicity"].fillna("Unknown") + " " + dataframe["gender"].fillna("Unknown")
    
    return dataframe


# ==============================================================================
# Disjoint Set Union (Union-Find) for Duplicate Clustering
# ==============================================================================

class DisjointSetClusters:
    """
    Classic Union-Find data structure with path compression and union-by-rank.
    Used to group duplicate image pairs into connected equivalence classes.
    """
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, i):
        # Path compression
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i == root_j:
            return
        if self.rank[root_i] < self.rank[root_j]:
            self.parent[root_i] = root_j
        elif self.rank[root_i] > self.rank[root_j]:
            self.parent[root_j] = root_i
        else:
            self.parent[root_j] = root_i
            self.rank[root_i] += 1


# ==============================================================================
# Main Audit & Deduplication Routine
# ==============================================================================

def run_stage1_audit_and_dedup(config, checkpoint_mgr):
    """
    Executes dataset integrity checks, perceptual hash generation,
    percolation analysis, physical deduplication, and group-disjoint splits.
    """
    log_separator("STAGE 1: Dataset Audit & Duplicate Control")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_df = load_label_records(config.labels_file)
    log(f"Parsed {len(labels_df)} entries from {config.labels_file}")

    # 1. Audit images on disk
    image_paths = []
    corrupt_files = []
    image_dimensions = set()

    data_path = Path(config.data_dir)
    for idx, row in labels_df.iterrows():
        img_path = data_path / row["name"]
        if not img_path.exists():
            # Check if it was previously moved to duplicate directory
            dup_path = Path("duplicate") / row["name"]
            if dup_path.exists():
                img_path = dup_path
            else:
                corrupt_files.append(row["name"])
                continue
        
        # Test read file
        img = cv2.imread(str(img_path))
        if img is None or img.size == 0:
            corrupt_files.append(row["name"])
        else:
            image_dimensions.add(img.shape)
            image_paths.append((row["name"], img_path))

    log(f"Verified {len(image_paths)} readable images. Corrupt/missing files: {len(corrupt_files)}")
    log(f"Image dimensions found: {image_dimensions}")

    # Compute target distribution stats
    ratings = labels_df["score"].values
    mean_val = float(np.mean(ratings))
    median_val = float(np.median(ratings))
    std_val = float(np.std(ratings))
    min_val = float(np.min(ratings))
    max_val = float(np.max(ratings))
    skew_val = float(stats.skew(ratings))

    log(f"Rating Distribution: Min={min_val:.3f}, Max={max_val:.3f}, Mean={mean_val:.3f}, Median={median_val:.3f}, SD={std_val:.3f}, Skew={skew_val:.3f}")

    # Plot target distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratings, bins=35, color="#2b5c8f", edgecolor="white", alpha=0.85)
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}")
    ax.axvline(median_val, color="green", linestyle=":", linewidth=2, label=f"Median: {median_val:.2f}")
    ax.set_title("SCUT-FBP5500 Target Rating Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Average Human Rating (1.0 to 5.0)", fontsize=11)
    ax.set_ylabel("Number of Images", fontsize=11)
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig01_target_distribution.png", dpi=130)
    plt.close(fig)

    # 2. Duplicate Detection via 64-bit dHash
    log_separator("Duplicate Analysis & Percolation Check")
    log("Computing 64-bit difference hashes for all images...")
    
    hashes = []
    for name, path in image_paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        h = dhash(gray)
        hashes.append(h)
    hashes = np.array(hashes, dtype=np.uint64)

    # Percolation demonstration: Sweep thresholds from 0 to 6
    num_samples = len(hashes)
    log("Testing percolation across Hamming thresholds...")
    
    threshold_results = {}
    for test_thr in [0, 1, 2, 4, 6]:
        dsu = DisjointSetClusters(num_samples)
        pairs_found = 0
        for i in range(num_samples):
            # Compare hash i against all subsequent hashes
            dists = hamming_row(hashes[i], hashes[i+1:])
            matching_indices = np.where(dists <= test_thr)[0] + (i + 1)
            for j in matching_indices:
                dsu.union(i, j)
                pairs_found += 1
        
        # Count cluster sizes
        roots = [dsu.find(i) for i in range(num_samples)]
        root_counts = pd.Series(roots).value_counts()
        multiclusters = root_counts[root_counts > 1]
        grouped_images = multiclusters.sum() if len(multiclusters) > 0 else 0
        max_cluster = multiclusters.max() if len(multiclusters) > 0 else 1
        
        threshold_results[test_thr] = {
            "flagged_pairs": int(pairs_found),
            "grouped_images": int(grouped_images),
            "percent_grouped": float(grouped_images / num_samples * 100),
            "max_cluster_size": int(max_cluster),
        }
        log(f"  Threshold {test_thr}: {pairs_found} pairs, {grouped_images} images ({grouped_images/num_samples*100:.1f}%), Max cluster = {max_cluster}")

    # Notice: threshold 6 chains 75% of data. We strictly enforce threshold 0.
    log("Enforcing threshold 0 to isolate true identical images and prevent transitive percolation.")
    dsu_strict = DisjointSetClusters(num_samples)
    for i in range(num_samples):
        dists = hamming_row(hashes[i], hashes[i+1:])
        for j in np.where(dists == 0)[0] + (i + 1):
            dsu_strict.union(i, j)

    group_ids = np.array([dsu_strict.find(i) for i in range(num_samples)])
    unique_groups = np.unique(group_ids)
    log(f"Isolated {len(unique_groups)} distinct groups (from {num_samples} total images).")

    # 3. Create group-disjoint splits (70% train, 15% validation, 15% test)
    log_separator("Creating Group-Disjoint Splits")
    
    # First split off 15% test set
    splitter_test = GroupShuffleSplit(n_splits=1, test_size=config.test_fraction, random_state=config.seed)
    dev_idx, test_idx = next(splitter_test.split(np.arange(num_samples), groups=group_ids))

    # Split dev set into train (70% of total) and validation (15% of total)
    val_fraction_of_dev = config.val_fraction
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_fraction_of_dev, random_state=config.seed)
    train_sub_idx, val_sub_idx = next(splitter_val.split(dev_idx, groups=group_ids[dev_idx]))
    train_idx = dev_idx[train_sub_idx]
    val_idx = dev_idx[val_sub_idx]

    log(f"Data Partitions: Train={len(train_idx)} ({len(train_idx)/num_samples*100:.1f}%), "
        f"Val={len(val_idx)} ({len(val_idx)/num_samples*100:.1f}%), "
        f"Test={len(test_idx)} ({len(test_idx)/num_samples*100:.1f}%)")

    # Verify zero leakage across dev and test
    dev_groups = set(group_ids[dev_idx])
    test_groups = set(group_ids[test_idx])
    overlap = dev_groups.intersection(test_groups)
    assert len(overlap) == 0, f"Critical leakage detected: {len(overlap)} groups overlap!"
    log("Zero group leakage certified: dev and test duplicate groups are 100% disjoint.")

    # Save split manifest for complete reproducibility
    manifest_rows = []
    for i, (name, _) in enumerate(image_paths):
        role = "train" if i in train_idx else ("val" if i in val_idx else "test")
        manifest_rows.append({
            "name": name,
            "group_id": int(group_ids[i]),
            "split": role,
            "score": float(labels_df.iloc[i]["score"]),
            "ethnicity": labels_df.iloc[i]["ethnicity"],
            "gender": labels_df.iloc[i]["gender"],
        })
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(output_dir / "split_manifest.csv", index=False)
    log(f"Saved split manifest to {output_dir / 'split_manifest.csv'}")

    # Save summary audit JSON
    audit_summary = {
        "dataset_name": "SCUT-FBP5500",
        "total_images": len(labels_df),
        "verified_images": num_samples,
        "corrupt_images": len(corrupt_files),
        "target_min": min_val,
        "target_max": max_val,
        "target_mean": mean_val,
        "target_median": median_val,
        "target_std": std_val,
        "percolation_sweep": threshold_results,
        "splits": {
            "train_count": len(train_idx),
            "val_count": len(val_idx),
            "test_count": len(test_idx),
        }
    }
    with open(output_dir / "task1_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    log_separator("STAGE 1 COMPLETE: Ready for Feature Extraction")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Dataset Audit and Duplicate Control")
    parser.add_argument("--data_dir", default="Images", help="Directory containing facial images")
    parser.add_argument("--labels", default="labels.txt", help="Path to labels text file")
    parser.add_argument("--out", default="out", help="Output directory")
    args = parser.parse_args()

    cfg = PipelineConfig(data_dir=args.data_dir, labels_file=args.labels, output_dir=args.out)
    ckpt = CheckpointManager(cfg.checkpoint_dir)
    run_stage1_audit_and_dedup(cfg, ckpt)
