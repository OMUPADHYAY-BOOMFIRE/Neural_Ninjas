#!/usr/bin/env python
"""
download_dataset.py
================================================================================
Neural Ninjas (AIP301 Machine Learning Lab, IIIT Kota)

Automatic Dataset Downloader & Directory Structuring Script
--------------------------------------------------------------------------------
Downloads the SCUT-FBP5500 v2 benchmark dataset from Kaggle using kagglehub:
Dataset: pranavchandane/scut-fbp5500-v2-facial-beauty-scores
URL: https://www.kaggle.com/datasets/pranavchandane/scut-fbp5500-v2-facial-beauty-scores

The script:
1. Downloads the latest version of the dataset using kagglehub.
2. Locates the nested 'Images/Images' or 'Images' directory inside the download cache.
3. Automatically creates the local 'Images/' directory if it doesn't exist.
4. Moves/copies all facial images directly into './Images/' (e.g. Images/AF1.jpg)
   so the entire pipeline runs seamlessly without manual path configuration.
5. Verifies and copies labels.txt / All_Ratings.xlsx if not already present.

Usage:
    python download_dataset.py
"""

import os
import sys
import shutil
from pathlib import Path


def download_and_setup_images():
    print("=" * 70)
    print("  NEURAL NINJAS — SCUT-FBP5500 DATASET DOWNLOADER")
    print("=" * 70)

    try:
        import kagglehub
    except ImportError:
        print("\n[!] 'kagglehub' is not installed.")
        print("    Installing kagglehub now via pip...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        import kagglehub

    dataset_slug = "pranavchandane/scut-fbp5500-v2-facial-beauty-scores"
    print(f"\n[1/3] Downloading dataset from Kaggle: {dataset_slug}")
    print("      (This may take a couple of minutes depending on network speed)...")

    try:
        download_cache_path = kagglehub.dataset_download(dataset_slug)
        print(f"[✓] Download completed!")
        print(f"    Path to downloaded files: {download_cache_path}")
    except Exception as exc:
        print(f"\n[X] Error downloading dataset via kagglehub: {exc}")
        print("    Please ensure you have an active internet connection or Kaggle API configured.")
        sys.exit(1)

    cache_dir = Path(download_cache_path)
    target_images_dir = Path("Images")
    target_images_dir.mkdir(parents=True, exist_ok=True)

    # In Kaggle dataset, images are often nested under 'Images/Images'
    # Check possible candidate subdirectories
    candidate_image_folders = [
        cache_dir / "Images" / "Images",
        cache_dir / "Images",
        cache_dir,
    ]

    source_dir = None
    for folder in candidate_image_folders:
        if folder.exists() and any(folder.glob("*.jpg")):
            source_dir = folder
            break

    # Fallback recursive search for .jpg files
    if source_dir is None:
        jpg_samples = list(cache_dir.rglob("*.jpg"))
        if jpg_samples:
            source_dir = jpg_samples[0].parent

    if source_dir is None:
        print(f"\n[X] ERROR: Could not find any .jpg image files in: {cache_dir}")
        sys.exit(1)

    print(f"\n[2/3] Found image files in: {source_dir}")
    source_images = list(source_dir.glob("*.jpg"))
    print(f"      Total images found: {len(source_images)}")
    print(f"      Delivering images directly into: {target_images_dir.resolve()} ...")

    copied_count = 0
    already_existing_count = 0

    for img_path in source_images:
        dest_file = target_images_dir / img_path.name
        if not dest_file.exists():
            shutil.copy2(img_path, dest_file)
            copied_count += 1
        else:
            already_existing_count += 1

    total_images_in_target = len(list(target_images_dir.glob("*.jpg")))
    print(f"[✓] Delivered {copied_count} new images (existing: {already_existing_count}).")
    print(f"    Total images in local Images/ directory: {total_images_in_target}")

    # Copy labels.txt or All_Ratings.xlsx to root if present in download
    print("\n[3/3] Checking dataset label and annotation files...")
    for label_filename in ["labels.txt", "All_Ratings.xlsx", "All_Ratings.csv"]:
        target_label_file = Path(label_filename)
        if not target_label_file.exists():
            matches = list(cache_dir.rglob(label_filename))
            if matches:
                shutil.copy2(matches[0], target_label_file)
                print(f"[✓] Copied {label_filename} to project root.")

    print("\n" + "=" * 70)
    print("  DATASET SETUP COMPLETE! You can now run:")
    print("    python 01_dataset_audit_and_dedup.py")
    print("    python 02_extract_features.py")
    print("    python 03_train_models_and_cv.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    download_and_setup_images()
