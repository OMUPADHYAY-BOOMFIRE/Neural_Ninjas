#!/usr/bin/env python
"""config.py - Central configuration for the Neural Ninjas pipeline.

All tunables, paths, hardware settings, and checkpoint logic live here.
Optimised for local execution on RTX 5050 + Radeon 780M + system RAM.
"""
import os
import sys
import json
import time
import pickle
import platform
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


# ================================================================
# Hardware-aware defaults
# ================================================================
def detect_gpu():
    """Detect available GPU for acceleration (NVIDIA RTX 5050 / Radeon 780M)."""
    info = {
        "cuda": False,
        "device_name": None,
        "gpu_memory": None,
        "opencl": False,
        "cpu_count": os.cpu_count() or 4,
    }

    # 1. Probe PyTorch CUDA if available
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda"] = True
            info["device_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    except Exception:
        pass

    # 2. Probe system nvidia-smi directly (RTX 5050 detection)
    if not info["cuda"]:
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                text=True, stderr=subprocess.DEVNULL
            ).strip()
            if out:
                parts = out.split(",")
                info["cuda"] = True
                info["device_name"] = parts[0].strip()
                info["gpu_memory"] = parts[1].strip() if len(parts) > 1 else "8151 MiB"
        except Exception:
            pass

    # 3. Probe OpenCV CUDA
    if not info["cuda"]:
        try:
            import cv2
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                info["cuda"] = True
                info["device_name"] = "OpenCV CUDA Device"
        except Exception:
            pass

    # 4. Probe AMD Radeon / OpenCL acceleration
    try:
        import cv2
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            info["opencl"] = True
    except Exception:
        pass

    return info


GPU_INFO = detect_gpu()

# RTX 5050: 8 GB VRAM - OpenCV can use CUDA for resize/filter if available.
# Radeon 780M: integrated GPU with OpenCL - cv2.ocl can use it.
# Combined system RAM: process images in batches to stay under ~12 GB.


@dataclass
class PipelineConfig:
    """All pipeline parameters in one place."""

    # -- paths
    data_dir: str = "Images"
    labels_file: str = "labels.txt"
    ratings_file: Optional[str] = None
    output_dir: str = "out"
    checkpoint_dir: str = "checkpoints"

    # -- run mode
    seed: int = 42
    limit: int = 0           # 0 = all images
    fast: bool = False       # True = smoke test

    # -- duplicate detection
    dup_threshold: int = 6   # max Hamming distance (of 64)

    # -- target
    target_type: str = "mean"  # mean | median | trim

    # -- split
    test_fraction: float = 0.15
    val_fraction: float = 0.176   # of dev set => ~15% of total
    cv_folds: int = 5
    cv_repeats: int = 3

    # -- hardware
    n_jobs: int = -1          # joblib parallelism (-1 = all cores)
    batch_size: int = 32      # images per joblib batch
    max_ram_gb: float = 12.0  # stay under this
    use_cuda: bool = True     # Use NVIDIA RTX 5050 CUDA GPU if present

    # -- high-rated threshold
    hi_threshold: Optional[float] = None  # None => auto (75th pct)

    # -- P2 imputation
    p2_impute: str = "auto"  # auto | mean | median

    # -- feature variants
    low_rgb_size: int = 128
    high_gray_size: int = 48

    # -- from-scratch MLP
    mlp_hidden: int = 64
    mlp_lr: float = 2e-3
    mlp_wd: float = 1e-2
    mlp_epochs: int = 200
    mlp_patience: int = 20
    mlp_batch: int = 128

    # -- PCA
    pca_components: int = 100

    def __post_init__(self):
        if self.fast:
            self.cv_folds = 3
            self.cv_repeats = 1
            if self.limit == 0:
                self.limit = 500

    def to_json(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls(**json.load(f))

    @classmethod
    def from_args(cls):
        """Build config from command-line arguments."""
        import argparse
        ap = argparse.ArgumentParser(
            description="Neural Ninjas: face-rating regression (classical ML)")
        ap.add_argument("--data_dir", default="Images")
        ap.add_argument("--labels", default="labels.txt")
        ap.add_argument("--ratings", default=None)
        ap.add_argument("--out", default="out")
        ap.add_argument("--limit", type=int, default=0)
        ap.add_argument("--fast", action="store_true")
        ap.add_argument("--seed", type=int, default=42)
        ap.add_argument("--dup_thr", type=int, default=6)
        ap.add_argument("--hi_thr", type=float, default=None)
        ap.add_argument("--target", default="mean",
                        choices=["mean", "median", "trim"])
        ap.add_argument("--jobs", type=int, default=-1)
        ap.add_argument("--batch_size", type=int, default=32)
        ap.add_argument("--cuda", action="store_true", default=True,
                        help="Enable NVIDIA CUDA acceleration (default: True)")
        ap.add_argument("--no_cuda", action="store_false", dest="cuda",
                        help="Disable CUDA acceleration")
        ap.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
        a = ap.parse_args()
        return cls(
            data_dir=a.data_dir, labels_file=a.labels,
            ratings_file=a.ratings, output_dir=a.out,
            limit=a.limit, fast=a.fast, seed=a.seed,
            dup_threshold=a.dup_thr, hi_threshold=a.hi_thr,
            target_type=a.target, n_jobs=a.jobs,
            batch_size=a.batch_size, use_cuda=a.cuda,
        ), a.resume


# ================================================================
# Checkpoint / progress-saving utilities
# ================================================================
class CheckpointManager:
    """Save and restore intermediate pipeline state.

    Every stage writes its outputs to disk so that a crash or
    termination lets you resume from the last completed stage.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.dir / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self):
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {"completed_stages": [], "stage_times": {}}

    def _save_manifest(self):
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def is_done(self, stage_name: str) -> bool:
        return stage_name in self.manifest["completed_stages"]

    def mark_done(self, stage_name: str, elapsed: float = 0.0):
        if stage_name not in self.manifest["completed_stages"]:
            self.manifest["completed_stages"].append(stage_name)
        self.manifest["stage_times"][stage_name] = round(elapsed, 2)
        self._save_manifest()

    def save(self, stage_name: str, data: dict):
        """Pickle a dict of arrays / objects for one stage."""
        path = self.dir / f"{stage_name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, stage_name: str) -> dict:
        path = self.dir / f"{stage_name}.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)

    def save_numpy(self, name: str, arr):
        """Save a single numpy array (memory-mapped friendly)."""
        np.save(self.dir / f"{name}.npy", arr)

    def load_numpy(self, name: str):
        return np.load(self.dir / f"{name}.npy", allow_pickle=True)

    def reset(self):
        import shutil
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest = {"completed_stages": [], "stage_times": {}}
        self._save_manifest()


# ================================================================
# Logging
# ================================================================
_T0 = time.time()


def log(*args, **kwargs):
    """Timestamped print that flushes immediately."""
    elapsed = time.time() - _T0
    print(f"[{elapsed:8.1f}s]", *args, flush=True, **kwargs)


def log_separator(title: str):
    log("=" * 60)
    log(f"  {title}")
    log("=" * 60)
