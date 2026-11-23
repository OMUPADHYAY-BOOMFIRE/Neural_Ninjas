#!/usr/bin/env python
"""features.py - All feature extraction code (classical CV only, no CNN).

Feature blocks:
  1. HOG  - histogram of oriented gradients (edge/contrast)
  2. LBP  - local binary patterns (micro-texture)
  3. COL  - colour histograms (HSV or grayscale intensity)
  4. EDGE - Canny edge density + Sobel gradient magnitude on a grid
  5. GEO  - hand-designed geometry ratios from Haar cascades + Canny

Two compression variants run in parallel per image:
  - low_rgb:   128x128 RGB/HSV (more detail, higher cost)
  - high_gray:  48x48 grayscale (fast, small, loses detail)

Run standalone to test on a single image:
    python features.py path/to/image.jpg
"""
import os
import time
import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern


# ================================================================
# Constants
# ================================================================
VARIANTS = {
    "low_rgb":   {"size": 128, "rgb": True},
    "high_gray": {"size": 48,  "rgb": False},
}

BLOCKS = ["hog", "lbp", "col", "edge", "geo"]

QUAL_NAMES = ["blur_lapvar", "brightness", "contrast"]


# ================================================================
# Haar Cascade Loader (OpenCV 4.x compatible)
# ================================================================
_CC = {}


def _cc(name):
    """Load a Haar cascade (classical face/eye/smile detector).

    Falls back to the official OpenCV repo XML if the local
    installation does not ship cascade files (e.g. OpenCV 5 build).
    """
    if name not in _CC:
        import urllib.request

        cascade_map = {
            "face":  "haarcascade_frontalface_default.xml",
            "eye":   "haarcascade_eye.xml",
            "smile": "haarcascade_smile.xml",
        }
        f = cascade_map[name]

        path = (
            os.path.join(cv2.data.haarcascades, f)
            if hasattr(cv2, "data")
            else f
        )

        if not os.path.exists(path):
            fallback_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "cascades"
            )
            path = os.path.join(fallback_dir, f)
            if not os.path.exists(path):
                os.makedirs(fallback_dir, exist_ok=True)
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/opencv/"
                    "opencv/4.x/data/haarcascades/" + f,
                    path,
                )

        c = cv2.CascadeClassifier(path)
        if c.empty():
            raise RuntimeError(f"Could not load cascade {path}")
        _CC[name] = c

    return _CC[name]


# ================================================================
# Perceptual Hashing (compressed comparison)
# ================================================================
_POP = np.array(
    [bin(i).count("1") for i in range(256)], dtype=np.uint8
)


def dhash(gray):
    """Compute a 64-bit difference hash from a grayscale image.

    The image is COMPRESSED to 9x8 first, so the hash compares
    64 adjacent-pixel differences, NOT the full pixel matrix.
    This makes O(N^2) duplicate detection feasible: each comparison
    is a single 64-bit XOR + popcount, not an n^2 pixel scan.
    """
    s = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return np.packbits(bits).view(np.uint64)[0]


def hamming_row(h1, H):
    """Hamming distance between one hash and an array of hashes.

    Uses byte-level lookup table: 8 bytes * 256 entries = fast.
    """
    x = np.bitwise_xor(H, h1)
    return _POP[x.view(np.uint8).reshape(-1, 8)].sum(1)


# ================================================================
# Feature Block 1: HOG
# ================================================================
def f_hog(gray, cell=8):
    """Histogram of Oriented Gradients.

    Captures edge directions and contrast in small cells.
    At 128px with 8px cells: ~8100 features.
    At  48px with 8px cells:  ~900 features.
    """
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(cell, cell),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    ).astype(np.float32)


# ================================================================
# Feature Block 2: LBP
# ================================================================
def f_lbp(gray, grid=4, P=8, R=1):
    """Local Binary Patterns on a grid (micro-texture descriptor)."""
    lbp_img = local_binary_pattern(gray, P, R, "uniform").astype(int)
    h, w = gray.shape
    out = []
    for i in range(grid):
        for j in range(grid):
            patch = lbp_img[
                i * h // grid : (i + 1) * h // grid,
                j * w // grid : (j + 1) * w // grid,
            ]
            c = np.bincount(patch.ravel(), minlength=P + 2).astype(
                np.float32
            )
            out.append(c / max(c.sum(), 1))
    return np.concatenate(out)


# ================================================================
# Feature Block 3: Colour Histograms
# ================================================================
def f_colour(bgr, rgb):
    """HSV colour histograms (RGB variant) or intensity (gray)."""
    if rgb:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hs = [
            cv2.calcHist(
                [hsv], [c], None,
                [8], [0, 180 if c == 0 else 256]
            ).ravel()
            for c in range(3)
        ]
    else:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hs = [cv2.calcHist([g], [0], None, [16], [0, 256]).ravel()]

    v = np.concatenate(hs).astype(np.float32)
    return v / max(v.sum(), 1) * len(hs)


# ================================================================
# Feature Block 4: Edge Grid (Canny + Sobel)
# ================================================================
def _canny(g):
    med = np.median(g)
    return cv2.Canny(
        cv2.GaussianBlur(g, (3, 3), 0),
        int(max(0, 0.66 * med)),
        int(min(255, 1.33 * med)),
    )


def f_edge(gray, g=8):
    """Canny edge density + Sobel gradient magnitude on a g x g grid."""
    e = (_canny(gray) > 0).astype(np.float32)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2) / 255.0
    return np.concatenate([
        cv2.resize(e, (g, g), interpolation=cv2.INTER_AREA).ravel(),
        cv2.resize(mag, (g, g), interpolation=cv2.INTER_AREA).ravel(),
    ]).astype(np.float32)


# ================================================================
# Feature Block 5: Geometry Ratios (Haar + Canny)
# ================================================================
def geo_names(rgb):
    """Feature names for the geometry block."""
    n = [
        "face_w_over_h", "face_area_frac", "face_cx_offset",
        "eye_sep_over_w", "eye_w_over_w", "eye_h_over_w",
        "eye_aspect_h_over_w", "eye_y_over_h", "eye_area_frac",
        "eye_tilt", "eye_count",
        "mouth_w_over_w", "mouth_h_over_h", "mouth_y_over_h",
        "teeth_bright_frac",
    ]
    n += [f"edge_zone{i}" for i in range(9)]
    n += [f"int_zone{i}" for i in range(9)]
    n += [
        "sym_corr", "sym_edge",
        "jaw_width75", "jaw_width90",
        "forehead_dark_frac", "forehead_over_cheek",
        "face_found",
    ]
    if rgb:
        n += ["cheek_H", "cheek_S", "cheek_V",
              "fore_H", "fore_S", "fore_V"]
    return n


def f_geometry(bgr, gray, rgb):
    """Hand-designed ratios from Haar cascades + Canny edges.

    Detects face, eyes, mouth; computes proportions, symmetry,
    zone textures, jaw width, forehead/cheek brightness.
    Missing detections => NaN (imputed later in the pipeline).
    """
    H, W = gray.shape
    v = []

    faces = _cc("face").detectMultiScale(
        gray, 1.1, 4, minSize=(max(W // 4, 12), max(H // 4, 12))
    )
    found = len(faces) > 0
    if found:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    else:
        x, y, w, h = int(0.1 * W), int(0.05 * H), int(0.8 * W), int(0.9 * H)

    x, y, w, h = int(x), int(y), int(w), int(h)
    v += [w / h, w * h / (W * H), (x + w / 2) / W - 0.5]

    # Eyes (upper 60% of face box)
    top = gray[y : y + int(0.6 * h), x : x + w]
    ms = max(w // 10, 4)
    eyes = (
        _cc("eye").detectMultiScale(top, 1.1, 4, minSize=(ms, ms))
        if top.size
        else []
    )

    if len(eyes) >= 2:
        e = sorted(
            sorted(eyes, key=lambda a: -a[2] * a[3])[:2],
            key=lambda a: a[0],
        )
        cx = [a[0] + a[2] / 2 for a in e]
        cy = [a[1] + a[3] / 2 for a in e]
        ew = np.mean([a[2] for a in e])
        eh = np.mean([a[3] for a in e])
        v += [
            (cx[1] - cx[0]) / w, ew / w, eh / w, eh / ew,
            np.mean(cy) / h, ew * eh / (w * h),
            abs(cy[1] - cy[0]) / w, len(eyes),
        ]
    else:
        v += [np.nan] * 7 + [len(eyes)]

    # Mouth (lower 45%)
    y0 = int(0.55 * h)
    low = gray[y + y0 : y + h, x : x + w]
    sm = (
        _cc("smile").detectMultiScale(
            low, 1.7, 18, minSize=(max(w // 5, 6), max(w // 12, 4))
        )
        if low.size
        else []
    )

    if len(sm):
        sx, sy, sw, sh = max(sm, key=lambda a: a[2] * a[3])
        box = low[sy : sy + sh, sx : sx + sw]
        teeth = float((box > np.percentile(low, 85)).mean())
        v += [sw / w, sh / h, (y0 + sy + sh / 2) / h, teeth]
    else:
        v += [np.nan] * 4

    # Canonical 64x64 face crop -> zones, symmetry, jaw, hair
    fc = gray[max(y, 0) : y + h, max(x, 0) : x + w]
    if fc.size == 0:
        fc = gray
    fc = cv2.resize(fc, (64, 64), interpolation=cv2.INTER_AREA)
    e = _canny(fc) > 0

    b = [0, 21, 43, 64]
    for i in range(3):
        for j in range(3):
            v.append(e[b[i] : b[i + 1], b[j] : b[j + 1]].mean())
    for i in range(3):
        for j in range(3):
            v.append(fc[b[i] : b[i + 1], b[j] : b[j + 1]].mean() / 255.0)

    a = fc.astype(float)
    c = np.corrcoef(a.ravel(), a[:, ::-1].ravel())[0, 1]
    v.append(0.0 if np.isnan(c) else c)

    l_, r_ = e[:, :32].mean(), e[:, 32:].mean()
    v.append(abs(l_ - r_) / (l_ + r_ + 1e-6))

    for r in (48, 58):
        cols = np.where(e[r - 2 : r + 2].any(0))[0]
        v.append((cols.max() - cols.min()) / 64.0 if cols.size else 0.0)

    strip = fc[:13]
    thr, _ = cv2.threshold(fc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cheeks = np.concatenate(
        [fc[32:45, 10:22].ravel(), fc[32:45, 42:54].ravel()]
    )
    v += [
        (strip < thr).mean(),
        strip.mean() / (cheeks.mean() + 1e-6),
        float(found),
    ]

    if rgb:
        bc = bgr[max(y, 0) : y + h, max(x, 0) : x + w]
        bc = cv2.resize(
            bc if bc.size else bgr,
            (64, 64), interpolation=cv2.INTER_AREA,
        )
        hsv = cv2.cvtColor(bc, cv2.COLOR_BGR2HSV).astype(float) / np.array(
            [180.0, 255.0, 255.0]
        )
        ch = np.concatenate([
            hsv[32:45, 10:22].reshape(-1, 3),
            hsv[32:45, 42:54].reshape(-1, 3),
        ])
        v += list(ch.mean(0)) + list(hsv[:13].reshape(-1, 3).mean(0))

    return np.array(v, dtype=np.float32)


# ================================================================
# Full extraction pipeline for one image
# ================================================================
def compute_blocks(img_bgr, cfg):
    """Extract all 5 feature blocks for one compression variant."""
    s = cfg["size"]
    b = cv2.resize(img_bgr, (s, s), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    feats, times = {}, {}
    for name, fn in [
        ("hog", lambda: f_hog(g)),
        ("lbp", lambda: f_lbp(g)),
        ("col", lambda: f_colour(b, cfg["rgb"])),
        ("edge", lambda: f_edge(g)),
        ("geo", lambda: f_geometry(b, g, cfg["rgb"])),
    ]:
        t = time.perf_counter()
        feats[name] = fn()
        times[name] = time.perf_counter() - t
    return feats, times


def quality(gray128):
    """Image quality metrics: blur (Laplacian var), brightness, contrast."""
    return np.array(
        [
            cv2.Laplacian(gray128, cv2.CV_64F).var(),
            gray128.mean(),
            gray128.std(),
        ],
        dtype=np.float32,
    )


def extract_one(path):
    """Extract all features from one image file.

    Returns None if the file is unreadable or corrupt.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None

    out = {"shape": img.shape[:2]}
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Raw 32x32 for baselines and PCA
    out["raw32"] = (
        cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA)
        .ravel()
        .astype(np.float32) / 255.0
    )
    out["dhash"] = dhash(g)

    # 128x128 grayscale for quality metrics and gallery
    g128 = cv2.resize(g, (128, 128), interpolation=cv2.INTER_AREA)
    out["g128"] = g128
    out["quality"] = quality(g128)

    # Both compression variants
    for vn, cfg in VARIANTS.items():
        out[vn] = compute_blocks(img, cfg)

    return out


# ================================================================
# From-scratch PCA (randomised SVD, Halko et al.)
# ================================================================
class NumpyPCA:
    """PCA from scratch: randomised SVD with power iterations.

    Not using sklearn on purpose - this is the 'substantial
    original code' required by the brief.
    """

    def __init__(self, k=100, oversample=10, n_iter=3, seed=0):
        self.k = k
        self.os = oversample
        self.n_iter = n_iter
        self.seed = seed

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        n, D = X.shape
        self.k_ = k = int(min(self.k, n - 1, D))
        self.mu_ = X.mean(0)
        Xc = X - self.mu_

        rng = np.random.RandomState(self.seed)
        Y = Xc @ rng.randn(D, k + self.os).astype(np.float32)

        # Power iteration sharpens spectrum
        for _ in range(self.n_iter):
            Q, _ = np.linalg.qr(Y)
            Z, _ = np.linalg.qr(Xc.T @ Q)
            Y = Xc @ Z

        Q, _ = np.linalg.qr(Y)
        _, s, Vt = np.linalg.svd(Q.T @ Xc, full_matrices=False)

        self.components_ = Vt[:k]
        self.var_ = (s[:k] ** 2) / (n - 1)
        self.total_var_ = float((Xc ** 2).sum() / (n - 1))
        self.evr_ = self.var_ / self.total_var_
        return self

    def transform(self, X, whiten=False):
        Z = (np.asarray(X, dtype=np.float32) - self.mu_) @ self.components_.T
        return Z / np.sqrt(self.var_ + 1e-8) if whiten else Z


# ================================================================
# CLI: test feature extraction on a single image
# ================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python features.py <image_path>")
        sys.exit(1)

    result = extract_one(sys.argv[1])
    if result is None:
        print("ERROR: Could not read image")
        sys.exit(1)

    print(f"Image shape: {result['shape']}")
    print(f"dHash: {result['dhash']:#018x}")
    print(f"Quality: blur={result['quality'][0]:.1f}, "
          f"bright={result['quality'][1]:.1f}, "
          f"contrast={result['quality'][2]:.1f}")

    for vn in VARIANTS:
        feats, times = result[vn]
        print(f"\n--- {vn} ---")
        for b in BLOCKS:
            print(f"  {b:5s}: dim={feats[b].shape[0]:6d}  "
                  f"time={times[b]*1000:.1f}ms  "
                  f"NaN={np.isnan(feats[b]).mean():.1%}")
