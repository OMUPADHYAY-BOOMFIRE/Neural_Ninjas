"""fbp_core.py - shared code: classical (no CNN / no DL) feature extractors,
from-scratch NumPy PCA + MLP, metrics, fold maker.
"""
import time
import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern

# low compression = bigger, colour ; high compression = tiny, grayscale
VARIANTS = {"low_rgb": dict(size=128, rgb=True),
            "high_gray": dict(size=48, rgb=False)}
BLOCKS = ["hog", "lbp", "col", "edge", "geo"]

_CC = {}


def _cc(name):
    """Load a Haar cascade (classical detector). Falls back to downloading the XML
    from the official OpenCV repo if the pip build (e.g. OpenCV 5) does not ship it."""
    if name not in _CC:
        import os, urllib.request
        f = {"face": "haarcascade_frontalface_default.xml",
             "eye": "haarcascade_eye.xml",
             "smile": "haarcascade_smile.xml"}[name]
        path = os.path.join(cv2.data.haarcascades, f) if hasattr(cv2, "data") else f
        if not os.path.exists(path):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cascades", f)
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/" + f, path)
        c = cv2.CascadeClassifier(path)
        if c.empty():
            raise RuntimeError(f"could not load cascade {path}")
        _CC[name] = c
    return _CC[name]


# ---------------------------------------------------------------- hashing
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(gray):
    s = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return np.packbits(bits).view(np.uint64)[0]


def hamming_row(h1, H):
    x = np.bitwise_xor(H, h1)
    return _POP[x.view(np.uint8).reshape(-1, 8)].sum(1)


# ---------------------------------------------------------------- features
def f_hog(gray, cell=8):
    """HOG = histogram of gradient orientations per cell (edge/contrast descriptor)."""
    return hog(gray, orientations=9, pixels_per_cell=(cell, cell),
               cells_per_block=(2, 2), block_norm="L2-Hys").astype(np.float32)


def f_lbp(gray, grid=4, P=8, R=1):
    l = local_binary_pattern(gray, P, R, "uniform").astype(int)
    h, w = gray.shape
    out = []
    for i in range(grid):
        for j in range(grid):
            p = l[i * h // grid:(i + 1) * h // grid, j * w // grid:(j + 1) * w // grid]
            c = np.bincount(p.ravel(), minlength=P + 2).astype(np.float32)
            out.append(c / max(c.sum(), 1))
    return np.concatenate(out)


def f_colour(bgr, rgb):
    if rgb:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hs = [cv2.calcHist([hsv], [c], None, [8], [0, 180 if c == 0 else 256]).ravel()
              for c in range(3)]
    else:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hs = [cv2.calcHist([g], [0], None, [16], [0, 256]).ravel()]
    v = np.concatenate(hs).astype(np.float32)
    return v / max(v.sum(), 1) * len(hs)


def _canny(g):
    med = np.median(g)
    return cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), int(max(0, .66 * med)),
                     int(min(255, 1.33 * med)))


def f_edge(gray, g=8):
    """cv2 edge detectors: Canny density + Sobel magnitude on a g x g grid."""
    e = (_canny(gray) > 0).astype(np.float32)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2) / 255.
    return np.concatenate([cv2.resize(e, (g, g), interpolation=cv2.INTER_AREA).ravel(),
                           cv2.resize(mag, (g, g), interpolation=cv2.INTER_AREA).ravel()]
                          ).astype(np.float32)


def geo_names(rgb):
    n = ["face_w_over_h", "face_area_frac", "face_cx_offset",
         "eye_sep_over_w", "eye_w_over_w", "eye_h_over_w", "eye_aspect_h_over_w",
         "eye_y_over_h", "eye_area_frac", "eye_tilt", "eye_count",
         "mouth_w_over_w", "mouth_h_over_h", "mouth_y_over_h", "teeth_bright_frac"]
    n += [f"edge_zone{i}" for i in range(9)] + [f"int_zone{i}" for i in range(9)]
    n += ["sym_corr", "sym_edge", "jaw_width75", "jaw_width90", "forehead_dark_frac",
          "forehead_over_cheek", "face_found"]
    if rgb:
        n += ["cheek_H", "cheek_S", "cheek_V", "fore_H", "fore_S", "fore_V"]
    return n


def f_geometry(bgr, gray, rgb):
    """Hand-designed ratios from classical detectors only (Haar cascades + Canny)."""
    H, W = gray.shape
    v = []
    faces = _cc("face").detectMultiScale(gray, 1.1, 4, minSize=(max(W // 4, 12), max(H // 4, 12)))
    found = len(faces) > 0
    if found:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    else:
        x, y, w, h = int(.1 * W), int(.05 * H), int(.8 * W), int(.9 * H)
    x, y, w, h = int(x), int(y), int(w), int(h)
    v += [w / h, w * h / (W * H), (x + w / 2) / W - .5]
    # eyes (upper 60 % of face box)
    top = gray[y:y + int(.6 * h), x:x + w]
    ms = max(w // 10, 4)
    eyes = _cc("eye").detectMultiScale(top, 1.1, 4, minSize=(ms, ms)) if top.size else []
    if len(eyes) >= 2:
        e = sorted(sorted(eyes, key=lambda a: -a[2] * a[3])[:2], key=lambda a: a[0])
        cx = [a[0] + a[2] / 2 for a in e]
        cy = [a[1] + a[3] / 2 for a in e]
        ew, eh = np.mean([a[2] for a in e]), np.mean([a[3] for a in e])
        v += [(cx[1] - cx[0]) / w, ew / w, eh / w, eh / ew, np.mean(cy) / h,
              ew * eh / (w * h), abs(cy[1] - cy[0]) / w, len(eyes)]
    else:
        v += [np.nan] * 7 + [len(eyes)]
    # mouth (lower 45 %)
    y0 = int(.55 * h)
    low = gray[y + y0:y + h, x:x + w]
    sm = _cc("smile").detectMultiScale(low, 1.7, 18, minSize=(max(w // 5, 6), max(w // 12, 4))) \
        if low.size else []
    if len(sm):
        sx, sy, sw, sh = max(sm, key=lambda a: a[2] * a[3])
        box = low[sy:sy + sh, sx:sx + sw]
        teeth = float((box > np.percentile(low, 85)).mean())  # bright-pixel proxy for teeth
        v += [sw / w, sh / h, (y0 + sy + sh / 2) / h, teeth]
    else:
        v += [np.nan] * 4
    # canonical 64x64 face crop -> zones, symmetry, jaw, hair
    fc = gray[max(y, 0):y + h, max(x, 0):x + w]
    if fc.size == 0:
        fc = gray
    fc = cv2.resize(fc, (64, 64), interpolation=cv2.INTER_AREA)
    e = _canny(fc) > 0
    b = [0, 21, 43, 64]
    for i in range(3):
        for j in range(3):
            v.append(e[b[i]:b[i + 1], b[j]:b[j + 1]].mean())
    for i in range(3):
        for j in range(3):
            v.append(fc[b[i]:b[i + 1], b[j]:b[j + 1]].mean() / 255.)
    a = fc.astype(float)
    c = np.corrcoef(a.ravel(), a[:, ::-1].ravel())[0, 1]
    v.append(0. if np.isnan(c) else c)
    l_, r_ = e[:, :32].mean(), e[:, 32:].mean()
    v.append(abs(l_ - r_) / (l_ + r_ + 1e-6))
    for r in (48, 58):
        cols = np.where(e[r - 2:r + 2].any(0))[0]
        v.append((cols.max() - cols.min()) / 64. if cols.size else 0.)
    strip = fc[:13]
    thr, _ = cv2.threshold(fc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cheeks = np.concatenate([fc[32:45, 10:22].ravel(), fc[32:45, 42:54].ravel()])
    v += [(strip < thr).mean(), strip.mean() / (cheeks.mean() + 1e-6), float(found)]
    if rgb:
        bc = bgr[max(y, 0):y + h, max(x, 0):x + w]
        bc = cv2.resize(bc if bc.size else bgr, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(bc, cv2.COLOR_BGR2HSV).astype(float) / np.array([180., 255., 255.])
        ch = np.concatenate([hsv[32:45, 10:22].reshape(-1, 3), hsv[32:45, 42:54].reshape(-1, 3)])
        v += list(ch.mean(0)) + list(hsv[:13].reshape(-1, 3).mean(0))
    return np.array(v, dtype=np.float32)


def compute_blocks(img_bgr, cfg):
    s = cfg["size"]
    b = cv2.resize(img_bgr, (s, s), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    feats, times = {}, {}
    for name, fn in [("hog", lambda: f_hog(g)), ("lbp", lambda: f_lbp(g)),
                     ("col", lambda: f_colour(b, cfg["rgb"])), ("edge", lambda: f_edge(g)),
                     ("geo", lambda: f_geometry(b, g, cfg["rgb"]))]:
        t = time.perf_counter()
        feats[name] = fn()
        times[name] = time.perf_counter() - t
    return feats, times


def quality(gray128):
    return np.array([cv2.Laplacian(gray128, cv2.CV_64F).var(), gray128.mean(), gray128.std()],
                    dtype=np.float32)


QUAL_NAMES = ["blur_lapvar", "brightness", "contrast"]


def extract_one(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)       # unreadable / truncated file -> None
    if img is None or img.size == 0:
        return None
    out = {"shape": img.shape[:2]}
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out["raw32"] = (cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).ravel()
                    .astype(np.float32) / 255.)
    out["dhash"] = dhash(g)
    g128 = cv2.resize(g, (128, 128), interpolation=cv2.INTER_AREA)
    out["g128"] = g128
    out["quality"] = quality(g128)
    for vn, cfg in VARIANTS.items():
        out[vn] = compute_blocks(img, cfg)
    return out


# ---------------------------------------------------------------- from-scratch PCA
class NumpyPCA:
    """PCA written from scratch: randomised SVD (Halko et al.) with power iterations."""

    def __init__(self, k=100, oversample=10, n_iter=3, seed=0):
        self.k, self.os, self.n_iter, self.seed = k, oversample, n_iter, seed

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        n, D = X.shape
        self.k_ = k = int(min(self.k, n - 1, D))
        self.mu_ = X.mean(0)
        Xc = X - self.mu_
        rng = np.random.RandomState(self.seed)
        Y = Xc @ rng.randn(D, k + self.os).astype(np.float32)
        for _ in range(self.n_iter):                       # power iteration sharpens spectrum
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


# ---------------------------------------------------------------- from-scratch MLP
class ScratchPCAMLP:
    """impute(median) -> standardise -> own PCA (whitened) -> own 1-hidden-layer MLP
    (ReLU, MSE + L2, manual backprop, Adam, early stopping on a group-aware inner
    validation split). history_ keeps train/val loss per epoch."""

    def __init__(self, k=100, hidden=64, lr=2e-3, wd=1e-2, epochs=200, patience=20,
                 batch=128, seed=0):
        self.k, self.hidden, self.lr, self.wd = k, hidden, lr, wd
        self.epochs, self.patience, self.batch, self.seed = epochs, patience, batch, seed

    # -- preprocessing
    def _prep(self, X):
        X = np.where(np.isnan(X), self.med_, X)
        return (X - self.mu_) / self.sd_

    # -- network
    def _init(self, d, rng):
        h = self.hidden
        self.P = {"W1": rng.normal(0, np.sqrt(2. / d), (d, h)), "b1": np.zeros(h),
                  "W2": rng.normal(0, np.sqrt(1. / h), (h, 1)), "b2": np.zeros(1)}

    @staticmethod
    def _fwd(P, Z):
        h = Z @ P["W1"] + P["b1"]
        a = np.maximum(h, 0)
        return h, a, (a @ P["W2"] + P["b2"]).ravel()

    def _grads(self, P, Z, y):
        n = len(y)
        h, a, o = self._fwd(P, Z)
        d = ((o - y) * (2. / n))[:, None]
        dh = (d @ P["W2"].T) * (h > 0)
        return {"W2": a.T @ d + self.wd * P["W2"], "b2": d.sum(0),
                "W1": Z.T @ dh + self.wd * P["W1"], "b1": dh.sum(0)}

    def _loss(self, P, Z, y):
        return float(np.mean((self._fwd(P, Z)[2] - y) ** 2))

    def fit(self, X, y, groups=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        with np.errstate(all="ignore"):
            med = np.nanmedian(X, 0)
        self.med_ = np.where(np.isnan(med), 0., med)
        X = np.where(np.isnan(X), self.med_, X)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-8
        Xs = (X - self.mu_) / self.sd_
        self.pca_ = NumpyPCA(self.k, seed=self.seed).fit(Xs)
        Z = self.pca_.transform(Xs, whiten=True).astype(np.float64)
        self.ym_, self.ys_ = y.mean(), y.std() + 1e-8
        t = (y - self.ym_) / self.ys_
        rng = np.random.RandomState(self.seed)
        if groups is None:
            groups = np.arange(len(y))
        ug = np.unique(groups)
        rng.shuffle(ug)
        vg = set(ug[:max(1, int(.1 * len(ug)))])
        vm = np.array([g in vg for g in groups])
        Ztr, ttr, Zva, tva = Z[~vm], t[~vm], Z[vm], t[vm]
        self._init(Z.shape[1], rng)
        m = {k: np.zeros_like(v) for k, v in self.P.items()}
        s = {k: np.zeros_like(v) for k, v in self.P.items()}
        best, best_P, bad, step = np.inf, None, 0, 0
        self.history_ = {"train": [], "val": []}
        for ep in range(self.epochs):
            perm = rng.permutation(len(ttr))
            for i in range(0, len(perm), self.batch):
                idx = perm[i:i + self.batch]
                gr = self._grads(self.P, Ztr[idx], ttr[idx])
                step += 1
                for k in self.P:                                   # Adam
                    m[k] = .9 * m[k] + .1 * gr[k]
                    s[k] = .999 * s[k] + .001 * gr[k] ** 2
                    mh, sh = m[k] / (1 - .9 ** step), s[k] / (1 - .999 ** step)
                    self.P[k] -= self.lr * mh / (np.sqrt(sh) + 1e-8)
            tl, vl = self._loss(self.P, Ztr, ttr), self._loss(self.P, Zva, tva)
            self.history_["train"].append(tl)
            self.history_["val"].append(vl)
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_P = {k: v.copy() for k, v in self.P.items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.P = best_P
        self.best_epoch_ = int(np.argmin(self.history_["val"]))
        return self

    def predict(self, X):
        Z = self.pca_.transform(self._prep(np.asarray(X, dtype=np.float64)), whiten=True)
        return self._fwd(self.P, Z.astype(np.float64))[2] * self.ys_ + self.ym_


def grad_check(seed=0):
    """Numerical vs analytic gradient: returns max relative error (should be < 1e-5)."""
    rng = np.random.RandomState(seed)
    mdl = ScratchPCAMLP(hidden=5, wd=.01)
    mdl._init(4, rng)
    Z, y = rng.randn(20, 4), rng.randn(20)
    # make loss include the L2 term so it matches the analytic gradient
    def L(P):
        return mdl._loss(P, Z, y) + .5 * mdl.wd * (np.sum(P["W1"] ** 2) + np.sum(P["W2"] ** 2))
    g = mdl._grads(mdl.P, Z, y)
    worst = 0
    for k in mdl.P:
        for idx in np.ndindex(*mdl.P[k].shape):
            P1 = {a: b.copy() for a, b in mdl.P.items()}
            P2 = {a: b.copy() for a, b in mdl.P.items()}
            P1[k][idx] += 1e-6
            P2[k][idx] -= 1e-6
            num = (L(P1) - L(P2)) / 2e-6
            worst = max(worst, abs(num - g[k][idx]) / (abs(num) + abs(g[k][idx]) + 1e-12))
    return worst


# ---------------------------------------------------------------- evaluation utils
def metrics(y, p):
    e = p - y
    pr = 0. if np.std(p) < 1e-12 else float(np.corrcoef(y, p)[0, 1])
    from scipy.stats import spearmanr
    sp = 0. if np.std(p) < 1e-12 else float(spearmanr(y, p)[0])
    ss = ((y - y.mean()) ** 2).sum()
    return {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
            "MedAE": float(np.median(np.abs(e))), "Pearson": pr, "Spearman": sp,
            "R2": float(1 - (e ** 2).sum() / ss)}


def make_folds(dev_idx, gid, K, R, seed):
    """R repeats of group-aware K-fold: all images of a duplicate-group stay together."""
    out = []
    for r in range(R):
        rng = np.random.RandomState(seed + r)
        ug = np.unique(gid[dev_idx])
        rng.shuffle(ug)
        parts = np.array_split(ug, K)
        rep = []
        for p in parts:
            m = np.isin(gid[dev_idx], p)
            rep.append((dev_idx[~m], dev_idx[m]))
        out.append(rep)
    return out
