"""Zero-shot (training-free) frame-importance scorer on CLIP features.

No importance labels are used anywhere: the licensed-for-use importance datasets were not
available (see README), and Samsung's FPVSum annotations are evaluation-only.

score_t = smooth( 0.5 * z(relevance_t) + 0.25 * z(novelty_t) + 0.25 * z(dynamics_t) ), min-max to [0,1]
  relevance_t = cos(x_t, highlight-prompt embedding of the predicted category) - cos(x_t, "boring" prompts)
  novelty_t   = 1 - cos(x_t, video-mean feature)                     (distinct content)
  dynamics_t  = mean over a +-2 s window of 1 - cos(x_t, x_{t-1})     (visual change / action)
Weights and smoothing were fixed a priori (not tuned on any labelled data).
"""
import numpy as np

DEFAULT_WEIGHTS = {"relevance": 0.5, "novelty": 0.25, "dynamics": 0.25}


def _z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-8 else np.zeros_like(x)


def _box(x, w):
    if w <= 1 or len(x) < 2:
        return x
    k = np.ones(w) / w
    pad = w // 2
    return np.convolve(np.pad(x, (pad, w - 1 - pad), mode="edge"), k, mode="valid")


def _gauss(x, sigma):
    if sigma <= 0 or len(x) < 2:
        return x
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, (r, r), mode="edge"), k, mode="valid")


def importance_scores(feats, pos_emb, neg_emb, fps=2.0, weights=None, smooth_sec=1.0, dyn_window_sec=4.0):
    """feats [T,768] L2-normalised; pos_emb [768]; neg_emb [768]. Returns (scores [T] in [0,1], components)."""
    X = np.asarray(feats, dtype=np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-8)
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    rel = X @ pos_emb - X @ neg_emb
    mu = X.mean(0)
    mu /= max(np.linalg.norm(mu), 1e-8)
    nov = 1.0 - X @ mu
    step = np.r_[0.0, 1.0 - np.sum(X[1:] * X[:-1], axis=1)]
    if len(step) > 1:
        step[0] = step[1]
    dyn = _box(step, max(1, int(round(dyn_window_sec * fps))))
    s = w["relevance"] * _z(rel) + w["novelty"] * _z(nov) + w["dynamics"] * _z(dyn)
    s = _gauss(s, smooth_sec * fps)
    lo, hi = s.min(), s.max()
    s = (s - lo) / (hi - lo) if hi > lo else np.full_like(s, 0.5)
    return s.astype(np.float32), {"relevance": rel, "novelty": nov, "dynamics": dyn}
