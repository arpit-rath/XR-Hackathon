"""Timestamp alignment between per-frame annotations and the fixed-rate feature timeline.

Annotation frame i of N (made on a copy whose frame rate may differ from ours) is placed at
    t_i = (i + 0.5) / N * duration
i.e. annotations are assumed to span the whole video uniformly. This is robust to frame-rate
differences between the annotators' copy and ours, provided the durations agree (verified for
every FPVSum video in reports/annotation_alignment.csv: |duration diff| <= 0.15 s).

Feature step k sits at t_k = k / fps. A step's target is the mean of the annotation frames that
fall in [t_k - 0.5/fps, t_k + 0.5/fps); empty bins are linearly interpolated.
"""
import numpy as np


def frame_scores_to_timeline(scores, duration, n_steps, fps=2.0):
    s = np.asarray(scores, dtype=np.float64).ravel()
    n = len(s)
    t = (np.arange(n) + 0.5) / n * duration
    k = np.clip(np.floor(t * fps + 0.5).astype(int), 0, n_steps - 1)
    tot = np.bincount(k, weights=s, minlength=n_steps)
    cnt = np.bincount(k, minlength=n_steps)
    out = np.full(n_steps, np.nan)
    m = cnt > 0
    out[m] = tot[m] / cnt[m]
    if not m.all():
        idx = np.arange(n_steps)
        out[~m] = np.interp(idx[~m], idx[m], out[m])
    return out


def minmax(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def top_fraction_mask(x, frac=0.15):
    x = np.asarray(x)
    k = max(1, int(round(frac * len(x))))
    order = np.argsort(-x, kind="stable")
    m = np.zeros(len(x), bool)
    m[order[:k]] = True
    return m


def f1_top_fraction(pred, gt, frac=0.15):
    """Frame-level F1 between the top-`frac` steps of pred and of gt (Samsung-style highlight F1 proxy)."""
    p, g = top_fraction_mask(pred, frac), top_fraction_mask(gt, frac)
    tp = (p & g).sum()
    if tp == 0:
        return 0.0
    prec, rec = tp / p.sum(), tp / g.sum()
    return float(2 * prec * rec / (prec + rec))


def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b).correlation
    return float(0.0 if np.isnan(r) else r)
