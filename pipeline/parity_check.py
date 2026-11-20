"""Parity check: do our Encoder A features reproduce the released ActionFormer EPIC features?

Compares OURS.npz (from encoder_a.py / run_manifest.py, key 'feats') with the released .npz for the
same video (key 'feats', e.g. actionformer_release/data/epic_kitchens/features/P01_11.npz). Uses no labels.

Pass rule (from the training plan): sequence lengths within one step, and median per-step cosine
similarity >= 0.95. Also prints the best alignment shift (an off-by-one start shows up there) and,
with --detect, how many of the top detections from the released features we also find.

    python pipeline/parity_check.py OURS.npz RELEASED.npz [--detect]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def cosine(a, b):
    return (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-8)


def compare(ours, ref, max_shift=3):
    n = min(len(ours), len(ref))
    cos = cosine(ours[:n], ref[:n])
    shifts = {}
    for k in range(-max_shift, max_shift + 1):  # ours[i + k] vs ref[i]
        a, b = (ours[k:], ref) if k >= 0 else (ours, ref[-k:])
        m = min(len(a), len(b))
        shifts[k] = float(np.median(cosine(a[:m], b[:m]))) if m else float("nan")
    return {"len_ours": len(ours), "len_ref": len(ref), "median": float(np.median(cos)),
            "p05": float(np.percentile(cos, 5)), "mean": float(cos.mean()),
            "first": float(cos[0]), "last": float(cos[-1]), "shifts": shifts,
            "mean_ours": float(ours.mean()), "mean_ref": float(ref.mean()),
            "norm_ratio": float(np.linalg.norm(ours[:n], axis=1).mean() / np.linalg.norm(ref[:n], axis=1).mean())}


def top_agreement(ours, ref, duration, k=100, iou=0.5):
    """Share of the top-k pair detections on the released features that we reproduce
    (same verb-noun pair, temporal IoU >= iou) from our features."""
    import detect_actions as da
    from fusion import tiou_matrix
    models = da.load_detectors()
    p_ours = da.detect(models, ours, duration, "ours")
    p_ref = da.detect(models, ref, duration, "ref")
    ref_s, ref_l = p_ref["segments"][:k], p_ref["labels"][:k]
    hit = 0
    for seg, lab in zip(ref_s, ref_l):
        same = p_ours["segments"][:2 * k][p_ours["labels"][:2 * k] == lab]
        hit += bool(len(same)) and tiou_matrix(seg[None], same).max() >= iou
    return hit / max(len(ref_s), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ours")
    ap.add_argument("released")
    ap.add_argument("--detect", action="store_true", help="also compare top detections (loads the detectors)")
    a = ap.parse_args()
    zo, zr = np.load(os.path.abspath(a.ours)), np.load(os.path.abspath(a.released))
    ours, ref = zo["feats"].astype(np.float32), zr["feats"].astype(np.float32)
    assert ours.shape[1] == ref.shape[1] == 2304, (ours.shape, ref.shape)
    r = compare(ours, ref)
    if "settings" in zo.files:
        print(f"our settings: {zo['settings']}")
    print(f"steps: ours {r['len_ours']}, released {r['len_ref']} (difference {r['len_ours'] - r['len_ref']:+d})")
    print(f"cosine per step: median {r['median']:.4f}, 5th pct {r['p05']:.4f}, mean {r['mean']:.4f}, "
          f"first {r['first']:.4f}, last {r['last']:.4f}")
    print("median cosine by shift (ours[i+k] vs released[i]): " +
          ", ".join(f"{k:+d}: {v:.4f}" for k, v in r["shifts"].items()))
    print(f"mean value: ours {r['mean_ours']:.4f}, released {r['mean_ref']:.4f}; norm ratio {r['norm_ratio']:.3f}")
    ok = abs(r["len_ours"] - r["len_ref"]) <= 1 and r["median"] >= 0.95
    best = max(r["shifts"], key=r["shifts"].get)
    if best != 0 and r["shifts"][best] > r["shifts"][0] + 0.01:
        print(f"note: alignment is better at shift {best:+d}, so our clips start at a different frame")
    if a.detect:
        dur = float(zo["duration"]) if "duration" in zo.files else (len(ref) * 16 + 16) / 30
        print(f"top-100 released detections we reproduce (same pair, tIoU >= 0.5): {top_agreement(ours, ref, dur):.0%}")
    print("PARITY PASS" if ok else "PARITY FAIL: check the ENCODER_* settings in encoder_a.py against the passing recipe")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
