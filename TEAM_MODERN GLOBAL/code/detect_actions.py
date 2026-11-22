"""Baseline action detector: MP4 (or cached features) -> verb-noun action records for Member 4.

Released ActionFormer EPIC verb and noun models, fused into pairs with the settings chosen on the
internal held-out split: fusion IoU 0.5, top 3 nouns per verb, beta 0 (no pair prior), max 2000.

Output: a JSON list sorted by score, each {"segment": [start, end], "verb": str, "noun": str, "score": float},
valid by the contract: 0 <= start < end <= duration (2 decimals), finite scores in [0, 1], never "other".
On any failure the output is [] and the error is logged; this script never raises.

Memory: one pass over the whole sequence needs > 10 GB on long videos. On GPUs under 12 GB (or after
an out-of-memory error) the sequence is cut into 2,304-step windows overlapping by 256; each
detection is kept only by the window that owns its centre (split at the middle of each overlap).
AF_WINDOW=<steps> forces windowing, AF_WINDOW=0 forces one pass.

    python detect_actions.py VIDEO.mp4 OUT.json [--features cached.npz] [--save-features feats.npz]
"""
import argparse
import csv
import json
import logging
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "training"))
sys.path.insert(0, HERE)

AF_DIR = os.environ.get("ACTIONFORMER_DIR", os.path.expanduser("~/epic/actionformer_release"))
WEIGHTS = {k: os.environ.get(f"{k.upper()}_CKPT", f"{AF_DIR}/pretrained/epic_slowfast_{k}_reproduce/epoch_020.pth.tar")
           for k in ("verb", "noun")}
CLASSES_JSON = os.environ.get("SUPPORTED_CLASSES", "")  # organisers' supported_classes.json, when available
ANN_DIR = os.environ.get("EPIC_ANN_DIR", os.path.expanduser("~/epic/epic-kitchens-100-annotations"))
FUSION = dict(iou_thr=0.5, top_nouns=3, beta=0.0, max_out=2000)
MAX_RECORDS = 2000
FPS, FEAT_STRIDE, FEAT_FRAMES = 30, 16, 32
WINDOW, OVERLAP, SMALL_GPU_GB = 2304, 256, 12
log = logging.getLogger("detect_actions")


def class_names():
    """verbs[i], nouns[j] in EPIC class-id order. Prefers supported_classes.json; falls back to EPIC CSVs."""
    if CLASSES_JSON and os.path.exists(CLASSES_JSON):
        d = json.load(open(CLASSES_JSON))
        verbs, nouns = [v for v in d["verbs"] if v != "other"], [n for n in d["nouns"] if n != "other"]
    else:
        def read(name):
            with open(os.path.join(ANN_DIR, name)) as f:
                rows = sorted(csv.DictReader(f), key=lambda r: int(r["id"]))
            return [r["key"] for r in rows]
        verbs, nouns = read("EPIC_100_verb_classes.csv"), read("EPIC_100_noun_classes.csv")
    assert len(verbs) == 97 and len(nouns) == 300, (len(verbs), len(nouns))
    return verbs, nouns


def load_detectors():
    os.chdir(AF_DIR)
    if AF_DIR not in sys.path:
        sys.path.insert(0, AF_DIR)
    from libs.core import load_config
    from libs.modeling import make_meta_arch
    models = {}
    for kind in ("verb", "noun"):
        cfg = load_config(f"configs/epic_slowfast_{kind}.yaml")
        model = make_meta_arch(cfg["model_name"], **cfg["model"]).cuda().eval()
        state = torch.load(WEIGHTS[kind], map_location="cuda", weights_only=False)["state_dict_ema"]
        model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        models[kind] = model
    return models


def window_spans(n, window=WINDOW, overlap=OVERLAP):
    """[(start, end, own_from, own_to)] in feature steps. A detection whose centre step c satisfies
    own_from <= c < own_to belongs to that window. One span covering everything when n <= window."""
    if n <= window:
        return [(0, n, -math.inf, math.inf)]
    starts = list(range(0, n - window, window - overlap)) + [n - window]
    cuts = [(b + a + window) / 2 for a, b in zip(starts, starts[1:])]  # middle of each overlap
    bounds = [-math.inf] + cuts + [math.inf]
    return [(s, s + window, bounds[k], bounds[k + 1]) for k, s in enumerate(starts)]


def choose_window():
    if os.environ.get("AF_WINDOW") is not None:
        return int(os.environ["AF_WINDOW"])
    try:
        gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    except Exception:
        gb = 0.0
    return WINDOW if gb < SMALL_GPU_GB else 0


def run_model(model, feats, duration, video_id, window):
    """One ActionFormer model over the sequence, in one pass (window=0) or in windows. Times in seconds."""
    segs, scores, labels = [], [], []
    for s, e, own_from, own_to in (window_spans(len(feats), window) if window else
                                   [(0, len(feats), -math.inf, math.inf)]):
        offset = s * FEAT_STRIDE / FPS  # global step s+i is centred at offset + (16 i + 16) / 30
        item = {"video_id": video_id, "feats": torch.from_numpy(np.ascontiguousarray(feats[s:e].T)).float(),
                "segments": None, "labels": None, "fps": FPS, "duration": duration - offset,
                "feat_stride": FEAT_STRIDE, "feat_num_frames": FEAT_FRAMES}
        r = model([item])[0]
        seg = r["segments"].numpy().reshape(-1, 2) + offset
        centre = seg.mean(1) * FPS / FEAT_STRIDE - 1  # back to global feature steps
        keep = (centre >= own_from) & (centre < own_to)
        segs.append(seg[keep]); scores.append(r["scores"].numpy()[keep])
        labels.append(r["labels"].numpy().astype(np.int64)[keep])
    segs, scores, labels = np.concatenate(segs), np.concatenate(scores), np.concatenate(labels)
    order = np.argsort(-scores, kind="stable")[:MAX_RECORDS]
    return {"segments": segs[order], "scores": scores[order], "labels": labels[order]}


@torch.no_grad()
def detect(models, feats, duration, video_id="video", window=None):
    window = choose_window() if window is None else window
    raw = {}
    for kind, model in models.items():
        try:
            raw[kind] = run_model(model, feats, duration, video_id, window)
        except torch.cuda.OutOfMemoryError:
            if window and window <= WINDOW:
                raise
            log.warning("%s: %s model ran out of GPU memory in one pass; retrying in %d-step windows",
                        video_id, kind, WINDOW)
            torch.cuda.empty_cache()
            raw[kind] = run_model(model, feats, duration, video_id, WINDOW)
    from fusion import fuse
    return fuse(raw["verb"], raw["noun"], np.zeros((97, 300), np.float32), **FUSION)


def to_records(pairs, duration, verbs, nouns):
    records = []
    for (s, e), score, label in zip(pairs["segments"], pairs["scores"], pairs["labels"]):
        if not (math.isfinite(s) and math.isfinite(e) and math.isfinite(score)):
            continue
        s, e = round(max(0.0, float(s)), 2), round(min(float(duration), float(e)), 2)
        if not (0 <= s < e <= duration):
            continue
        v, n = int(label) // 300, int(label) % 300
        records.append({"segment": [s, e], "verb": verbs[v], "noun": nouns[n],
                        "score": round(min(1.0, max(0.0, float(score))), 6)})
    records.sort(key=lambda r: -r["score"])
    return records[:MAX_RECORDS]


def check_contract(records, duration, verbs, nouns):
    assert len(records) <= MAX_RECORDS
    vs, ns = set(verbs), set(nouns)
    for r in records:
        s, e = r["segment"]
        assert 0 <= s < e <= duration, r
        assert r["verb"] in vs and r["noun"] in ns and "other" not in (r["verb"], r["noun"]), r
        assert 0 <= r["score"] <= 1 and math.isfinite(r["score"]), r
    assert all(a["score"] >= b["score"] for a, b in zip(records, records[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--features", help="use cached Encoder A features (.npz with 'feats') instead of the MP4")
    ap.add_argument("--save-features", help="also write the extracted features here")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = os.path.abspath(a.out)
    records = []
    try:
        t0 = time.time()
        import encoder_a
        _, _, duration = encoder_a.probe(a.video)
        if a.features:
            feats = np.load(a.features)["feats"]
        else:
            feats, duration = encoder_a.extract(a.video)
            torch.cuda.empty_cache()  # the encoder is gone; give its memory to the detectors
            if a.save_features:
                np.savez(a.save_features, feats=feats, duration=duration, settings=encoder_a.SETTINGS)
        t1 = time.time()
        verbs, nouns = class_names()
        models = load_detectors()
        pairs = detect(models, feats, duration, os.path.splitext(os.path.basename(a.video))[0])
        records = to_records(pairs, duration, verbs, nouns)
        check_contract(records, duration, verbs, nouns)
        log.info("%s: %.1fs video, %d feature steps, %d records (features %.1fs, detection %.1fs)",
                 a.video, duration, len(feats), len(records), t1 - t0, time.time() - t1)
    except Exception:
        log.exception("action detection failed for %s; writing []", a.video)
        records = []
    with open(out, "w") as f:
        json.dump(records, f)


if __name__ == "__main__":
    main()
