"""Baseline action detector: MP4 (or cached features) -> verb-noun action records for Member 4.

Released ActionFormer EPIC verb and noun models, fused into pairs with the settings chosen on the
internal held-out split: fusion IoU 0.5, top 3 nouns per verb, beta 0 (no pair prior), max 2000.

Output: a JSON list sorted by score, each {"segment": [start, end], "verb": str, "noun": str, "score": float},
valid by the contract: 0 <= start < end <= duration (2 decimals), finite scores in [0, 1], never "other".
On any failure the output is [] and the error is logged; this script never raises.

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


@torch.no_grad()
def detect(models, feats, duration, video_id="video"):
    item = {"video_id": video_id, "feats": torch.from_numpy(np.ascontiguousarray(feats.T)).float(),
            "segments": None, "labels": None, "fps": 30, "duration": duration,
            "feat_stride": 16, "feat_num_frames": 32}
    raw = {}
    for kind, model in models.items():
        r = model([item])[0]
        raw[kind] = {"segments": r["segments"].numpy(), "scores": r["scores"].numpy(),
                     "labels": r["labels"].numpy().astype(np.int64)}
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
            if a.save_features:
                np.savez(a.save_features, feats=feats)
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
