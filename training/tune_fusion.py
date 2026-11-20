"""Sweep fusion and ensemble options on the held-out split, from cached model predictions.

First run caches raw held-out predictions of the control verb / noun models and the held-out joint
model to ~/epic/cache/; later runs only re-score. Nothing here touches validation videos.
    python tune_fusion.py
"""
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_holdout as E  # noqa: E402  (switches into the ActionFormer checkout)
from fusion import dedup, fuse  # noqa: E402

CK = "./ckpt"
CACHE = os.path.expanduser("~/epic/cache/holdout_preds.pkl")


def cached_predictions():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    from libs.core import load_config
    from libs.datasets import make_dataset
    raw = {}
    for kind in ("verb", "noun"):
        cfg = load_config(f"configs/epic_holdout_{kind}.yaml")
        model = E.load_model(cfg, E.resolve_ckpt(f"{CK}/epic_holdout_{kind}_control"))
        raw[kind] = E.run(model, make_dataset(cfg["dataset_name"], False, ["holdout"], **cfg["dataset"]))
    cfg = load_config(os.path.join(HERE, "configs", "epic_joint_holdout.yaml"))
    model = E.load_model(cfg, E.resolve_ckpt(f"{CK}/epic_joint_holdout_joint"))
    model.pair_topk_verb, model.pair_topk_noun = 10, 20
    raw["joint"] = E.run(model, make_dataset(cfg["dataset_name"], False, ["holdout"], **cfg["dataset"]))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    pickle.dump(raw, open(CACHE, "wb"))
    return raw


def main():
    import json
    raw = cached_predictions()
    gt_db = json.load(open(E.JOINT_JSON))["database"]
    pair_eval = E.ANETdetection(E.JOINT_JSON, "holdout", tiou_thresholds=E.TIOUS, label="pair_id")
    prior = np.load(E.PRIOR)
    vids = sorted(raw["verb"])

    def score(name, preds):
        maps, avg = E.pair_map(preds, pair_eval)
        E.log_row("tune", "cached", name, maps, avg, E.coverage_proxy(preds, gt_db))
        return avg

    def control(**kw):
        base = dict(iou_thr=0.5)
        base.update(kw)
        return {v: fuse(raw["verb"][v], raw["noun"][v], prior, **base) for v in vids}

    def ensemble(ctrl, w, joint=None):
        joint = joint or raw["joint"]
        out = {}
        for v in vids:
            a, b = ctrl[v], joint[v]
            out[v] = dedup(np.concatenate([a["segments"], b["segments"]]),
                           np.concatenate([a["scores"], w * b["scores"]]),
                           np.concatenate([a["labels"], b["labels"]]), dedup_iou=0.7)
        return out

    if "--seeds" in sys.argv:
        cache2 = CACHE.replace(".pkl", "_s2.pkl")
        if os.path.exists(cache2):
            raw2 = pickle.load(open(cache2, "rb"))
        else:
            from libs.core import load_config
            from libs.datasets import make_dataset
            raw2 = {}
            for kind in ("verb", "noun"):
                cfg = load_config(f"configs/epic_holdout_{kind}_s2.yaml")
                model = E.load_model(cfg, E.resolve_ckpt(f"{CK}/epic_holdout_{kind}_s2_control"))
                raw2[kind] = E.run(model, make_dataset(cfg["dataset_name"], False, ["holdout"], **cfg["dataset"]))
            pickle.dump(raw2, open(cache2, "wb"))

        def merge_seeds(kind, vote):
            out = {}
            for v in vids:
                a, b = raw[kind][v], raw2[kind][v]
                out[v] = dedup(np.concatenate([a["segments"], b["segments"]]),
                               np.concatenate([a["scores"], b["scores"]]),
                               np.concatenate([a["labels"], b["labels"]]), dedup_iou=0.7, vote=vote)
            return out

        def fused(verb, noun, k):
            return {v: fuse(verb[v], noun[v], prior, iou_thr=0.5, top_nouns=k) for v in vids}

        for k in (3, 10):
            score(f"seed1 only top_nouns={k}", fused(raw["verb"], raw["noun"], k))
            score(f"seed2 only top_nouns={k}", fused(raw2["verb"], raw2["noun"], k))
            for vote in (False, True):
                c = fused(merge_seeds("verb", vote), merge_seeds("noun", vote), k)
                score(f"seed1+2 vote={vote} top_nouns={k}", c)
                score(f"  + 0.5*joint", ensemble(c, 0.5))
        return

    if "--round2" in sys.argv:
        for k in (10, 20, 40):
            for tv in (0, 1, 2):
                for iou in (0.4, 0.5):
                    c = control(top_nouns=k, top_verbs=tv, iou_thr=iou)
                    score(f"control iou={iou} top_nouns={k} top_verbs={tv}", c)
                    for w in (0.25, 0.5):
                        score(f"  + {w}*joint", ensemble(c, w))
        return

    score("control base iou0.5 top3", control())
    for k in (1, 5, 10):
        score(f"control top_nouns={k}", control(top_nouns=k))
    for mode in ("verb", "noun", "union", "inter"):
        score(f"control seg={mode}", control(seg_mode=mode))
    for tv in (1, 3):
        score(f"control symmetric top_verbs={tv}", control(top_verbs=tv))
    for vp, np_ in ((0.5, 1.0), (1.0, 0.5), (1.5, 1.0), (1.0, 1.5)):
        score(f"control pow verb={vp} noun={np_}", control(verb_pow=vp, noun_pow=np_))
    for d in (0.5, 0.9):
        score(f"control dedup_iou={d}", control(dedup_iou=d))

    ctrl = control()
    score("joint alone 10x20", raw["joint"])
    for w in (0.5, 1.0, 2.0):
        for vote in (False, True):
            ens = {}
            for v in vids:
                a, b = ctrl[v], raw["joint"][v]
                ens[v] = dedup(np.concatenate([a["segments"], b["segments"]]),
                               np.concatenate([a["scores"], w * b["scores"]]),
                               np.concatenate([a["labels"], b["labels"]]), dedup_iou=0.7, vote=vote)
            score(f"ensemble control+{w}*joint vote={vote}", ens)


if __name__ == "__main__":
    main()
