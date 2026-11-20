"""Score models on the internal held-out split (never on validation).

Metric: verb-noun pair mAP averaged over tIoU 0.1-0.5 (ActionFormer's ANETdetection, pair label =
verb * 300 + noun), plus a coverage proxy: per video, take the top predictions until their union
covers 15% of the video, and report the share of distinct GT pairs hit by an overlapping prediction
of the same pair.

  joint model (sweeps the pair prior exponent beta):
    python eval_holdout.py joint CKPT [--config configs/epic_joint_holdout.yaml] [--betas 0 0.25 0.5]
  control = separate verb + noun models fused into pairs (sweeps beta and the fusion IoU):
    python eval_holdout.py control VERB_CKPT NOUN_CKPT [--betas ...] [--ious 0.1 0.3 0.5]

CKPT may be a .pth.tar file or a checkpoint folder (latest epoch is used).
Results are appended to ~/epic/logs/holdout_results.tsv.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import joint_actionformer  # noqa: E402

os.chdir(joint_actionformer.AF_DIR)
import torch  # noqa: E402
from fusion import NUM_NOUNS, fuse, tiou_matrix  # noqa: E402
from libs.core import load_config  # noqa: E402
from libs.datasets import make_dataset  # noqa: E402
from libs.modeling import make_meta_arch  # noqa: E402
from libs.utils import ANETdetection, fix_random_seed  # noqa: E402

ANN = "./data/epic_kitchens/annotations"
JOINT_JSON = f"{ANN}/joint/epic_joint_holdout.json"
PRIOR = f"{ANN}/joint/pair_log_prior_holdout.npy"
TIOUS = np.linspace(0.1, 0.5, 5)
RESULTS_TSV = os.path.expanduser("~/epic/logs/holdout_results.tsv")


def resolve_ckpt(path):
    if path.endswith(".pth.tar"):
        return path
    files = sorted(glob.glob(os.path.join(path, "epoch_*.pth.tar")))
    assert files, f"no checkpoints in {path}"
    return files[-1]


def load_model(cfg, ckpt):
    model = make_meta_arch(cfg["model_name"], **cfg["model"]).cuda().eval()
    state = torch.load(ckpt, map_location="cuda", weights_only=False)["state_dict_ema"]
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
    return model


def run(model, dataset):
    """Inference on every video of `dataset`; returns {video_id: dict of numpy arrays}."""
    out = {}
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            r = model([item])[0]
            out[item["video_id"]] = {"segments": r["segments"].numpy(), "scores": r["scores"].numpy(),
                                     "labels": r["labels"].numpy().astype(np.int64)}
    return out


def to_frame(preds):
    vids, st, en, lab, sc = [], [], [], [], []
    for vid, p in preds.items():
        n = len(p["scores"])
        vids += [vid] * n
        st.append(p["segments"][:, 0]); en.append(p["segments"][:, 1])
        lab.append(p["labels"]); sc.append(p["scores"])
    return {"video-id": vids, "t-start": np.concatenate(st), "t-end": np.concatenate(en),
            "label": np.concatenate(lab), "score": np.concatenate(sc)}


def evaluate(preds, evaluator):
    """ANETdetection remaps GT labels to 0..K but leaves unknown predicted labels as raw ids, which
    can collide with those indices; predictions of labels absent from GT cannot change AP, so drop them."""
    frame = to_frame(preds)
    keep = np.isin(frame["label"], np.asarray(list(evaluator.activity_index.keys())))
    frame = {k: (np.asarray(v)[keep] if k == "video-id" else v[keep]) for k, v in frame.items()}
    maps, avg, _ = evaluator.evaluate(frame, verbose=False)
    return maps, avg


def pair_map(preds, evaluator):
    return evaluate(preds, evaluator)


def coverage_proxy(preds, gt_db, budget=0.15):
    shares = []
    for vid, p in preds.items():
        entry = gt_db[vid]
        gt_pairs = {}
        for a in entry["annotations"]:
            gt_pairs.setdefault(a["pair_id"], []).append(a["segment"])
        if not gt_pairs:
            continue
        dur, picked, covered = entry["duration"], [], []
        for i in np.argsort(-p["scores"]):
            picked.append(i)
            covered.append(p["segments"][i])
            iv = sorted(map(tuple, covered))
            total, cur_s, cur_e = 0.0, iv[0][0], iv[0][1]
            for s, e in iv[1:]:
                if s > cur_e:
                    total += cur_e - cur_s
                    cur_s, cur_e = s, e
                else:
                    cur_e = max(cur_e, e)
            total += cur_e - cur_s
            if total >= budget * dur:
                break
        picked = np.asarray(picked)
        hit = 0
        for pair, segs in gt_pairs.items():
            sel = picked[p["labels"][picked] == pair]
            if sel.size and (tiou_matrix(np.asarray(segs, np.float32), p["segments"][sel]) > 0).any():
                hit += 1
        shares.append(hit / len(gt_pairs))
    return float(np.mean(shares))


def log_row(name, ckpt, extra, maps, avg, cov):
    row = [time.strftime("%Y-%m-%d %H:%M"), name, ckpt, extra] + [f"{m * 100:.2f}" for m in maps] + \
          [f"{avg * 100:.2f}", f"{cov * 100:.1f}"]
    new = not os.path.exists(RESULTS_TSV)
    with open(RESULTS_TSV, "a") as f:
        if new:
            f.write("\t".join(["time", "model", "ckpt", "setting"] + [f"mAP@{t:.1f}" for t in TIOUS] +
                              ["avg_mAP", "coverage_proxy_%"]) + "\n")
        f.write("\t".join(row) + "\n")
    print(f"{name:8s} {extra:28s} pair mAP " + " / ".join(f"{m * 100:.2f}" for m in maps) +
          f" | avg {avg * 100:.2f} | coverage proxy {cov * 100:.1f}%", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["joint", "control"])
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "epic_joint_holdout.yaml"))
    ap.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    ap.add_argument("--ious", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    ap.add_argument("--pair-topk", nargs="+", default=["5x10"],
                    help="joint only: verbs x nouns kept per point, e.g. 5x10 10x20")
    ap.add_argument("--pre-nms-topk", type=int, default=None, help="joint only: candidates kept per level")
    args = ap.parse_args()

    fix_random_seed(0, include_cuda=True)
    import json
    gt_db = json.load(open(JOINT_JSON))["database"]
    pair_eval = ANETdetection(JOINT_JSON, "holdout", tiou_thresholds=TIOUS, label="pair_id")

    if args.mode == "joint":
        ckpt = resolve_ckpt(args.ckpts[0])
        cfg = load_config(args.config)
        model = load_model(cfg, ckpt)
        ds = make_dataset(cfg["dataset_name"], False, ["holdout"], **cfg["dataset"])
        print(f"joint: {ckpt} on {len(ds)} held-out videos")
        if args.pre_nms_topk:
            model.test_pre_nms_topk = args.pre_nms_topk
        for topk in args.pair_topk:
            model.pair_topk_verb, model.pair_topk_noun = map(int, topk.split("x"))
            for beta in args.betas:
                model.pair_beta = beta
                preds = run(model, ds)
                maps, avg = pair_map(preds, pair_eval)
                log_row("joint", ckpt, f"topk={topk} pre={model.test_pre_nms_topk} beta={beta}",
                        maps, avg, coverage_proxy(preds, gt_db))
        return

    # control: separate verb / noun models (stock ActionFormer) fused into pairs
    assert len(args.ckpts) == 2, "control needs VERB_CKPT NOUN_CKPT"
    raw = {}
    for kind, path in zip(("verb", "noun"), args.ckpts):
        ckpt = resolve_ckpt(path)
        cfg = load_config(f"configs/epic_holdout_{kind}.yaml")
        model = load_model(cfg, ckpt)
        ds = make_dataset(cfg["dataset_name"], False, ["holdout"], **cfg["dataset"])
        raw[kind] = run(model, ds)
        single = ANETdetection(f"{ANN}/holdout/epic_kitchens_100_{kind}_holdout.json", "holdout",
                               tiou_thresholds=TIOUS)
        maps, avg = evaluate(raw[kind], single)
        print(f"{kind}-only mAP: " + " / ".join(f"{m * 100:.2f}" for m in maps) + f" | avg {avg * 100:.2f}"
              f"  ({ckpt})", flush=True)
        del model
        torch.cuda.empty_cache()

    log_prior = np.load(PRIOR)
    for iou_thr in args.ious:
        for beta in args.betas:
            preds = {vid: fuse(raw["verb"][vid], raw["noun"][vid], log_prior, iou_thr=iou_thr, beta=beta)
                     for vid in raw["verb"]}
            maps, avg = pair_map(preds, pair_eval)
            log_row("control", f"{args.ckpts[0]} + {args.ckpts[1]}", f"fusion_iou={iou_thr} beta={beta}",
                    maps, avg, coverage_proxy(preds, gt_db))


if __name__ == "__main__":
    main()
