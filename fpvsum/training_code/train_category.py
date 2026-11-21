"""Train the FPVSum video-category classifier on the NON-Samsung unlabeled FPVSum videos.

Training data : training/features/fpvsum_unlabeled/*.npz  (category from the official FPVSum list)
Never used    : Samsung's 49 manifest videos (asserted below).
Protocol (all hyper-parameters fixed a priori, no search):
  * Linear probe on pooled CLIP features ([mean,max] of random temporal crops), AdamW lr 1e-3,
    wd 1e-2, 600 steps, batch 64 crops; blended with zero-shot CLIP: p = 0.5*p_probe + 0.5*p_zeroshot.
  * Validation A (fixed held-out split): 1 video per class held out for classes with >= 2 videos.
  * Validation B: leave-one-video-out (LOVO) over all videos.
  * Final model: retrained on all videos with the same settings.
Checkpoints are written every 100 steps to training/checkpoints/category/.
"""
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

WS = Path(__file__).resolve().parent.parent
PKG = WS / "export" / "Member3_FPVSum_Trained_Model"
sys.path.insert(0, str(PKG))
from member3_fpvsum.models import CategoryHead, pool_features  # noqa: E402

FEAT = WS / "training" / "features" / "fpvsum_unlabeled"
CKPT = WS / "training" / "checkpoints" / "category"
OUT_METRICS = WS / "training" / "metrics"
HP = dict(steps=600, lr=1e-3, wd=1e-2, batch=64, min_crop=0.3, alpha_probe=0.5, temperature_zs=100.0, seed=0)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load():
    te = torch.load(PKG / "weights" / "text_embeddings.pt", weights_only=True)
    labels = te["labels"]
    samsung = {r["samsung_id"] for r in csv.DictReader(open(WS / "reports" / "fpvsum_matched.csv", encoding="utf-8"))}
    rep = {r["video_name"]: r for r in csv.DictReader(open(WS / "reports" / "unlabeled_download_report.csv", encoding="utf-8"))}
    vids = []
    for p in sorted(FEAT.glob("*.npz")):
        assert p.stem not in samsung, f"LEAK: {p.stem} is a Samsung evaluation video"
        cat = rep[p.stem]["category"]
        vids.append({"id": p.stem, "y": labels.index(cat), "x": torch.from_numpy(np.load(p)["feats"].astype(np.float32))})
    return vids, labels, te["category"].float()


def crops(x, n, rng):
    T = len(x)
    out = []
    for _ in range(n):
        L = max(1, int(T * rng.uniform(HP["min_crop"], 1.0)))
        s = rng.randint(0, T - L)
        out.append(pool_features(x[s:s + L]))
    return torch.stack(out)


def train(train_vids, n_cls, tag, seed=0):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    head = CategoryHead(1536, n_cls).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    (CKPT / tag).mkdir(parents=True, exist_ok=True)
    for step in range(1, HP["steps"] + 1):
        batch = [rng.choice(train_vids) for _ in range(HP["batch"])]
        X = torch.stack([crops(v["x"], 1, rng)[0] for v in batch]).to(DEV)
        y = torch.tensor([v["y"] for v in batch], device=DEV)
        loss = F.cross_entropy(head(X) , y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            torch.save({"state_dict": head.state_dict(), "step": step, "loss": float(loss), "hp": HP},
                       CKPT / tag / f"step{step:04d}.pt")
    return head.eval()


@torch.no_grad()
def predict(head, x, text_cat):
    v = pool_features(x)
    p_probe = F.softmax(head(v.to(DEV)).cpu(), -1)
    p_zs = F.softmax(HP["temperature_zs"] * (F.normalize(x.mean(0), dim=-1) @ text_cat.T), -1)
    p = HP["alpha_probe"] * p_probe + (1 - HP["alpha_probe"]) * p_zs
    return p, p_probe, p_zs


def evaluate(heads_by_vid, vids, text_cat, labels):
    rows, acc = [], {"blend": 0, "probe": 0, "zeroshot": 0}
    for v in vids:
        p, pp, pz = predict(heads_by_vid[v["id"]], v["x"], text_cat)
        r = {"video": v["id"], "true": labels[v["y"]], "blend": labels[int(p.argmax())], "blend_conf": round(float(p.max()), 3),
             "probe": labels[int(pp.argmax())], "zeroshot": labels[int(pz.argmax())]}
        for k in acc:
            acc[k] += int(r[k] == r["true"])
        rows.append(r)
    return {k: round(a / len(vids), 4) for k, a in acc.items()}, rows


def main():
    vids, labels, text_cat = load()
    n = len(labels)
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    print(f"{len(vids)} training-source videos; classes present: {len({v['y'] for v in vids})}/{n}")

    # A: fixed held-out split
    rng = random.Random(HP["seed"])
    by = {}
    for v in vids:
        by.setdefault(v["y"], []).append(v)
    val = [rng.choice(g) for g in by.values() if len(g) >= 2]
    val_ids = {v["id"] for v in val}
    tr = [v for v in vids if v["id"] not in val_ids]
    head = train(tr, n, "holdout")
    accA, rowsA = evaluate({v["id"]: head for v in val}, val, text_cat, labels)
    print("held-out split:", len(tr), "train /", len(val), "val ->", accA)

    # B: leave-one-video-out
    heads = {}
    for i, v in enumerate(vids):
        heads[v["id"]] = train([u for u in vids if u["id"] != v["id"]], n, f"lovo/{i:02d}")
    accB, rowsB = evaluate(heads, vids, text_cat, labels)
    print("LOVO:", accB)

    # Final model on all training-source videos
    final = train(vids, n, "final")
    torch.save({"state_dict": final.state_dict(), "cfg": final.cfg, "labels": labels, "hp": HP},
               PKG / "weights" / "category_head.pt")

    metrics = {"n_videos": len(vids), "hyperparameters": HP,
               "heldout_split": {"n_train": len(tr), "n_val": len(val), "accuracy": accA, "per_video": rowsA},
               "leave_one_video_out": {"accuracy": accB, "per_video": rowsB},
               "note": "Validation uses only non-Samsung FPVSum videos. Samsung's 49 evaluation videos were not used."}
    json.dump(metrics, open(OUT_METRICS / "category_metrics.json", "w"), indent=2)
    print("saved final head + metrics")


if __name__ == "__main__":
    main()
