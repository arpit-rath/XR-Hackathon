"""Merge ActionFormer's EPIC verb and noun JSONs into one joint JSON, and build pair priors.

Each joint annotation carries `label_id` (verb), `noun_id` and `pair_id` = verb * 300 + noun.
Two outputs, both without any validation or test video:
  epic_joint_holdout.json  subsets 'training' / 'holdout' (held-out list from splits/holdout.txt)
  epic_joint_full.json     all 495 train videos as 'training'
and a pair prior for each: log((count + 1) / (max_count + 1)) over 97 x 300, from its 'training' subset only.

Usage:
    python build_joint_json.py ANNOTATION_DIR OUT_DIR HOLDOUT_LIST
"""
import json
import sys
from pathlib import Path

import numpy as np

NUM_VERBS, NUM_NOUNS = 97, 300


def merge(verb_db, noun_db, video_ids, holdout):
    out = {}
    for vid in video_ids:
        v, n = verb_db[vid], noun_db[vid]
        va, na = v["annotations"], n["annotations"]
        assert len(va) == len(na), f"{vid}: verb/noun annotation counts differ"
        anns = []
        for a, b in zip(va, na):
            assert a["segment"] == b["segment"], f"{vid}: verb/noun segments not aligned"
            vi, ni = int(a["label_id"]), int(b["label_id"])
            anns.append({"segment": a["segment"], "label": str(vi), "label_id": vi,
                         "noun_id": ni, "pair_id": vi * NUM_NOUNS + ni})
        entry = {k: val for k, val in v.items() if k != "annotations"}
        entry["annotations"] = anns
        entry["subset"] = "holdout" if vid in holdout else "training"
        out[vid] = entry
    return out


def pair_prior(db):
    counts = np.zeros((NUM_VERBS, NUM_NOUNS), dtype=np.float64)
    for entry in db.values():
        if entry["subset"] != "training":
            continue
        for a in entry["annotations"]:
            counts[a["label_id"], a["noun_id"]] += 1
    return np.log((counts + 1.0) / (counts.max() + 1.0)).astype(np.float32), counts


def main(ann_dir: Path, out_dir: Path, holdout_list: Path) -> None:
    verb = json.loads((ann_dir / "epic_kitchens_100_verb.json").read_text())
    noun = json.loads((ann_dir / "epic_kitchens_100_noun.json").read_text())
    train_ids = sorted(k for k, v in verb["database"].items() if v["subset"] == "training")
    holdout = set(holdout_list.read_text().split())
    assert holdout <= set(train_ids), "held-out list must only contain train videos"

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, held in (("holdout", holdout), ("full", set())):
        db = merge(verb["database"], noun["database"], train_ids, held)
        (out_dir / f"epic_joint_{name}.json").write_text(json.dumps({"version": "joint", "database": db}))
        prior, counts = pair_prior(db)
        np.save(out_dir / f"pair_log_prior_{name}.npy", prior)
        n_train = sum(e["subset"] == "training" for e in db.values())
        print(f"{name}: {n_train} training videos, {len(db) - n_train} held-out, "
              f"{int((counts > 0).sum())} distinct training pairs")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
