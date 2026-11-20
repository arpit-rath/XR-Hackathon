"""Build the internal held-out split and the ActionFormer JSONs for the control run.

Held-out = every train video of two participants (mimicking the two unseen
participants in validation) + a random 5% of the remaining train videos.
Validation and test videos are dropped from the output JSONs entirely, so no
validation label can reach training or tuning.

Usage:
    python make_holdout_split.py ANNOTATION_DIR OUT_DIR SPLIT_FILE
      ANNOTATION_DIR  folder with epic_kitchens_100_{verb,noun}.json from the ActionFormer release
      OUT_DIR         where epic_kitchens_100_{verb,noun}_holdout.json are written
      SPLIT_FILE      where the held-out video list is written (one id per line)
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

SEED = 2026
PARTICIPANT_VIDEO_RANGE = (10, 15)  # held-out participants are picked among those with this many train videos
RANDOM_FRACTION = 0.05


def main(ann_dir: Path, out_dir: Path, split_file: Path) -> None:
    rng = random.Random(SEED)
    verb = json.loads((ann_dir / "epic_kitchens_100_verb.json").read_text())
    train_ids = sorted(k for k, v in verb["database"].items() if v["subset"] == "training")

    per_participant = Counter(vid.split("_")[0] for vid in train_ids)
    lo, hi = PARTICIPANT_VIDEO_RANGE
    candidates = sorted(p for p, n in per_participant.items() if lo <= n <= hi)
    held_participants = sorted(rng.sample(candidates, 2))

    rest = [v for v in train_ids if v.split("_")[0] not in held_participants]
    n_random = round(RANDOM_FRACTION * len(rest))
    holdout = sorted(
        [v for v in train_ids if v.split("_")[0] in held_participants] + rng.sample(rest, n_random)
    )
    holdout_set = set(holdout)

    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text("\n".join(holdout) + "\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("verb", "noun"):
        src = json.loads((ann_dir / f"epic_kitchens_100_{kind}.json").read_text())
        db = {}
        for vid in train_ids:
            entry = dict(src["database"][vid])
            entry["subset"] = "holdout" if vid in holdout_set else "training"
            db[vid] = entry
        out = {"version": src.get("version", ""), "database": db}
        (out_dir / f"epic_kitchens_100_{kind}_holdout.json").write_text(json.dumps(out))

    n_seg = sum(len(verb["database"][v]["annotations"]) for v in holdout)
    hours = sum(verb["database"][v]["duration"] for v in holdout) / 3600
    print(f"held-out participants: {held_participants} ({sum(per_participant[p] for p in held_participants)} videos)")
    print(f"held-out videos: {len(holdout)} ({n_random} random), {hours:.1f} h, {n_seg} segments")
    print(f"training videos: {len(train_ids) - len(holdout)}; validation/test videos dropped")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
