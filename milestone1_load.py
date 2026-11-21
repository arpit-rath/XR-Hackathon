import argparse
import csv
import json
from pathlib import Path

base_downloads = Path.home() / "Downloads"
manifest_path = base_downloads / "XR-Hackathon" / "package" / "manifest.csv"
actions_dir = base_downloads / "transfer" / "transfer" / "actions_validation"
fpvsum_dir = base_downloads / "transfer" / "transfer" / "fpvsum_outputs"

parser = argparse.ArgumentParser()
parser.add_argument("--video", default="P01_11")
args = parser.parse_args()

video_id = args.video
duration_sec = None
dataset = None

with manifest_path.open("r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        if row["video_id"] == video_id:
            duration_sec = float(row["duration_sec"])
            dataset = row["dataset"]
            break

actions = []
action_file = actions_dir / f"{video_id}.json"
if action_file.exists():
    with action_file.open("r", encoding="utf-8") as f:
        actions = json.load(f)

label = {"name": "cooking", "score": 1.0} if dataset == "epic_kitchens_100" else None
importance_count = 0
top_5_importance = []

fpvsum_file = fpvsum_dir / f"{video_id}.json"
if fpvsum_file.exists():
    with fpvsum_file.open("r", encoding="utf-8") as f:
        fpv_data = json.load(f)
    label = fpv_data.get("label")
    scores = fpv_data.get("importance", {}).get("scores", [])
    importance_count = len(scores)
    top_5_importance = sorted(scores, reverse=True)[:5]

combined = {
    "video_id": video_id,
    "dataset": dataset,
    "duration_sec": duration_sec,
    "label": label,
    "action_count": len(actions),
    "importance_score_count": importance_count,
    "top_5_importance_scores": top_5_importance,
}

print(json.dumps(combined, indent=2))
