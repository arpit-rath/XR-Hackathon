import argparse
import csv
import json
from pathlib import Path

base_downloads = Path.home() / "Downloads"
manifest_path = base_downloads / "XR-Hackathon" / "package" / "manifest.csv"
actions_dir = base_downloads / "transfer" / "transfer" / "actions_validation"
fpvsum_dir = base_downloads / "transfer" / "transfer" / "fpvsum_outputs"
default_out_dir = base_downloads / "XR-Hackathon" / "outputs"


def r2(val: float) -> float:
    return round(float(val), 2)


def ensure_min_highlight(chosen, duration_sec: float, min_len: float = 4.0):
    """Ensure at least one highlight segment has length >= min_len within [0, duration_sec]."""
    if not chosen:
        end_t = min(duration_sec, min_len)
        return [(0.0, end_t)] if end_t > 0 else [(0.0, 0.1)]

    if max((e - s) for s, e in chosen) >= min_len - 1e-9:
        return chosen

    idx = max(range(len(chosen)), key=lambda i: chosen[i][1] - chosen[i][0])
    s, e = chosen[idx]
    target_len = min(duration_sec, min_len)
    new_s = max(0.0, min(s, duration_sec - target_len))
    new_e = min(duration_sec, new_s + target_len)
    new_s = max(0.0, new_e - target_len)

    kept = [(new_s, new_e)]
    for i, (os_, oe) in enumerate(chosen):
        if i == idx:
            continue
        if oe <= new_s or os_ >= new_e:
            kept.append((os_, oe))
    kept.sort(key=lambda x: x[0])
    return kept


def select_non_overlapping(candidates, duration_sec: float, min_len: float = 4.0):
    """candidates: list of (start, end, score). Returns list of {"segment": [s, e]}."""
    budget_sec = 0.15 * duration_sec

    cleaned = []
    for s, e, sc in candidates:
        s_c = max(0.0, min(float(s), duration_sec))
        e_c = max(0.0, min(float(e), duration_sec))
        if e_c - s_c >= min_len:
            cleaned.append((s_c, e_c, float(sc)))

    cleaned.sort(key=lambda x: (-x[2], -(x[1] - x[0])))

    chosen = []
    used_sec = 0.0
    for s, e, _ in cleaned:
        length = e - s
        if used_sec + length > budget_sec + 1e-9:
            continue
        overlaps = any(not (e <= cs or s >= ce) for cs, ce in chosen)
        if not overlaps:
            chosen.append((s, e))
            used_sec += length

    if not chosen:
        raw_cands = []
        for s, e, sc in candidates:
            s_c = max(0.0, min(float(s), duration_sec))
            e_c = max(0.0, min(float(e), duration_sec))
            if e_c > s_c:
                center = 0.5 * (s_c + e_c)
                ws = max(0.0, min(center - 0.5 * min_len, duration_sec - min_len))
                we = min(duration_sec, ws + min_len)
                raw_cands.append((ws, we, float(sc)))
        raw_cands.sort(key=lambda x: -x[2])
        for ws, we, _ in raw_cands:
            wlen = we - ws
            if used_sec + wlen <= budget_sec + 1e-9 and not any(not (we <= cs or ws >= ce) for cs, ce in chosen):
                chosen.append((ws, we))
                used_sec += wlen

    chosen = ensure_min_highlight(chosen, duration_sec, min_len=min_len)
    chosen.sort(key=lambda x: x[0])
    return [{"segment": [r2(s), r2(e)]} for s, e in chosen]


def fpvsum_candidates(fpv_data: dict, duration_sec: float, win_len: float = 6.0):
    imp = fpv_data.get("importance", {})
    ts = imp.get("timestamps_sec", [])
    sc = imp.get("scores", [])
    if not ts or not sc or len(ts) != len(sc):
        return []
    half = win_len / 2.0
    cands = []
    for t, s in zip(ts, sc):
        start = max(0.0, float(t) - half)
        end = min(duration_sec, start + win_len)
        start = max(0.0, end - win_len)
        if end - start >= 4.0:
            cands.append((start, end, float(s)))
    return cands


def build_video_json(video_id: str, duration_sec: float, dataset: str, team_id: str):
    actions = []
    action_file = actions_dir / f"{video_id}.json"
    if action_file.exists():
        with action_file.open("r", encoding="utf-8") as f:
            raw_actions = json.load(f)
        if isinstance(raw_actions, list):
            actions = raw_actions[:2000]

    if dataset == "epic_kitchens_100":
        label = {"name": "cooking", "score": 1.0}
        cands = [
            (a["segment"][0], a["segment"][1], a.get("score", 0.0))
            for a in actions
            if isinstance(a, dict) and isinstance(a.get("segment"), list) and len(a["segment"]) == 2
        ]
        highlights = select_non_overlapping(cands, duration_sec, min_len=4.0)
    else:
        fpv_file = fpvsum_dir / f"{video_id}.json"
        if fpv_file.exists():
            with fpv_file.open("r", encoding="utf-8") as f:
                fpv_data = json.load(f)
            label = fpv_data.get("label") or {"name": "other", "score": 0.5}
            cands = fpvsum_candidates(fpv_data, duration_sec, win_len=6.0)
        else:
            label = {"name": "other", "score": 0.0}
            cands = [(0.0, min(duration_sec, max(4.0, 0.15 * duration_sec)), 0.0)]
        highlights = select_non_overlapping(cands, duration_sec, min_len=4.0)

    return {
        "team_id": team_id,
        "video_id": video_id,
        "duration_sec": r2(duration_sec),
        "label": label,
        "actions": actions,
        "highlights": highlights,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-id", default="team_04")
    parser.add_argument("--video", help="Generate only one video_id (optional)")
    parser.add_argument("--out-dir", default=str(default_out_dir))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if args.video:
        rows = [r for r in rows if r["video_id"] == args.video]

    for r in rows:
        vid = r["video_id"]
        dur = float(r["duration_sec"])
        ds = r["dataset"]
        payload = build_video_json(vid, dur, ds, args.team_id)
        out_path = out_dir / f"{vid}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(f"Wrote {len(rows)} JSON file(s) to {out_dir}")


if __name__ == "__main__":
    main()
