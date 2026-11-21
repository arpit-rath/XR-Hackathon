"""Match Samsung manifest FPVSum IDs against the original FPVSum video list.

Matching is by exact `video_name` (the Samsung ID). The YouTube ID is taken from the
original `youtube_id` column -- never parsed from the name (BikePolo names do not
contain their real YouTube ID). A case-insensitive fallback is reported as such.
"""
import csv
import sys
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
MANIFEST = WS / "samsung_kit" / "manifest.csv"
ORIG = WS / "original_fpvsum" / "fpvsum" / "crawler" / "FPVSum_videolist.csv"
OUT = WS / "reports" / "fpvsum_matched.csv"


def main():
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        manifest = [r for r in csv.DictReader(f) if r["dataset"].strip() == "fpvsum"]
    with open(ORIG, newline="", encoding="utf-8") as f:
        orig = list(csv.DictReader(f))

    exact = {r["video_name"]: r for r in orig}
    lower = {}
    for r in orig:
        lower.setdefault(r["video_name"].lower(), []).append(r)

    ids = [r["video_id"].strip() for r in manifest]
    dupes = {i for i in ids if ids.count(i) > 1}

    rows = []
    for m in manifest:
        sid = m["video_id"].strip()
        o, status = exact.get(sid), "matched_exact"
        if o is None:
            cands = lower.get(sid.lower(), [])
            if len(cands) == 1:
                o, status = cands[0], "matched_case_insensitive"
            elif len(cands) > 1:
                status = "ambiguous"
            else:
                status = "unmatched"
        if sid in dupes:
            status += ";duplicate_in_manifest"
        yt = o["youtube_id"] if o else ""
        rows.append({
            "samsung_id": sid,
            "expected_duration_sec": m["duration_sec"],
            "original_video_name": o["video_name"] if o else "",
            "youtube_id": yt,
            "category": o["category"] if o else "",
            "category_label": o["category"].lower() if o else "",
            "split": o["split"] if o else "",
            "source_url": f"https://www.youtube.com/watch?v={yt}" if yt else "",
            "match_status": status,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_unique = len(set(ids))
    matched = [r for r in rows if r["match_status"].startswith("matched")]
    print(f"Manifest FPVSum rows: {len(manifest)}  unique IDs: {n_unique}")
    print(f"Matched: {len(matched)}  (exact: {sum(r['match_status']=='matched_exact' for r in rows)})")
    print("Splits of matched:", {s: sum(r['split'] == s for r in matched) for s in {r['split'] for r in matched}})
    for r in rows:
        if not r["match_status"].startswith("matched_exact"):
            print("  NON-EXACT:", r["samsung_id"], r["match_status"])
    for r in rows:
        if r["youtube_id"] and r["youtube_id"] not in r["samsung_id"]:
            print("  NOTE id-not-in-name:", r["samsung_id"], "->", r["youtube_id"])
    print("Wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
