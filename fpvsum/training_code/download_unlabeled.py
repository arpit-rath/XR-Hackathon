"""Download the live, NON-manifest, unlabeled FPVSum videos (category-classifier training data).

Source list: reports/non_manifest_availability.csv (split=unlabeled, available=True).
Output: training/data/fpvsum_unlabeled/<video_name>.mp4 and reports/unlabeled_download_report.csv
These videos are NOT in Samsung's manifest. Re-runnable: existing valid files are skipped.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_fpvsum as d  # noqa: E402  (reuses yt-dlp format, retries, ffprobe verification)

WS = d.WS
OUT = WS / "training" / "data" / "fpvsum_unlabeled"


def main():
    rows = [r for r in csv.DictReader(open(WS / "reports" / "non_manifest_availability.csv", encoding="utf-8"))
            if r["split"] == "unlabeled" and r["available"] == "True"]
    d.VIDEOS = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    res = []
    for r in rows:
        row = {"samsung_id": r["video_name"], "source_url": f"https://www.youtube.com/watch?v={r['youtube_id']}",
               "expected_duration_sec": ""}
        path = OUT / f"{r['video_name']}.mp4"
        err = ""
        if not (path.exists() and path.stat().st_size > 0):
            ok, err = d.download(row)
        ver, verr, info = ("NOT_VERIFIED", "", {})
        if path.exists() and path.stat().st_size > 0:
            ver, verr, info = d.verify(row, path, 0.5, do_decode=False)
        d.log(f"[unlabeled {r['video_name']}] verify={ver} {err or verr}")
        res.append({"video_name": r["video_name"], "youtube_id": r["youtube_id"], "category": r["category"].lower(),
                    "file_path": str(path) if path.exists() else "", "verification_result": ver,
                    "duration_sec": info.get("actual_duration_sec", ""), "resolution": info.get("resolution", ""),
                    "fps": info.get("fps", ""), "error_message": err or verr})
    with open(WS / "reports" / "unlabeled_download_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)
    print("PASS:", sum(x["verification_result"] == "PASS" for x in res), "/", len(res))


if __name__ == "__main__":
    main()
