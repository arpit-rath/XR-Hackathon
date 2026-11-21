"""Check (metadata only, no download) which FPVSum videos NOT in Samsung's manifest are still live.

These are candidate leakage-free training data. Writes reports/non_manifest_availability.csv.
"""
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
ORIG = WS / "original_fpvsum" / "fpvsum" / "crawler" / "FPVSum_videolist.csv"
MATCHED = WS / "reports" / "fpvsum_matched.csv"
GT_ONLY_NAME = {"Jumping_mU-p2wuk90"}  # gt spelling of a manifest video, not a separate video


def check(row):
    r = subprocess.run([sys.executable, "-m", "yt_dlp", "--skip-download", "--no-warnings",
                        "--print", "%(duration)s", f"https://www.youtube.com/watch?v={row['youtube_id']}"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = r.returncode == 0
    err = "" if ok else next((l for l in r.stderr.splitlines() if "ERROR" in l), r.stderr.strip()[-200:])
    return {**row, "available": ok, "yt_duration_sec": r.stdout.strip() if ok else "", "error": err[:200]}


def main():
    manifest = {r["samsung_id"] for r in csv.DictReader(open(MATCHED, encoding="utf-8"))}
    rows = [r for r in csv.DictReader(open(ORIG, encoding="utf-8")) if r["video_name"] not in manifest]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(check, rows))
    out = WS / "reports" / "non_manifest_availability.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)
    for split in ("labeled", "unlabeled"):
        s = [r for r in res if r["split"] == split]
        print(f"{split}: {sum(r['available'] for r in s)}/{len(s)} available")
    for r in res:
        if r["split"] == "labeled":
            print("  labeled:", r["video_name"], r["available"], r["yt_duration_sec"], r["error"][:90])
    print("wrote", out)


if __name__ == "__main__":
    main()
