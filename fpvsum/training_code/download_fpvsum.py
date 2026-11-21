"""Resumable downloader + verifier for the Samsung XR Hackathon FPVSum videos.

Reads reports/fpvsum_matched.csv (built by match_fpvsum.py from Samsung's manifest),
downloads ONLY those videos with yt-dlp (<=720p, MP4), and verifies each with ffprobe.
Existing files that pass verification are skipped, so the script can be re-run any time.

Usage (from anywhere; use the workspace venv's python):
    .venv\\Scripts\\python.exe scripts\\download_fpvsum.py                 # all videos
    .venv\\Scripts\\python.exe scripts\\download_fpvsum.py --only Ski_-Z5a4UQmZLs
    .venv\\Scripts\\python.exe scripts\\download_fpvsum.py --verify-only    # no downloads
Options: --limit N, --no-decode (skip full decode check), --tolerance SEC (default 0.5)
"""
import argparse
import csv
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
MATCHED = WS / "reports" / "fpvsum_matched.csv"
VIDEOS = WS / "videos"
REPORTS = WS / "reports"
LOG = WS / "logs" / "download.log"


def find_tool(name):
    local = list((WS / "tools").glob(f"**/bin/{name}.exe"))
    if local:
        return str(local[0])
    p = shutil.which(name)
    if p:
        return p
    sys.exit(f"{name} not found (expected under tools/ or on PATH)")


FFMPEG, FFPROBE = find_tool("ffmpeg"), find_tool("ffprobe")
# Prefer H.264 (avc1) for broad decoder support (OpenCV/decord/PyAV); fall back to anything <=720p.
FORMAT = ("bv*[height<=720][vcodec^=avc1]+ba[ext=m4a]/b[height<=720][vcodec^=avc1]/"
          "bv*[height<=720][ext=mp4]+ba[ext=m4a]/bv*[height<=720]+ba/b[height<=720]/b")


def log(msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def probe(path):
    r = subprocess.run([FFPROBE, "-v", "error", "-print_format", "json", "-show_format",
                        "-show_streams", str(path)], capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr.strip()[:300]
    d = json.loads(r.stdout)
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in d.get("streams", []) if s.get("codec_type") == "audio"), None)
    if v is None:
        return None, "no video stream"
    num, _, den = v.get("avg_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den) if float(den or 0) else 0.0
    return {
        "duration": float(d["format"].get("duration", 0)),
        "width": v.get("width"), "height": v.get("height"),
        "fps": round(fps, 3), "vcodec": v.get("codec_name"),
        "acodec": a.get("codec_name") if a else "",
    }, ""


def decode_ok(path):
    r = subprocess.run([FFMPEG, "-v", "error", "-xerror", "-i", str(path), "-map", "0:v:0",
                        "-f", "null", "-"], capture_output=True, text=True)
    return r.returncode == 0, r.stderr.strip()[:300]


def download(row, retries=3):
    sid, url = row["samsung_id"], row["source_url"]
    out = VIDEOS / f"{sid}.%(ext)s"
    cmd = [sys.executable, "-m", "yt_dlp", "-f", FORMAT, "--merge-output-format", "mp4",
           "--ffmpeg-location", str(Path(FFMPEG).parent), "--no-playlist", "--continue",
           "--retries", "10", "--fragment-retries", "10", "--no-progress",
           "--no-mtime", "-o", str(out), url]
    err = ""
    for attempt in range(1, retries + 1):
        log(f"[{sid}] download attempt {attempt}: {url}")
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode == 0:
            return True, ""
        err = " | ".join(l for l in r.stderr.splitlines() if "ERROR" in l) or r.stderr.strip()[-300:]
        log(f"[{sid}] failed: {err}")
        permanent = any(k in err.lower() for k in (
            "private", "unavailable", "removed", "terminated", "copyright", "sign in",
            "members-only", "not available", "blocked"))
        if permanent:
            break
        time.sleep(10 * attempt)
    return False, err


def verify(row, path, tol, do_decode):
    info, err = probe(path)
    res = {"actual_duration_sec": "", "resolution": "", "fps": "", "video_codec": "",
           "audio_codec": "", "duration_diff_sec": ""}
    if info is None:
        return "FAIL_PROBE", err, res
    res.update(actual_duration_sec=round(info["duration"], 3),
               resolution=f'{info["width"]}x{info["height"]}', fps=info["fps"],
               video_codec=info["vcodec"], audio_codec=info["acodec"])
    if do_decode:
        ok, derr = decode_ok(path)
        if not ok:
            return "FAIL_DECODE", derr, res
    exp = row.get("expected_duration_sec")
    if exp:
        diff = info["duration"] - float(exp)
        res["duration_diff_sec"] = round(diff, 3)
        if abs(diff) > tol:
            return "DURATION_MISMATCH", f"actual-expected={diff:+.2f}s (tol {tol}s)", res
    return "PASS", "", res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="Samsung IDs to process")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--no-decode", action="store_true")
    ap.add_argument("--tolerance", type=float, default=0.5)
    a = ap.parse_args()

    with open(MATCHED, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    targets = [r for r in rows if r["match_status"].startswith("matched")]
    if a.only:
        targets = [r for r in targets if r["samsung_id"] in set(a.only)]
    if a.limit:
        targets = targets[: a.limit]
    VIDEOS.mkdir(exist_ok=True)

    results = {}
    for r in targets:
        sid = r["samsung_id"]
        path = VIDEOS / f"{sid}.mp4"
        status, err = "", ""
        if path.exists() and path.stat().st_size > 0:
            status = "existing"
        elif a.verify_only:
            status = "not_downloaded"
        else:
            ok, err = download(r)
            status = "downloaded" if ok and path.exists() else "failed"
            if ok and not path.exists():
                others = [p.name for p in VIDEOS.glob(f"{sid}.*")]
                err = f"yt-dlp succeeded but {path.name} missing; found {others}"
        ver, verr, res = ("NOT_VERIFIED", "", {})
        if path.exists() and path.stat().st_size > 0:
            ver, verr, res = verify(r, path, a.tolerance, not a.no_decode)
        log(f"[{sid}] status={status} verify={ver} {verr}")
        results[sid] = {
            "samsung_id": sid, "youtube_id": r["youtube_id"], "category": r["category_label"],
            "download_status": status,
            "file_path": str(path) if path.exists() else "",
            "file_size_mb": round(path.stat().st_size / 2**20, 2) if path.exists() else "",
            **{k: res.get(k, "") for k in ("actual_duration_sec",)},
            "expected_duration_sec": r["expected_duration_sec"],
            **{k: res.get(k, "") for k in ("duration_diff_sec", "resolution", "fps",
                                             "video_codec", "audio_codec")},
            "verification_result": ver, "error_message": err or verr,
        }

    # Merge with any previous report so partial runs keep a complete picture.
    report = REPORTS / "fpvsum_download_report.csv"
    merged = {}
    if report.exists():
        with open(report, newline="", encoding="utf-8") as f:
            merged = {x["samsung_id"]: x for x in csv.DictReader(f)}
    merged.update(results)
    order = [r["samsung_id"] for r in rows if r["samsung_id"] in merged]
    fields = list(next(iter(results.values())).keys()) if results else list(next(iter(merged.values())).keys())
    with open(report, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(merged[s] for s in order)
    failed = [merged[s] for s in order if merged[s]["verification_result"] != "PASS"]
    with open(REPORTS / "failed_downloads.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(failed)
    n_pass = sum(merged[s]["verification_result"] == "PASS" for s in order)
    log(f"SUMMARY: processed={len(results)} report_rows={len(order)} pass={n_pass} not_pass={len(failed)}")


if __name__ == "__main__":
    main()
