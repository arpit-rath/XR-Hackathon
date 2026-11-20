"""Batch run of the action detector over every EPIC video in manifest.csv. Resumable.

For each EPIC row in the manifest: find (or download) the MP4, extract Encoder A features from it,
run the released verb and noun models plus fusion, and write
    OUT/actions/<video_id>.json   action records (same list detect_actions.py writes) for Member 4
    OUT/features/<video_id>.npz   Encoder A features + duration + preprocessing settings (cache only)
    OUT/status.tsv                one line per attempt: id, status, duration, steps, records, seconds, note
The encoder and both detectors load once. Finished videos are skipped on a rerun, and features are
reused when they were made with the same preprocessing settings. A failed video is logged and
retried on the next run; it never stops the batch. Files are written atomically.

With --download, a missing MP4 is fetched with the official EPIC downloader (the next video is
fetched while the current one runs) and deleted after processing, so at most two sit on disk.

    python pipeline/run_manifest.py --videos ~/epic/videos --out ~/epic/manifest_run [--download]
                                    [--only P01_11,P02_12] [--limit N] [--keep-videos]

The FPVSum rows are skipped: their actions may be [] and are not scored.
"""
import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import detect_actions as da  # noqa: E402
import encoder_a  # noqa: E402

log = logging.getLogger("run_manifest")
DOWNLOADER = os.environ.get("EPIC_DOWNLOADER", os.path.expanduser("~/epic/epic-kitchens-download-scripts"))


def manifest_ids(path):
    with open(path, newline="") as f:
        return [(r["video_id"], float(r["duration_sec"])) for r in csv.DictReader(f)
                if r["dataset"] == "epic_kitchens_100"]


def find_video(root, vid):
    for d, _, files in os.walk(root):
        for name in files:
            stem, ext = os.path.splitext(name)
            if stem == vid and ext.lower() == ".mp4":
                return os.path.join(d, name)
    return None


def fetch(root, vid, download):
    """(path, downloaded_by_us). Raises if the video is missing and cannot be downloaded."""
    path = find_video(root, vid)
    if path:
        return path, False
    if not download:
        raise FileNotFoundError(f"{vid}.MP4 not found under {root} (use --download)")
    cmd = [sys.executable, "epic_downloader.py", "--videos", "--specific-videos", vid, "--output-path", root]
    r = subprocess.run(cmd, cwd=DOWNLOADER, capture_output=True, text=True)
    path = find_video(root, vid)
    if not path:
        raise RuntimeError(f"download of {vid} failed (exit {r.returncode}): {(r.stderr or r.stdout)[-500:]}")
    return path, True


def save_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


def save_features(path, feats, duration):
    with open(path + ".tmp", "wb") as f:
        np.savez(f, feats=feats, duration=duration, settings=encoder_a.SETTINGS)
    os.replace(path + ".tmp", path)


def load_cached(path):
    """(feats, duration) if a complete cache entry made with the current settings exists, else None."""
    try:
        with np.load(path) as z:
            if "settings" in z.files and str(z["settings"]) == encoder_a.SETTINGS:
                return z["feats"], float(z["duration"])
    except (OSError, ValueError, KeyError):
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=os.path.join(REPO, "package", "manifest.csv"))
    ap.add_argument("--videos", required=True, help="folder holding (or receiving) the EPIC MP4s; searched recursively")
    ap.add_argument("--out", required=True)
    ap.add_argument("--download", action="store_true", help="fetch missing MP4s with the official downloader")
    ap.add_argument("--keep-videos", action="store_true", help="keep MP4s this script downloaded")
    ap.add_argument("--only", default="", help="comma-separated video ids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--redo", action="store_true", help="recompute videos that already have actions")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # absolute paths first: load_detectors() changes the working directory
    videos, out = os.path.abspath(os.path.expanduser(a.videos)), os.path.abspath(os.path.expanduser(a.out))
    act_dir, feat_dir = os.path.join(out, "actions"), os.path.join(out, "features")
    for d in (videos, act_dir, feat_dir):
        os.makedirs(d, exist_ok=True)
    status_path = os.path.join(out, "status.tsv")
    if not os.path.exists(status_path):
        with open(status_path, "w") as f:
            f.write("time\tvideo_id\tstatus\tduration\tsteps\trecords\tseconds\tnote\n")

    rows = manifest_ids(os.path.abspath(a.manifest))
    if a.only:
        wanted = [v.strip() for v in a.only.split(",") if v.strip()]
        known = dict(rows)
        rows = [(v, known[v]) for v in wanted if v in known]
        missing = [v for v in wanted if v not in known]
        if missing:
            log.warning("not EPIC ids in the manifest, skipped: %s", ",".join(missing))
    todo = [(v, d) for v, d in rows if a.redo or not os.path.exists(os.path.join(act_dir, f"{v}.json"))]
    if a.limit:
        todo = todo[:a.limit]
    log.info("%d EPIC videos in scope, %d to do; encoder settings: %s", len(rows), len(todo), encoder_a.SETTINGS)
    if not todo:
        return

    verbs, nouns = da.class_names()
    encoder = encoder_a.load_encoder()
    models = da.load_detectors()

    def note(vid, status, dur="", steps="", n="", secs="", msg=""):
        with open(status_path, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{vid}\t{status}\t{dur}\t{steps}\t{n}\t{secs}\t{msg}\n")

    pool = ThreadPoolExecutor(1)
    feat_path = lambda v: os.path.join(feat_dir, f"{v}.npz")  # noqa: E731
    # only videos without usable cached features need their MP4
    submit = lambda v: None if load_cached(feat_path(v)) else pool.submit(fetch, videos, v, a.download)  # noqa: E731
    nxt = submit(todo[0][0])
    done = failed = 0
    for i, (vid, man_dur) in enumerate(todo):
        cur = nxt
        nxt = submit(todo[i + 1][0]) if i + 1 < len(todo) else None
        t0, path, ours = time.time(), None, False
        try:
            cached = load_cached(feat_path(vid)) if cur is None else None
            if cached:
                feats, duration = cached
            else:
                path, ours = cur.result()
                feats, duration = encoder_a.extract(path, model=encoder)
                save_features(feat_path(vid), feats, duration)
            if abs(duration - man_dur) > 0.5:
                log.warning("%s: decoded duration %.2fs vs manifest %.2fs", vid, duration, man_dur)
            records = da.to_records(da.detect(models, feats, duration, vid), duration, verbs, nouns)
            da.check_contract(records, duration, verbs, nouns)
            save_json(os.path.join(act_dir, f"{vid}.json"), records)
            secs = time.time() - t0
            note(vid, "ok", f"{duration:.2f}", len(feats), len(records), f"{secs:.0f}")
            done += 1
            log.info("[%d/%d] %s: %.0fs video, %d steps, %d records in %.0fs",
                     i + 1, len(todo), vid, duration, len(feats), len(records), secs)
        except Exception as e:
            failed += 1
            note(vid, "failed", msg=repr(e)[:300].replace("\t", " ").replace("\n", " "))
            log.exception("[%d/%d] %s failed; continuing", i + 1, len(todo), vid)
        finally:
            if ours and path and not a.keep_videos and os.path.exists(path):
                os.remove(path)
            torch.cuda.empty_cache()
    pool.shutdown()
    left = sum(not os.path.exists(os.path.join(act_dir, f"{v}.json")) for v, _ in rows)
    log.info("finished: %d ok, %d failed this run; %d of %d EPIC videos still without actions",
             done, failed, left, len(rows))


if __name__ == "__main__":
    main()
