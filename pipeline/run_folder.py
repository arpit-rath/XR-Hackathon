"""Batch version of run_video.py: the same build() for every video given, models loaded once.
This is how outputs/ was produced (with --cache-dir pointing at features the same code saved earlier).

    python run_folder.py OUT_DIR VIDEO [VIDEO ...] [--team-id team_04] [--cache-dir DIR] [--redo]
"""
import argparse
import json
import logging
import os
import time
import traceback

import run_video as rv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--team-id", default=os.environ.get("TEAM_ID", "team_04"))
    ap.add_argument("--cache-dir")
    ap.add_argument("--redo", action="store_true", help="rebuild files that already exist")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(a.out_dir, exist_ok=True)
    failed, t_all = [], time.time()
    for i, v in enumerate(a.videos, 1):
        vid = os.path.splitext(os.path.basename(v))[0]
        out = os.path.join(os.path.abspath(a.out_dir), f"{vid}.json")
        if os.path.exists(out) and not a.redo:
            continue
        t0 = time.time()
        try:
            result = rv.build(os.path.abspath(v), a.team_id, a.cache_dir)
        except Exception:
            rv.log.error("%s failed:\n%s", vid, traceback.format_exc())
            failed.append(vid)
            continue
        with open(out + ".tmp", "w") as f:
            json.dump(result, f, indent=1)
        os.replace(out + ".tmp", out)
        errors = rv.check(out)
        if errors:
            failed.append(vid)
            rv.log.error("%s fails validation: %s", vid, errors)
        print(f"[{i}/{len(a.videos)}] {vid}: {result['label']['name']} ({result['label']['score']}), "
              f"{len(result['actions'])} actions, {len(result['highlights'])} highlights, {time.time() - t0:.0f}s",
              flush=True)
    print(f"DONE {len(a.videos)} videos in {(time.time() - t_all) / 60:.1f} min; failed: {failed}")


if __name__ == "__main__":
    main()
