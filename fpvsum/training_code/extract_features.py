"""Extract 2-fps CLIP ViT-L/14 features for a set of videos (resumable).

usage: python extract_features.py <dataset_name> <video_dir_or_glob> [--workers 4]
writes training/features/<dataset_name>/<video_stem>.npz  {feats fp16 [T,768], ts [T], duration}
"""
import argparse
import glob
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS / "export" / "Member3_FPVSum_Trained_Model"))
from member3_fpvsum.features import ClipVisual, iter_frames, probe_duration, find_ffmpeg, DEFAULT_FPS  # noqa

LOCK = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("source")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    src = Path(a.source)
    vids = sorted(src.glob("*.mp4")) + sorted(src.glob("*.webm")) + sorted(src.glob("*.avi")) \
        if src.is_dir() else sorted(Path(p) for p in glob.glob(a.source))
    out = WS / "training" / "features" / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    todo = [v for v in vids if not (out / f"{v.stem}.npz").exists()]
    print(f"[{a.dataset}] {len(vids)} videos, {len(todo)} to extract", flush=True)
    enc = ClipVisual()
    ff = find_ffmpeg()

    def work(v):
        t0 = time.time()
        try:
            dur = probe_duration(v, ff)
            feats, buf = [], []
            for fr in iter_frames(v, DEFAULT_FPS, ff):
                buf.append(fr)
                if len(buf) == 64:
                    with LOCK:
                        feats.append(enc.encode(buf))
                    buf = []
            if buf:
                with LOCK:
                    feats.append(enc.encode(buf))
            F = np.concatenate(feats)
            tmp = out / f"{v.stem}.tmp.npz"
            np.savez(tmp, feats=F.astype(np.float16), ts=np.arange(len(F)) / DEFAULT_FPS, duration=dur)
            tmp.replace(out / f"{v.stem}.npz")
            print(f"[{a.dataset}] {v.stem}: T={len(F)} dur={dur:.2f}s ({time.time()-t0:.1f}s)", flush=True)
        except Exception:
            print(f"[{a.dataset}] FAILED {v.name}\n{traceback.format_exc()}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))
    print(f"[{a.dataset}] done", flush=True)


if __name__ == "__main__":
    main()
