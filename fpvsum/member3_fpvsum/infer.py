"""Member 3 inference: video -> timestamp-aligned importance scores + category + confidence.

usage:
    python -m member3_fpvsum.infer <video.mp4> <out.json> [--video-id ID] [--weights DIR] [--device cpu|cuda]
    python -m member3_fpvsum.infer --features <feats.npz> <out.json> ...   (re-use cached features)

Output JSON (consumed by Member 4 for highlight selection; no highlights are produced here):
{
  "video_id", "duration_sec", "module": "member3_fpvsum", "version",
  "feature_fps": 2.0,
  "importance": {"timestamps_sec": [...], "scores": [...], "method": "clip_zeroshot_v1"},
  "label": {"name": <one of 14 FPVSum categories>, "score": <confidence 0-1>},
  "category_probs": {label: prob, ...}
}
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import __version__
from .features import ClipVisual, DEFAULT_FPS
from .importance import importance_scores
from .models import CategoryHead, pool_features

PKG_ROOT = Path(__file__).resolve().parent.parent


class Member3Model:
    def __init__(self, weights_dir=None, device=None, load_visual=True):
        wd = Path(weights_dir) if weights_dir else PKG_ROOT / "weights"
        self.cfg = json.load(open(wd / "config.json"))
        te = torch.load(wd / "text_embeddings.pt", weights_only=True)
        self.labels = te["labels"]
        self.text_cat = te["category"].float()
        self.imp_pos = te["importance_pos"].float().numpy()
        self.imp_neg = te["importance_neg"].float().numpy()
        ck = torch.load(wd / "category_head.pt", weights_only=False)
        self.head = CategoryHead(**ck["cfg"])
        self.head.load_state_dict(ck["state_dict"])
        self.head.eval()
        self.alpha = self.cfg["category"]["alpha_probe"]
        self.temp = self.cfg["category"]["temperature_zs"]
        self.clip = ClipVisual(weights=wd / "clip_vit_l14_visual_fp16.pt", device=device) if load_visual else None

    @torch.no_grad()
    def classify(self, feats):
        x = torch.as_tensor(np.asarray(feats, np.float32))
        p_probe = F.softmax(self.head(pool_features(x)), -1)
        p_zs = F.softmax(self.temp * (F.normalize(x.mean(0), dim=-1) @ self.text_cat.T), -1)
        return (self.alpha * p_probe + (1 - self.alpha) * p_zs).numpy()

    def predict_features(self, feats, duration, video_id):
        fps = self.cfg["feature_fps"]
        probs = self.classify(feats)
        c = int(probs.argmax())
        scores, _ = importance_scores(feats, self.imp_pos[c], self.imp_neg, fps=fps,
                                      weights=self.cfg["importance"]["weights"],
                                      smooth_sec=self.cfg["importance"]["smooth_sec"],
                                      dyn_window_sec=self.cfg["importance"]["dyn_window_sec"])
        ts = np.arange(len(scores)) / fps
        return {
            "video_id": video_id,
            "duration_sec": round(float(duration), 3),
            "module": "member3_fpvsum",
            "version": __version__,
            "feature_fps": fps,
            "importance": {"timestamps_sec": [round(float(t), 2) for t in ts],
                           "scores": [round(float(s), 4) for s in scores],
                           "method": self.cfg["importance"]["method"],
                           "timestamp_convention": "score i describes the frame sampled at t = i / feature_fps seconds"},
            "label": {"name": self.labels[c], "score": round(float(probs[c]), 4)},
            "category_probs": {l: round(float(p), 4) for l, p in zip(self.labels, probs)},
        }

    def predict_video(self, video, video_id=None):
        feats, _, duration = self.clip.video_features(video, fps=self.cfg["feature_fps"])
        return self.predict_features(feats, duration, video_id or Path(video).stem), feats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?")
    ap.add_argument("out_json")
    ap.add_argument("--features", help="cached .npz with feats/duration instead of a video")
    ap.add_argument("--video-id")
    ap.add_argument("--weights")
    ap.add_argument("--device")
    ap.add_argument("--save-features", help="also save CLIP features to this .npz")
    a = ap.parse_args()
    t0 = time.time()
    m = Member3Model(a.weights, a.device, load_visual=a.features is None)
    if a.features:
        d = np.load(a.features)
        out = m.predict_features(d["feats"].astype(np.float32), float(d["duration"]),
                                 a.video_id or Path(a.features).stem)
    else:
        out, feats = m.predict_video(a.video, a.video_id)
        if a.save_features:
            np.savez(a.save_features, feats=feats.astype(np.float16), duration=out["duration_sec"])
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out_json, "w"), indent=1)
    print(f'{out["video_id"]}: label={out["label"]["name"]} ({out["label"]["score"]:.3f}), '
          f'{len(out["importance"]["scores"])} scores @ {out["feature_fps"]} fps, {time.time()-t0:.1f}s -> {a.out_json}')


if __name__ == "__main__":
    main()
