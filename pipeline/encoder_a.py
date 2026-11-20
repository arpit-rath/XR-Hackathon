"""Encoder A: raw MP4 -> EPIC SlowFast features in the ActionFormer layout (T x 2304, float32).

Recipe (ActionFormer EPIC release + epic-kitchens-slowfast config):
  * decode at 30 fps, resize so the short side is 256 (EPIC frames are 456 x 256), RGB
  * clips of 32 frames every 16 frames; fast path = all 32, slow path = 8 (alpha = 4)
  * normalise (x/255 - 0.45) / 0.225
  * feature = global average pool of the last stage, slow (2048) ++ fast (256) = 2304-d
  * feature i covers frames [16 i, 16 i + 32) -> centred at (16 i + 16) / 30 s
Still unverified until the parity check: resolution / crop, channel order, the last partial clip.

    python encoder_a.py VIDEO.mp4 OUT.npz [--batch 16]
"""
import argparse
import json
import os
import subprocess
import sys
import types

import numpy as np
import torch

SLOWFAST_DIR = os.environ.get("SLOWFAST_DIR", os.path.expanduser("~/epic/epic-kitchens-slowfast"))
SLOWFAST_CKPT = os.environ.get("SLOWFAST_CKPT", os.path.expanduser("~/epic/SlowFast.pyth"))
FPS, CLIP, STRIDE, ALPHA, SHORT_SIDE = 30, 32, 16, 4, 256
MEAN, STD = 0.45, 0.225


def _import_slowfast():
    # slowfast's head_helper imports detectron2 only for the box head of detection models; stub it
    if "detectron2" not in sys.modules:
        d2 = types.ModuleType("detectron2")
        layers = types.ModuleType("detectron2.layers")
        layers.ROIAlign = None
        d2.layers = layers
        sys.modules["detectron2"], sys.modules["detectron2.layers"] = d2, layers
    if SLOWFAST_DIR not in sys.path:
        sys.path.insert(0, SLOWFAST_DIR)
    from slowfast.config.defaults import get_cfg
    from slowfast.models.video_model_builder import SlowFast
    return get_cfg, SlowFast


def load_encoder(device="cuda"):
    get_cfg, SlowFast = _import_slowfast()
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(SLOWFAST_DIR, "configs/EPIC-KITCHENS/SLOWFAST_8x8_R50.yaml"))
    cfg.NUM_GPUS = 1
    model = SlowFast(cfg)
    state = torch.load(SLOWFAST_CKPT, map_location="cpu", weights_only=False)
    state = state.get("model_state", state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing, f"missing weights: {missing[:5]}"
    return model.to(device).eval()


@torch.no_grad()
def pooled_features(model, fast):
    """fast: B x 3 x 32 x H x W (normalised). Returns B x 2304."""
    idx = torch.linspace(0, fast.shape[2] - 1, fast.shape[2] // ALPHA, device=fast.device).long()
    x = [fast.index_select(2, idx), fast]
    x = model.s1(x); x = model.s1_fuse(x)
    x = model.s2(x); x = model.s2_fuse(x)
    for p in range(2):
        x[p] = getattr(model, f"pathway{p}_pool")(x[p])
    x = model.s3(x); x = model.s3_fuse(x)
    x = model.s4(x); x = model.s4_fuse(x)
    x = model.s5(x)
    return torch.cat([x[0].mean([2, 3, 4]), x[1].mean([2, 3, 4])], 1)


def probe(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:format=duration", "-of", "json", video],
                         capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    s = info["streams"][0]
    return int(s["width"]), int(s["height"]), float(info["format"]["duration"])


def frames(video, width, height):
    """Yields RGB uint8 frames (H x W x 3) at 30 fps, short side 256, streamed (never the whole video)."""
    if width >= height:
        w, h = round(width * SHORT_SIDE / height / 2) * 2, SHORT_SIDE
    else:
        w, h = SHORT_SIDE, round(height * SHORT_SIDE / width / 2) * 2
    cmd = ["ffmpeg", "-v", "error", "-threads", "8", "-i", video, "-vf", f"fps={FPS},scale={w}:{h}",
           "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = w * h * 3
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    proc.wait()
    if proc.returncode:
        raise RuntimeError(f"ffmpeg failed on {video}")


@torch.no_grad()
def extract(video, model=None, batch=16, device="cuda"):
    model = model or load_encoder(device)
    width, height, duration = probe(video)
    buf, feats, clips = [], [], []
    start = 0

    def flush():
        if clips:
            x = torch.from_numpy(np.stack(clips)).to(device)          # B x 32 x H x W x 3
            x = x.permute(0, 4, 1, 2, 3).float().div_(255).sub_(MEAN).div_(STD)
            feats.append(pooled_features(model, x).float().cpu())
            clips.clear()

    for f in frames(video, width, height):
        buf.append(f)
        if len(buf) == CLIP:
            clips.append(np.stack(buf))
            buf = buf[STRIDE:]
            start += STRIDE
            if len(clips) == batch:
                flush()
    n_full = sum(len(f) for f in feats) + len(clips)
    if n_full == 0 and buf:  # video shorter than one clip: pad by repeating the last frame
        clips.append(np.stack(buf + [buf[-1]] * (CLIP - len(buf))))
    flush()
    out = torch.cat(feats).numpy().astype(np.float32) if feats else np.zeros((0, 2304), np.float32)
    return out, duration


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    feats, dur = extract(a.video, batch=a.batch)
    np.savez(a.out, feats=feats)
    print(f"{a.video}: {dur:.2f}s -> features {feats.shape}")
