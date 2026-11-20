"""Encoder A: raw MP4 -> EPIC SlowFast features in the ActionFormer layout (T x 2304, float32).

Recipe (ActionFormer EPIC release + epic-kitchens-slowfast config), settled by the parity check:
  * decode at 30 fps, resize so the short side is 256 (EPIC frames are 456 x 256), full frame, no crop
  * JPEG round trip at ffmpeg -q:v 4, because the released features were made from EPIC's JPEG frames
  * BGR channel order: the EPIC SlowFast loader reads frames with cv2.imread and never converts them
  * clips of 32 frames every 16 frames; fast path = all 32, slow path = 8 (alpha = 4)
  * normalise (x/255 - 0.45) / 0.225
  * feature = the model's own head pooling (AvgPool3d [T, 7, 7], stride 1) averaged over the
    remaining positions, slow (2048) ++ fast (256) = 2304-d
  * feature i covers frames [16 i, 16 i + 32) -> centred at (16 i + 16) / 30 s; clips keep starting
    every 16 frames up to the last frame, padded with the last frame (ceil(N / 16) features)
Parity on the first 2 min of P01_11 (median per-step cosine vs the released features): RGB 0.51,
BGR 0.93, + head pooling 0.955, + JPEG round trip 0.966.
The choices stay switchable for checks: ENCODER_BGR=1|0, ENCODER_POOL=head|mean, ENCODER_JPEG=4|0
(0 = no round trip), ENCODER_RESIZE=short256|456x256.
Batch size: ENCODER_BATCH, else 16 on GPUs with >= 12 GB and 4 below; a batch that runs out of
memory is split in half and retried, so small GPUs give the same features, only slower.

    python encoder_a.py VIDEO.mp4 OUT.npz [--batch N]
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
RESIZE = os.environ.get("ENCODER_RESIZE", "short256")
BGR = os.environ.get("ENCODER_BGR", "1") == "1"
POOL = os.environ.get("ENCODER_POOL", "head")
JPEG_Q = int(os.environ.get("ENCODER_JPEG", "4"))
SETTINGS = (f"resize={RESIZE} bgr={int(BGR)} pool={POOL} jpeg={JPEG_Q} fps={FPS} clip={CLIP} "
            f"stride={STRIDE} tail=pad")  # stored with cached features; a change invalidates the cache


def gpu_gb():
    try:
        return torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    except Exception:
        return 0.0


def default_batch():
    if os.environ.get("ENCODER_BATCH"):
        return int(os.environ["ENCODER_BATCH"])
    return 16 if gpu_gb() >= 12 else 4


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
    if POOL == "head":  # ResNetBasicHead: AvgPool3d [T, 7, 7] stride 1, then mean over what remains
        return torch.cat([torch.nn.functional.avg_pool3d(p, (p.shape[2], min(7, p.shape[3]), min(7, p.shape[4])), 1)
                          .mean([2, 3, 4]) for p in x], 1)
    return torch.cat([x[0].mean([2, 3, 4]), x[1].mean([2, 3, 4])], 1)


def safe_features(model, fast):
    """pooled_features, splitting the batch in half on CUDA out-of-memory (same result, smaller peaks)."""
    try:
        return pooled_features(model, fast)
    except torch.cuda.OutOfMemoryError:
        if fast.shape[0] == 1:
            raise
        torch.cuda.empty_cache()
        h = fast.shape[0] // 2
        return torch.cat([safe_features(model, fast[:h]), safe_features(model, fast[h:])])


def probe(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:format=duration", "-of", "json", video],
                         capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    s = info["streams"][0]
    return int(s["width"]), int(s["height"]), float(info["format"]["duration"])


def frames(video, width, height):
    """Yields RGB uint8 frames (H x W x 3) at 30 fps, short side 256, streamed (never the whole video)."""
    if RESIZE == "456x256" and width >= height:
        w, h = 456, 256  # EPIC's own frame release size
    elif width >= height:
        w, h = round(width * SHORT_SIDE / height / 2) * 2, SHORT_SIDE
    else:
        w, h = SHORT_SIDE, round(height * SHORT_SIDE / width / 2) * 2
    vf = ["-vf", f"fps={FPS},scale={w}:{h}"]
    raw = ["-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    procs = []
    if JPEG_Q:  # decode -> JPEG (like EPIC's frame release) -> decode again, streamed through a pipe
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-threads", "8", "-i", video, *vf, "-q:v", str(JPEG_Q),
                                "-c:v", "mjpeg", "-f", "image2pipe", "-"], stdout=subprocess.PIPE)
        proc = subprocess.Popen(["ffmpeg", "-v", "error", "-f", "mjpeg", "-i", "-", *raw],
                                stdin=enc.stdout, stdout=subprocess.PIPE)
        enc.stdout.close()  # so enc gets SIGPIPE if the reader stops
        procs = [enc, proc]
    else:
        proc = subprocess.Popen(["ffmpeg", "-v", "error", "-threads", "8", "-i", video, *vf, *raw],
                                stdout=subprocess.PIPE)
        procs = [proc]
    size = w * h * 3
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    for p in procs:
        p.wait()
    if any(p.returncode for p in procs):
        raise RuntimeError(f"ffmpeg failed on {video}")


@torch.no_grad()
def extract(video, model=None, batch=None, device="cuda"):
    model = model or load_encoder(device)
    batch = batch or default_batch()
    width, height, duration = probe(video)
    buf, feats, clips = [], [], []
    start = 0

    def flush():
        if clips:
            x = torch.from_numpy(np.stack(clips)).to(device)          # B x 32 x H x W x 3 (RGB)
            if BGR:
                x = x.flip(-1)
            x = x.permute(0, 4, 1, 2, 3).float().div_(255).sub_(MEAN).div_(STD)
            feats.append(safe_features(model, x).float().cpu())
            clips.clear()

    for f in frames(video, width, height):
        buf.append(f)
        if len(buf) == CLIP:
            clips.append(np.stack(buf))
            buf = buf[STRIDE:]
            start += STRIDE
            if len(clips) == batch:
                flush()
    # Tail: keep starting a clip every 16 frames while the start is inside the video, padding with the
    # last frame, so a video of N frames gives ceil(N / 16) features like the released ones
    # (P01_11: 16,846 frames -> 1,053 steps).
    while buf:
        clips.append(np.stack(buf + [buf[-1]] * (CLIP - len(buf))))
        buf = buf[STRIDE:]
        if len(clips) == batch:
            flush()
    flush()
    out = torch.cat(feats).numpy().astype(np.float32) if feats else np.zeros((0, 2304), np.float32)
    return out, duration


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--batch", type=int, default=None)
    a = ap.parse_args()
    feats, dur = extract(a.video, batch=a.batch)
    np.savez(a.out, feats=feats, duration=dur, settings=SETTINGS)
    print(f"{a.video}: {dur:.2f}s -> features {feats.shape}")
