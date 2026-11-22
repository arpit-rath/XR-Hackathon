"""Video decoding + frozen CLIP ViT-L/14 (OpenAI) image features at a fixed sampling rate.

The same code path is used for training-feature extraction and for inference, so preprocessing
is identical:  ffmpeg -> fps=2 -> bicubic resize (short side 224) -> center crop 224x224 -> RGB
-> CLIP mean/std normalisation -> visual encoder -> L2-normalised 768-d embedding.

Timestamp convention: feature k corresponds to video time t_k = k / fps seconds (the ffmpeg
`fps` filter emits the frame nearest to each k/fps instant).
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

CLIP_ARCH = "ViT-L-14-quickgelu"  # OpenAI ViT-L/14 weights were trained with QuickGELU
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
SIZE = 224
DEFAULT_FPS = 2.0


def find_ffmpeg():
    """ffmpeg from $FFMPEG, PATH, or the imageio-ffmpeg pip package (in that order)."""
    env = os.environ.get("FFMPEG")
    if env and Path(env).exists():
        return env
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover
        raise RuntimeError("ffmpeg not found: install ffmpeg, set $FFMPEG, or `pip install imageio-ffmpeg`") from e


def probe_duration(video, ffmpeg=None):
    """Container duration in seconds, parsed from `ffmpeg -i` (no ffprobe needed)."""
    ffmpeg = ffmpeg or find_ffmpeg()
    r = subprocess.run([ffmpeg, "-hide_banner", "-i", str(video)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr)
    if not m:
        raise RuntimeError(f"could not read duration of {video}: {r.stderr[-300:]}")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def iter_frames(video, fps=DEFAULT_FPS, ffmpeg=None, max_sec=None):
    """Yield uint8 RGB frames (224x224x3) sampled at `fps` (only the first `max_sec` seconds if given)."""
    ffmpeg = ffmpeg or find_ffmpeg()
    vf = (f"fps={fps},"
          f"scale='if(gt(iw,ih),-2,{SIZE})':'if(gt(iw,ih),{SIZE},-2)':flags=bicubic,"
          f"crop={SIZE}:{SIZE}")
    limit = ["-t", str(max_sec)] if max_sec else []
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-i", str(video), *limit, "-an", "-vf", vf,
           "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10 ** 7)
    n = SIZE * SIZE * 3
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(SIZE, SIZE, 3)
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        proc.wait()
        if proc.returncode not in (0, None):
            raise RuntimeError(f"ffmpeg failed on {video}: {err[-300:]}")


class ClipVisual:
    """Frozen CLIP ViT-L/14 visual tower. Loads OpenAI weights (training) or a local file (inference)."""

    def __init__(self, weights=None, device=None, pretrained_tag="openai"):
        import open_clip
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if weights:
            model = open_clip.create_model(CLIP_ARCH, pretrained=None)
            sd = torch.load(weights, map_location="cpu", weights_only=True)
            model.visual.load_state_dict({k: v.float() for k, v in sd.items()})
        else:
            model = open_clip.create_model(CLIP_ARCH, pretrained=pretrained_tag)
        self.model = model
        self.visual = model.visual.to(self.device).eval()
        self.half = self.device.type == "cuda"
        if self.half:
            self.visual.half()
        self.mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode(self, frames_uint8):
        x = torch.from_numpy(np.stack(frames_uint8)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        x = (x - self.mean) / self.std
        if self.half:
            x = x.half()
        f = self.visual(x).float()
        return torch.nn.functional.normalize(f, dim=-1).cpu().numpy()

    def video_features(self, video, fps=DEFAULT_FPS, batch=64, ffmpeg=None, max_sec=None):
        """Return (features [T,768] float32, timestamps [T] seconds, duration_sec).
        max_sec: encode only the first max_sec seconds (duration is still the whole video's)."""
        ffmpeg = ffmpeg or find_ffmpeg()
        duration = probe_duration(video, ffmpeg)
        feats, buf = [], []
        for fr in iter_frames(video, fps, ffmpeg, max_sec):
            buf.append(fr)
            if len(buf) == batch:
                feats.append(self.encode(buf))
                buf = []
        if buf:
            feats.append(self.encode(buf))
        if not feats:
            raise RuntimeError(f"no frames decoded from {video}")
        F = np.concatenate(feats).astype(np.float32)
        ts = np.arange(len(F), dtype=np.float64) / fps
        return F, ts, duration
