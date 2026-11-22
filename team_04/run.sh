#!/usr/bin/env bash
# Track 04 single-video entry point:  ./run.sh <video.mp4> <out.json>
# Everything is computed from the video (code/run_video.py). Needs the environment in requirements.txt,
# ffmpeg/ffprobe on PATH and an NVIDIA GPU (tested on a 6 GB RTX 4050 and a 48 GB RTX 6000 Ada).
set -euo pipefail
if [ $# -ne 2 ]; then
  echo "Usage: ./run.sh <video.mp4> <out.json>" >&2
  exit 2
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

export ACTIONFORMER_DIR="$HERE/code/third_party/actionformer_release"
export SLOWFAST_DIR="$HERE/code/third_party/epic-kitchens-slowfast"
export SLOWFAST_CKPT="$HERE/weights/SlowFast.pyth"
export VERB_CKPT="$HERE/weights/actionformer_epic_verb_epoch_020.pth.tar"
export NOUN_CKPT="$HERE/weights/actionformer_epic_noun_epoch_020.pth.tar"
export SUPPORTED_CLASSES="$HERE/code/supported_classes.json"
export M3_DIR="$HERE/code"
export M3_WEIGHTS="$HERE/weights"
export HF_HUB_OFFLINE=1   # every weight is local; nothing is downloaded

# ActionFormer's 1-D NMS is a small C++ extension: build it once if this Python doesn't have it yet.
if ! "$PY" -c "import torch, nms_1d_cpu" 2>/dev/null; then
  echo "run.sh: building ActionFormer's NMS extension (one time)" >&2
  (cd "$ACTIONFORMER_DIR/libs/utils" && "$PY" setup.py install > /dev/null)
fi

exec "$PY" "$HERE/code/run_video.py" "$1" "$2"
