#!/usr/bin/env bash
# Member 3 single-video inference: ./run_member3.sh <video.mp4> <out.json>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
python -m member3_fpvsum.infer "$1" "$2"
