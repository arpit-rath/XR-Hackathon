#!/usr/bin/env bash
set -e
VIDEO_PATH="$1"
OUT_JSON="$2"
if [ -z "$VIDEO_PATH" ] || [ -z "$OUT_JSON" ]; then
  echo "Usage: ./run.sh <video.mp4> <out.json>"
  exit 1
fi
BASE_NAME="$(basename "$VIDEO_PATH")"
VID="${BASE_NAME%.*}"
mkdir -p "$(dirname "$OUT_JSON")"
python3 code/build_outputs.py --video "$VID" --out-dir "/tmp/xr_single_out"
cp "/tmp/xr_single_out/${VID}.json" "$OUT_JSON"
