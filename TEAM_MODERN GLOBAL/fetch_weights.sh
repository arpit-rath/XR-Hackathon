#!/usr/bin/env bash
# Download the 4 large model weights (1.8 GB, too big for git) from the repo's GitHub release into weights/,
# and check each one against WEIGHTS_SHA256SUMS.txt. Files that are already present and correct are skipped.
#   ./fetch_weights.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URL="${WEIGHTS_URL:-https://github.com/arpit-rath/XR-Hackathon/releases/download/weights-v1}"
BIG="actionformer_epic_verb_epoch_020.pth.tar actionformer_epic_noun_epoch_020.pth.tar SlowFast.pyth clip_vit_l14_visual_fp16.pt"
mkdir -p "$HERE/weights"
cd "$HERE/weights"
ok() { grep -E "  $1\$" "$HERE/WEIGHTS_SHA256SUMS.txt" | sha256sum -c --quiet - 2>/dev/null; }
for f in $BIG; do
  if [ -f "$f" ] && ok "$f"; then echo "ok        $f"; continue; fi
  echo "download  $f"
  if command -v curl > /dev/null; then curl -fL --retry 3 -o "$f.part" "$URL/$f"; else wget -q -O "$f.part" "$URL/$f"; fi
  mv "$f.part" "$f"
  ok "$f" || { echo "checksum mismatch: $f" >&2; exit 1; }
  echo "ok        $f"
done
echo "all weights present and verified"
