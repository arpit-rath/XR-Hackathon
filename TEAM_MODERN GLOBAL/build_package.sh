#!/usr/bin/env bash
# (Re)assemble the submission package from the repo's sources.
#   ./build_package.sh                  update this folder in place (code/, small weights, checksums)
#   ./build_package.sh <dest> [<weights dir>]
#                                       build a full copy elsewhere; <weights dir> holds the 4 large files
# Third-party code comes from ACTIONFORMER_DIR (61ea7eb) and SLOWFAST_DIR (bf50519).
# The 4 large weights are not in git: fetch_weights.sh downloads them from the GitHub release.
# outputs/ is not touched; produce it with code/run_folder.py.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$HERE}"; W="${2:-}"
REPO="${REPO:-$(cd "$HERE/.." && pwd)}"
AF="${ACTIONFORMER_DIR:?set ACTIONFORMER_DIR}"; SF="${SLOWFAST_DIR:?set SLOWFAST_DIR}"
BIG="actionformer_epic_verb_epoch_020.pth.tar actionformer_epic_noun_epoch_020.pth.tar SlowFast.pyth clip_vit_l14_visual_fp16.pt"
SMALL="category_head.pt category_mapping.json config.json text_embeddings.pt cooking_text_embedding.pt"
mkdir -p "$DEST/code" "$DEST/weights" "$DEST/outputs"
DEST="$(cd "$DEST" && pwd)"

if [ "$DEST" != "$HERE" ]; then
  cp "$HERE/run.sh" "$HERE/README.md" "$HERE/fetch_weights.sh" "$HERE/.gitignore" "$DEST/"
  cp "$HERE/WEIGHTS_SHA256SUMS.txt" "$DEST/" 2>/dev/null || true
fi
cp "$REPO/requirements.txt" "$DEST/"
chmod +x "$DEST/run.sh" "$DEST/fetch_weights.sh"

rm -rf "$DEST/code"; mkdir -p "$DEST/code/third_party"
cp "$REPO"/pipeline/{run_video,run_folder,video_type,highlights,encoder_a,detect_actions}.py "$DEST/code/"
cp "$REPO/training/fusion.py" "$REPO/video_ingest.py" "$DEST/code/"
cp "$REPO/package/validate.py" "$REPO/package/supported_classes.json" "$DEST/code/"
cp -r "$REPO/fpvsum/member3_fpvsum" "$DEST/code/"
cp "$REPO/fpvsum/training_code/build_cooking_embedding.py" "$REPO/fpvsum/training_code/build_text_embeddings.py" "$DEST/code/"

mkdir -p "$DEST/code/third_party/actionformer_release/configs"
cp -r "$AF/libs" "$DEST/code/third_party/actionformer_release/"
cp "$AF"/configs/epic_slowfast_{verb,noun}.yaml "$DEST/code/third_party/actionformer_release/configs/"
cp "$AF/LICENSE" "$DEST/code/third_party/actionformer_release/"
rm -rf "$DEST/code/third_party/actionformer_release/libs/utils/"{build,dist,nms_1d_cpu.egg-info}
mkdir -p "$DEST/code/third_party/epic-kitchens-slowfast/configs/EPIC-KITCHENS"
cp -r "$SF/slowfast" "$DEST/code/third_party/epic-kitchens-slowfast/"
cp "$SF/configs/EPIC-KITCHENS/SLOWFAST_8x8_R50.yaml" "$DEST/code/third_party/epic-kitchens-slowfast/configs/EPIC-KITCHENS/"
cp "$SF/LICENSE" "$DEST/code/third_party/epic-kitchens-slowfast/"
find "$DEST/code" -name __pycache__ -type d -prune -exec rm -rf {} +

# weights: the small ones always; the large ones only when a source folder is given
for f in $SMALL; do cp "$REPO/fpvsum/weights/$f" "$DEST/weights/"; done
if [ -n "$W" ]; then
  for f in $BIG; do
    # same size = already copied (a byte compare of 1.8 GB across Windows drives takes half an hour)
    [ "$(stat -c %s "$W/$f")" = "$(stat -c %s "$DEST/weights/$f" 2>/dev/null)" ] || cp "$W/$f" "$DEST/weights/"
  done
fi
# checksums: large files keep their recorded hash unless present; small files are recomputed
SUMS="$DEST/WEIGHTS_SHA256SUMS.txt"; NEW="$(mktemp)"
for f in $BIG; do
  if [ -f "$DEST/weights/$f" ] && [ -n "$W" ]; then (cd "$DEST/weights" && sha256sum "$f");
  else grep -E "  $f\$" "$SUMS" || { echo "no recorded checksum for $f" >&2; exit 1; }; fi
done > "$NEW"
(cd "$DEST/weights" && sha256sum $SMALL) >> "$NEW"
mv "$NEW" "$SUMS"
echo "assembled $DEST: $(find "$DEST/code" -name '*.py' | wc -l) python files"
