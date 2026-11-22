#!/usr/bin/env bash
# Assemble the submission folder from this repo:  team_04/build_package.sh <dest> <weights dir>
#   <weights dir> holds actionformer_epic_{verb,noun}_epoch_020.pth.tar, SlowFast.pyth and
#   clip_vit_l14_visual_fp16.pt (the transfer folder's weights/).
# Third-party code comes from ACTIONFORMER_DIR (61ea7eb) and SLOWFAST_DIR (bf50519).
# outputs/ is not touched; produce it with code/run_folder.py.
set -euo pipefail
DEST="$1"; W="$2"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
AF="${ACTIONFORMER_DIR:?set ACTIONFORMER_DIR}"; SF="${SLOWFAST_DIR:?set SLOWFAST_DIR}"
mkdir -p "$DEST/code" "$DEST/weights" "$DEST/outputs"

cp "$REPO/team_04/run.sh" "$REPO/team_04/README.md" "$DEST/"
cp "$REPO/requirements.txt" "$DEST/"
chmod +x "$DEST/run.sh"

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

for f in actionformer_epic_verb_epoch_020.pth.tar actionformer_epic_noun_epoch_020.pth.tar SlowFast.pyth clip_vit_l14_visual_fp16.pt; do
  # same size = already copied (a byte compare of 1.8 GB across Windows drives takes half an hour)
  if [ "$(stat -c %s "$W/$f")" != "$(stat -c %s "$DEST/weights/$f" 2>/dev/null)" ]; then
    cp "$W/$f" "$DEST/weights/"; COPIED=1
  fi
done
cp "$REPO"/fpvsum/weights/{category_head.pt,category_mapping.json,config.json,text_embeddings.pt,cooking_text_embedding.pt} "$DEST/weights/"
if [ "${COPIED:-0}" = 1 ] || [ ! -f "$DEST/WEIGHTS_SHA256SUMS.txt" ]; then
  (cd "$DEST/weights" && sha256sum * > ../WEIGHTS_SHA256SUMS.txt)
else  # big files unchanged: refresh only the small ones' checksums
  (cd "$DEST/weights" && { grep -E ' (actionformer|SlowFast|clip_vit)' ../WEIGHTS_SHA256SUMS.txt; \
    sha256sum category_head.pt category_mapping.json config.json text_embeddings.pt cooking_text_embedding.pt; } > ../W.tmp \
    && mv ../W.tmp ../WEIGHTS_SHA256SUMS.txt)
fi
echo "assembled $DEST: $(find "$DEST/code" -name '*.py' | wc -l) python files, weights $(du -sh "$DEST/weights" | cut -f1)"
