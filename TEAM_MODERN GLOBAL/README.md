# TEAM_MODERN GLOBAL: Action-Aware Video Summarization (Track 04)

Team Modern Global, SRM Institute of Science and Technology: Dhanush Poduval, Arpit Rath, Mehak Sharma,
Divya Narayanan.

`./run.sh <video.mp4> <out.json>` turns one video into the 6-field JSON. Everything is computed from
the video itself: no manifest, no video-ID lookups, and no network during inference. `outputs/` holds
the 187 files that this same code produced, with `"team_id": "TEAM_MODERN GLOBAL"`.

## How one video is processed (`code/run_video.py`)

1. **Duration:** ffprobe on the video.
2. **Video type:** CLIP ViT-L/14 features of the first 30 s at 2 fps are compared zero-shot against
   15 prompts: the 14 FPVSum categories plus one "cooking" prompt (`code/video_type.py`). The prompts
   were written in advance and never tuned on evaluation videos. On the 185 downloadable manifest videos,
   this decision was right for all of them.
3. **Cooking videos:**
   - Encoder A (EPIC SlowFast 8x8 R50, `code/encoder_a.py`) gives 2304-d features every 16 frames at 30 fps.
   - ActionFormer verb and noun models detect actions, fused into verb–noun pairs (IoU 0.5, top 3 nouns
     per verb, at most 2000; `code/detect_actions.py`, `code/fusion.py`).
   - Label: `cooking`, scored with its zero-shot probability.
   - Highlights: the best-scoring detected actions, at least 4 s each, non-overlapping, within 15% of
     the duration (`code/highlights.py`).
4. **FPVSum videos:**
   - CLIP features of the whole video at 2 fps feed the category classifier: a linear probe blended
     50/50 with zero-shot prompts.
   - A training-free importance scorer (`code/member3_fpvsum/`) gives a score every 0.5 s.
   - Highlights: 6 s windows around the most important moments, with the same 4 s and 15% rules.
   - `actions` is `[]`, which the rules allow for FPVSum videos.
5. The JSON is written, then checked with the organisers' `validate.py`. If any step fails, a minimal
   valid file is still written and the exit code is 1.

## Setup (Linux or WSL2 Ubuntu 22.04, Python 3.10, NVIDIA GPU with CUDA 12.8 drivers)

```bash
sudo apt install -y ffmpeg build-essential python3.10-venv curl
python3.10 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
./fetch_weights.sh                    # one time: 1.8 GB of weights from the GitHub release, SHA-256 checked
./run.sh sample.mp4 out/sample.json   # the first run also builds ActionFormer's NMS extension
python code/validate.py --dir out
```

The 4 large weights (ActionFormer verb and noun, EPIC SlowFast, CLIP visual tower) are over GitHub's
100 MB file limit, so they are attached to the release
[`weights-v1`](https://github.com/arpit-rath/XR-Hackathon/releases/tag/weights-v1) instead of git.
`run.sh` runs `fetch_weights.sh` by itself if they are missing. The small weights are in `weights/`.

GPU memory: tested on a 6 GB RTX 4050 Laptop GPU. Below 12 GB the encoder uses batches of 4 and the
detector runs in overlapping windows, which gives the same result, only slower. Rough speed on that
GPU: a 3-minute EPIC video takes about 2 minutes, a 30-minute one about 10 minutes, and a 3-minute
FPVSum video about 30 s.

## Layout

| Path | What it is |
|---|---|
| `run.sh` | Entry point; sets the paths below and calls `code/run_video.py` |
| `fetch_weights.sh` | Downloads and checks the 4 large weights (run automatically by `run.sh` when needed) |
| `build_package.sh` | Rebuilds `code/` and the small weights from the repo's sources |
| `code/run_video.py`, `code/run_folder.py` | One video / a batch (models loaded once). `outputs/` came from `run_folder.py` |
| `code/video_type.py`, `code/highlights.py` | Cooking-or-FPV decision; highlight selection |
| `code/encoder_a.py`, `code/detect_actions.py`, `code/fusion.py` | EPIC action detection |
| `code/member3_fpvsum/` | FPVSum features, category classifier and importance scorer |
| `code/third_party/` | ActionFormer (`61ea7eb`, MIT) and epic-kitchens-slowfast (`bf50519`, CC BY-NC-SA 4.0), only the parts inference needs |
| `code/build_text_embeddings.py`, `code/build_cooking_embedding.py` | How the text embeddings in `weights/` were made (need the CLIP text tower, so they're not run at inference) |
| `code/validate.py`, `code/supported_classes.json` | Organisers' validator and class list |
| `weights/` | In git: category head, text embeddings, config. From the release: ActionFormer EPIC verb and noun (`epoch_020`), EPIC SlowFast 8x8 R50, CLIP ViT-L/14 visual tower (fp16). Checksums for all 9 in `WEIGHTS_SHA256SUMS.txt` |
| `outputs/` | 187 `<video_id>.json` files |
| `requirements.txt` | Exact Python dependencies |

## How `outputs/` was made

`code/run_folder.py` ran `run_video.build()` over every downloadable manifest video, using
`--cache-dir` so that features already computed by the same code (`encoder_a.py`, `member3_fpvsum`)
were not recomputed. Two FPVSum videos can't be downloaded any more: `Scuba_KD74Ywuy9ng` is private and
`Scuba_Um4kqB8bQBg` has been removed. With no video there is nothing to run, so their two files are
placeholders: label `other` with score 0, the manifest duration, and one highlight of 15% of it from the start.

## Models and data

- **Action detector:** the released ActionFormer EPIC-KITCHENS-100 verb and noun checkpoints, on
  features from the EPIC SlowFast checkpoint, both trained on EPIC-KITCHENS-100 *train*. The manifest's
  EPIC videos are the *validation* split, so nothing trained on them. Held-out verb–noun pair mAP was 27.23
  (tIoU 0.1–0.5), on an internal split carved out of train.
- **FPVSum module:** frozen OpenAI CLIP ViT-L/14. The category head was trained on 37 FPVSum videos that
  are not in the manifest. The importance scorer is training-free. Samsung's 49 evaluation videos were
  never used for training or tuning.
