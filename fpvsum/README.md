# Member 3: FPVSum importance scoring and category classification

Samsung XR Hackathon 2026, Track 04 (Action-Aware Video Summarization).

**What this module produces** for one first-person video:
- **importance scores** aligned to timestamps, at 2 fps, each in 0–1;
- a **predicted category** (one of the 14 FPVSum labels in `supported_classes.json`) with a **confidence** score;
- the probability for every category.

Highlight selection (segments of at least 4 s, within the 15% budget) is **not** done here. It belongs to Member 4, who reads `importance.timestamps_sec` and `importance.scores` from this module's output.

## Quick start

```bash
# 1. Python 3.12 (tested). Install the tested versions:
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121   # drop --index-url for CPU
pip install -r requirements.txt
pip install --no-deps open_clip_torch==4.2.2 timm==1.0.31   # see requirements.txt for why --no-deps
# 2. ffmpeg: on PATH, or set FFMPEG=/path/to/ffmpeg
# 3. Run from this folder:
python -m member3_fpvsum.infer path/to/video.mp4 out/video.json
#    or: ./run_member3.sh video.mp4 out.json   (Windows: run_member3.bat video.mp4 out.json)
# Optional: --device cpu|cuda, --video-id ID, --save-features feats.npz, --features feats.npz
```

From Python:
```python
from member3_fpvsum.infer import Member3Model
m = Member3Model()                       # loads ./weights; no internet needed
out, feats = m.predict_video("Ski_-Z5a4UQmZLs.mp4")
```

Measured: a 3-minute video takes 9.3 s end to end on an RTX 6000 Ada, including model loading. CPU inference wasn't measured.

**Verification (2026-11-21):** the unzipped package was run from a separate folder, with Hugging Face access switched off and an empty model cache. On `Ski_-Z5a4UQmZLs.mp4`, which was never used in training, it produced 369 scores and timestamps (0–184 s at 2 fps) and the label `ski` (0.563). That matches the original cached-feature run exactly (Pearson 1.0). The same test then **passed in a brand-new venv**, built from this ZIP's `requirements.txt` plus the `--no-deps` step: same label, scores within 0.0001. In that venv torch 2.5.1 came from the machine's existing Python install, because PyPI's torch download had stalled; everything else was installed fresh. See `metrics/verification.json`.

The message "No pretrained weights loaded for model 'ViT-L-14-quickgelu'" is expected. open_clip prints it before the package loads the bundled CLIP weights.

## Output format (`sample_prediction.json`)

```json
{
  "video_id": "Ski_-Z5a4UQmZLs", "duration_sec": 184.599, "module": "member3_fpvsum", "version": "1.0.0",
  "feature_fps": 2.0,
  "importance": {"timestamps_sec": [0.0, 0.5, ...], "scores": [0.41, 0.44, ...],
                 "method": "clip_zeroshot_v1",
                 "timestamp_convention": "score i describes the frame sampled at t = i / feature_fps seconds"},
  "label": {"name": "ski", "score": 0.97},
  "category_probs": {"bike": 0.001, ...}
}
```

## Models

| Part | Model | Trained? | File |
|---|---|---|---|
| Feature extractor | OpenAI CLIP ViT-L/14, visual tower, frozen (`open_clip` arch `ViT-L-14-quickgelu`) | pretrained | `weights/clip_vit_l14_visual_fp16.pt` (580 MB) |
| Category classifier | Linear probe on [mean, max]-pooled CLIP features, blended 50/50 with CLIP zero-shot prompts | **yes**, on 37 non-Samsung FPVSum videos | `weights/category_head.pt`, `weights/text_embeddings.pt` |
| Importance scorer | Zero-shot CLIP scorer: 0.5 × prompt relevance + 0.25 × novelty + 0.25 × visual dynamics, smoothed over 1 s | **no** (training-free, see below) | `weights/text_embeddings.pt`, `weights/config.json` |

**Preprocessing**, identical for training and inference (`member3_fpvsum/features.py`):
1. ffmpeg `fps=2`.
2. Bicubic resize so the short side is 224, then a 224×224 center crop.
3. RGB, normalized with CLIP mean/std.
4. One L2-normalized 768-d embedding per sampled frame.

The prompts and the category mapping are in `weights/category_mapping.json` and `member3_fpvsum/prompts.py`.

## Data and leakage rules

- **Samsung's 49 FPVSum evaluation videos were never used** for training, validation, model selection or tuning. `train_category.py` asserts this. The only place they're used is the report-only diagnostic below, run after the models were frozen.
- **Category classifier training data:** the 37 still-downloadable *unlabeled* FPVSum videos that are **not** in Samsung's manifest. They cover all 14 categories, and their labels come from the official `FPVSum_videolist.csv`.
- **Why the importance scorer is not trained:**
  - The 7 annotated FPVSum videos outside Samsung's list are all deleted or private on YouTube.
  - **TVSum** is released under the Yahoo! Webscope licence. It may only be used under a signed Data Sharing Agreement, for academic research, and not by any commercial entity. We have no such agreement, and the hackathon is run by a company, so TVSum was **not used**. Its annotation file was extracted locally and checked (1,000 rows, 50 videos, zip integrity OK), but not used.
  - **SumMe**'s only download server was throttled to about 6 KB/s (2.3 GB), and its licence could not be confirmed. Not used.
  - Rather than invent labels, importance uses a training-free CLIP scorer with weights fixed in advance. `member3_fpvsum/models.py` also has an `ImportanceTransformer`, ready to train if a properly licensed importance dataset becomes available.

## Results

**Category classifier**, validated only on non-Samsung FPVSum videos (`metrics/category_metrics.json`):

| Protocol | Blend (shipped) | Trained part alone | Zero-shot alone |
|---|---|---|---|
| Fixed held-out split (24 train / 13 val, one video per class) | **100%** (13/13) | 92.3% | 100% |
| Leave-one-video-out (37 videos) | **97.3%** (36/37) | 83.8% | 97.3% |

These validation sets are tiny, so treat the numbers as rough.

**Final diagnostic on Samsung's 47 downloadable evaluation videos**: report-only, computed after the models were frozen, never used to change anything (`metrics/diagnostic_samsung47.json`):

| Metric | Value |
|---|---|
| Category accuracy | 95.7% (45/47) |
| Importance F1, top 15% predicted vs top 15% consensus frames | 0.226 (random: 0.150) |
| Importance Spearman ρ against consensus scores | 0.208 |

## Matching annotation frames to timestamps (for whoever evaluates this)

- `gt_data.mat` `Score` gives one value per annotation frame. Those frames come from the annotators' copy of each video, which is often 24–30 fps. Today's YouTube copies are sometimes 48–60 fps, so frame *i* of the annotations is **not** frame *i* of the video.
- Map annotation frame *i* of *N* to time **t = (i + 0.5) / N × duration**. Every downloaded duration matches the manifest within 0.15 s, so this is safe. `member3_fpvsum/align.py` implements it.
- `gt_data.mat` spells one ID `Jumping_mU-p2wuk90`, but the manifest has `Jumping__mU-p2wuk90`.
- For BikePolo videos (and `Skate_OporDjnbc3E`), the YouTube ID is **not** part of the name. Use the `youtube_id` column of `FPVSum_videolist.csv`.

## Known limitations

- 2 of the 49 Samsung FPVSum videos can't be downloaded: `Scuba_KD74Ywuy9ng` is private and `Scuba_Um4kqB8bQBg` is unavailable. Ask the organizers how to handle them, and don't hard-code labels for them.
- 3 BikePolo videos only exist at 360p on YouTube.
- Importance scoring is zero-shot, so it beats random clearly but only modestly. The biggest possible improvement is training on a properly licensed egocentric importance dataset.
- Low-confidence confusions seen so far: plane ↔ jumping (aerial views) and bike ↔ bikepolo.

## Layout

```
member3_fpvsum/   inference package (features, align, importance, models, prompts, infer)
weights/          CLIP visual weights, category head, text embeddings, config, category mapping
training_code/    scripts used to download, match, extract features, train and evaluate (they expect the
                  original D:\XR_Hackathon\XR_Hackathon_FPVSum layout; included for reproducibility)
metrics/          category_metrics.json, diagnostic_samsung47.json
logs/             download, feature extraction, training and diagnostic logs
reports/          manifest matching, download verification, annotation alignment, data availability
outputs/          Member 3 predictions for the 47 Samsung videos (this module's format, not the final submission JSON)
sample_prediction.json, requirements.txt, run_member3.sh / .bat
```

## Putting this on GitHub

`weights/clip_vit_l14_visual_fp16.pt` is 580 MB, over GitHub's 100 MB per-file limit. Either:
- track it with Git LFS: `git lfs track "weights/*.pt"`; or
- leave it out (it's in `.gitignore`) and regenerate it with `training_code/build_text_embeddings.py`, which downloads the public OpenAI CLIP weights through open_clip.

The small weights (`category_head.pt`, `text_embeddings.pt`) are under 100 KB and can be committed normally. No raw videos and no TVSum/SumMe data are included.

## Credits

CLIP (Radford et al., 2021; OpenAI weights via `open_clip`). FPVSum (Ho, Chiu and Wang, ECCV 2018).
