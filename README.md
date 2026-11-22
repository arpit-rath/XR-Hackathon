# AURIS: Action-Aware Video Summarization

**Samsung PRISM XR Hackathon 2026 · Track 04 · TEAM_MODERN GLOBAL**
Team Modern Global, SRM Institute of Science and Technology: Dhanush Poduval, Arpit Rath, Mehak Sharma,
Divya Narayanan.

AURIS takes a first-person video and, fully offline, finds the hand–object actions, picks highlights
within 15% of the video, labels the video, and writes the organisers' 6-field JSON. The AURIS app lets
you explore the result and ask questions about the video.

| What the organisers asked for | Where it is |
|---|---|
| **Submission package** (`TEAM_ID/`: `run.sh`, `code/`, `weights/`, `requirements.txt`, `outputs/`, `README.md`) | [`TEAM_MODERN GLOBAL/`](TEAM_MODERN%20GLOBAL/) |
| **187 output files** (all pass `validate.py`) | [`TEAM_MODERN GLOBAL/outputs/`](TEAM_MODERN%20GLOBAL/outputs/) |
| **Large weights** (1.8 GB, over GitHub's 100 MB file limit) | Release [`weights-v1`](https://github.com/arpit-rath/XR-Hackathon/releases/tag/weights-v1); `run.sh` downloads them automatically |
| **Presentation** | [`docs/LangAI3.0_AI_Disclosure(1).docx`] |
| **AI Disclosure Form ** | [`docs/AURIS - Team Modern Global - Track 04.pptx`](docs/AURIS%20-%20Team%20Modern%20Global%20-%20Track%2004.pptx) |
| **Demo app** (brownie feature) | [`app/`](app/), with screenshots in [`docs/screenshots/`](docs/screenshots/) |

## Run it

Linux or WSL2 Ubuntu 22.04, Python 3.10, an NVIDIA GPU (tested on 6 GB and 48 GB), ffmpeg.

```bash
git clone https://github.com/arpit-rath/XR-Hackathon.git
cd "XR-Hackathon/TEAM_MODERN GLOBAL"
python3.10 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
./run.sh sample.mp4 out/sample.json     # first run: fetches the weights and builds one small C++ extension
python code/validate.py --dir out
```

Everything is computed from the video itself: no manifest, no video-ID lookups. Renamed videos give
identical output. Details are in the [package README](TEAM_MODERN%20GLOBAL/README.md).

## How it works

1. **Video type:** CLIP ViT-L/14 on the first 30 s, zero-shot against 14 sports + "cooking". Right on
   all 185 downloadable manifest videos.
2. **Kitchen videos:** EPIC SlowFast features → ActionFormer verb + noun → fused verb–noun actions
   (held-out pair mAP 27.23). The features match the released ones (cosine 0.965).
3. **Sports videos:** CLIP at 2 fps → category (95.7% on Samsung's 47, report-only) and importance per 0.5 s.
4. **Highlights:** greedy, non-overlapping, ≥ 4 s each, ≤ 15% of the video, from the actions or the importance.
5. **Output:** the 6-field JSON, checked with the organisers' validator.

## AURIS app (brownie feature)

A local web app (no cloud): upload a video, see the label, highlights and actions on a timeline, play only
the highlights, view the exact JSON output, and ask questions such as "When do they open the fridge?".
Answers come from the JSON (a local llama3.2 model only interprets the question), so every timestamp is
one the system detected. See [`app/README.md`](app/README.md); start it with `app\start_app.bat` on Windows.

![AURIS](docs/screenshots/2_kitchen_analysis.png)

## Repository layout

| Folder | What it is | Member |
|---|---|---|
| `TEAM_MODERN GLOBAL/` | The submission package | all |
| `video_ingest.py` | MP4 ingestion and streaming decoder | Member 1 · Dhanush Poduval |
| `pipeline/` | EPIC action detector; `run_video.py` is the whole system for one video | Member 2 · Arpit Rath |
| `fpvsum/` | FPVSum importance scores and category classifier | Member 3 · Mehak Sharma |
| `pipeline/highlights.py`, `build_outputs.py` | Highlight selection and output assembly | Member 4 · Divya Narayanan |
| `app/` | AURIS demo app and chatbot | |
| `training/`, `splits/` | Training and held-out evaluation for the action detector | |
| `package/` | Organisers' student package (manifest, classes, validator) | |
| `docs/` | Presentation and screenshots | |

## Notes for the organisers

- Two FPVSum manifest videos can no longer be downloaded (`Scuba_KD74Ywuy9ng` is private,
  `Scuba_Um4kqB8bQBg` removed). Their output files are valid placeholders.
- `code/third_party/epic-kitchens-slowfast` is licensed CC BY-NC-SA 4.0 (non-commercial); ActionFormer is MIT.
- Samsung's evaluation videos were never used for training or tuning. The EPIC manifest videos are the
  EPIC-KITCHENS-100 validation split; all action models were trained on the train split.
