# XR-Hackathon

Samsung PRISM Hackathon 2026, Track 04 (Action-Aware Video Summarization).

| Folder | What it is |
|---|---|
| `pipeline/` | EPIC verb-noun action detector (Member 2); `run_video.py` = the whole system for one video, `run_folder.py` = the same for a batch, plus `video_type.py` (cooking or FPV) and `highlights.py` (Member 4) |
| `fpvsum/` | FPVSum importance scores and category (Member 3) |
| `video_ingest.py` | MP4 ingestion and streaming decoder (Member 1) |
| `team_04/` | Submission: `run.sh`, `README.md`, and `build_package.sh`, which assembles the full `team_04/` folder (code, third-party code, weights) |
| `training/`, `splits/` | Training and held-out evaluation for the action detector |
| `package/` | Student package from the organisers (manifest, classes, validator) |

One environment runs everything: Python 3.10 on WSL2 Ubuntu 22.04 with CUDA 12.8.

```bash
pip install -r requirements.txt
```

Then follow `pipeline/README.md` for ActionFormer, epic-kitchens-slowfast and the weights, and `fpvsum/README.md`
for the CLIP weights.
