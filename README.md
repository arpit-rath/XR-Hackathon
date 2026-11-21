# XR-Hackathon

Samsung PRISM Hackathon 2026, Track 04 (Action-Aware Video Summarization).

| Folder | What it is |
|---|---|
| `pipeline/` | EPIC verb-noun action detector (Member 2) |
| `fpvsum/` | FPVSum importance scores and category (Member 3) |
| `training/`, `splits/` | Training and held-out evaluation for the action detector |
| `package/` | Student package from the organisers (manifest, classes, validator) |

One environment runs everything: Python 3.10 on WSL2 Ubuntu 22.04 with CUDA 12.8.

```bash
pip install -r requirements.txt
```

Then follow `pipeline/README.md` for ActionFormer, epic-kitchens-slowfast and the weights, and `fpvsum/README.md`
for the CLIP weights.
