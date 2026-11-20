# Action detector (Member 2)

MP4 → EPIC SlowFast features (Encoder A) → released ActionFormer verb and noun models → fused
verb–noun records, at most 2,000 per video, sorted by score. Shipped settings: fusion IoU 0.5,
3 nouns per verb, β = 0 (held-out pair mAP 27.23 for the same settings on the retrained control).

| File | Does |
|---|---|
| `encoder_a.py` | MP4 → `T × 2304` features, one per 16 frames at 30 fps |
| `detect_actions.py` | One video → action records JSON (`[]` and a logged error on failure) |
| `run_manifest.py` | All 138 EPIC manifest videos, resumable, loads the models once |
| `parity_check.py` | Our features vs the released features for the same video |
| `requirements.txt` | Pinned versions that ran on the training machine |

Paths come from environment variables: `ACTIONFORMER_DIR`, `SLOWFAST_DIR`, `SLOWFAST_CKPT`,
`VERB_CKPT`, `NOUN_CKPT`, `SUPPORTED_CLASSES` and, for downloads, `EPIC_DOWNLOADER`. The defaults
match the training machine (`~/epic/...`).

## Full run on the training machine

Inside WSL Ubuntu, with the environment that ran training activated:

```bash
cd /mnt/d/XR_Hackathon/XR-Hackathon && git checkout Training && git pull   # the repo on that PC (D:\XR_Hackathon\XR-Hackathon)
export SUPPORTED_CLASSES=$PWD/package/supported_classes.json
git clone https://github.com/epic-kitchens/epic-kitchens-download-scripts.git ~/epic/epic-kitchens-download-scripts  # once
```

**1. Parity check on one video first** (about 5 minutes):

```bash
python pipeline/run_manifest.py --videos ~/epic/videos --out ~/epic/manifest_run --download --keep-videos --only P01_11
python pipeline/parity_check.py ~/epic/manifest_run/features/P01_11.npz \
    ~/epic/actionformer_release/data/epic_kitchens/features/P01_11.npz --detect
```

Result on 20 Nov (laptop, RTX 4050): **PARITY PASS**: 1,053 vs 1,053 steps, median per-step cosine
0.965, 93% of the top-100 released detections reproduced. The defaults in `encoder_a.py` are that
recipe: BGR input, full 456×256 frame, JPEG round trip at q4, the model's head pooling, and padded
tail clips. The first default (RGB, global mean, no padding) scored 0.67 and 49%. The `ENCODER_*`
variables in `encoder_a.py` switch each choice back for comparison.

**2. All 138 videos** (run in `tmux` so it survives a closed terminal; rerun the same command to resume):

```bash
python pipeline/run_manifest.py --videos ~/epic/videos --out ~/epic/manifest_run --download
```

Drop `--download` if the MP4s are already on disk (point `--videos` at them; it searches subfolders).
Progress and failures go to `~/epic/manifest_run/status.tsv`.

**3. Copy back** `~/epic/manifest_run/actions/` (138 JSON files, a few MB) and, optionally,
`features/` (about 1 GB, a cache for re-running detection without the videos). From Windows on
the same machine, the folder is at `\\wsl$\Ubuntu-22.04\root\epic\manifest_run`.

## Small GPUs

Under 12 GB the encoder uses batches of 4 (`ENCODER_BATCH` overrides) and detection runs in
2,304-step windows overlapping by 256 (`AF_WINDOW` overrides; `0` forces one pass). On larger GPUs
nothing changes, and a one-pass out-of-memory error falls back to windows.
