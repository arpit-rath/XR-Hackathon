# Team Submission (`team_04`)

## Contents
- `outputs/`: 187 validated `<video_id>.json` files matching `manifest.csv`.
- `code/`: Integration and highlight-selection scripts (`build_outputs.py`, `video_ingest.py`).
- `run.sh`: Single-video entry point (`./run.sh <video.mp4> <out.json>`).
- `requirements.txt`: Python dependencies.
- `weights/`: Model checkpoints/weights directory.

## Validation
Run `python package/validate.py --dir team_04/outputs` to verify all 187 output files.
