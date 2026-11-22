"""Track 04 end to end for ONE video: MP4 -> the 6-field submission JSON. This is what run.sh calls.

    python run_video.py VIDEO.mp4 OUT.json [--team-id team_04] [--cache-dir DIR] [--keep-actions-for-fpv]

Everything is computed from the video itself (no manifest, no video-ID lookups):
  1. duration_sec  ffprobe on the video.
  2. video type    CLIP on the first 30 s, zero-shot over Member 3's 14 FPVSum prompts + "cooking"
                   (video_type.py).
  3. cooking       Encoder A (EPIC SlowFast) -> ActionFormer verb + noun -> fused verb-noun actions
                   (encoder_a.py, detect_actions.py); label "cooking" with its zero-shot probability;
                   highlights from the actions (highlights.py, Member 4).
     otherwise     Member 3's module on CLIP features of the whole video at 2 fps -> FPVSum label and
                   importance scores; highlights from the importance scores (highlights.py);
                   actions [] (allowed for FPVSum videos).
  4. write the JSON and check it with the organisers' rules (validate.py, when it is next to this file
     or in ../package).

--cache-dir DIR reuses features this same code saved earlier under DIR/<kind>/<video_id>.npz (kinds:
clip30, clip, encoder_a); Encoder A entries are reused only when their settings string matches.
It is only a speed-up for batch runs; without it every feature is computed from the video.

Paths come from the environment (run.sh sets them): ACTIONFORMER_DIR, SLOWFAST_DIR, SLOWFAST_CKPT,
VERB_CKPT, NOUN_CKPT, SUPPORTED_CLASSES, M3_DIR (folder holding member3_fpvsum/), M3_WEIGHTS.
If anything fails, a minimal valid JSON is still written (so the file exists) and the exit code is 1.
"""
import argparse
import json
import logging
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
M3_DIR = os.environ.get("M3_DIR") or (HERE if os.path.isdir(os.path.join(HERE, "member3_fpvsum"))
                                       else os.path.join(REPO, "fpvsum"))
M3_WEIGHTS = os.environ.get("M3_WEIGHTS") or os.path.join(M3_DIR, "weights")
sys.path.insert(0, M3_DIR)
log = logging.getLogger("run_video")


def r2(x):
    return round(float(x), 2)


def cache_path(cache_dir, kind, vid):
    return os.path.join(cache_dir, kind, f"{vid}.npz") if cache_dir else None


def load_npz(path, settings=None):
    if not path or not os.path.exists(path):
        return None
    with np.load(path) as z:
        if settings is not None and ("settings" not in z.files or str(z["settings"]) != settings):
            return None
        return z["feats"].astype(np.float32)


def save_npz(path, **arrays):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "wb") as f:
            np.savez(f, **arrays)
        os.replace(path + ".tmp", path)


def clip_features(m3, video, path, max_sec=None):
    feats = load_npz(path)
    if feats is None:
        feats, _, _ = m3.clip.video_features(video, fps=m3.cfg["feature_fps"], max_sec=max_sec)
        save_npz(path, feats=feats.astype(np.float16))
    return feats


_MODELS = {}  # loaded once per process, so a batch run does not reload them for every video


def model(name):
    if name not in _MODELS:
        if name == "member3":
            from member3_fpvsum.infer import Member3Model
            _MODELS[name] = Member3Model(M3_WEIGHTS)
        elif name == "video_type":
            from video_type import VideoTypeClassifier
            _MODELS[name] = VideoTypeClassifier(M3_WEIGHTS)
        elif name == "encoder_a":
            import encoder_a
            _MODELS[name] = encoder_a.load_encoder()
        elif name == "detectors":
            import detect_actions as da
            cwd = os.getcwd()
            try:
                _MODELS[name] = da.load_detectors()  # chdirs into ActionFormer for its configs
            finally:
                os.chdir(cwd)
    return _MODELS[name]


def cooking_branch(video, vid, duration, cache_dir):
    import torch
    import encoder_a
    import detect_actions as da
    path = cache_path(cache_dir, "encoder_a", vid)
    feats = load_npz(path, encoder_a.SETTINGS)
    if feats is None:
        feats, _ = encoder_a.extract(video, model=model("encoder_a"))
        save_npz(path, feats=feats, duration=duration, settings=encoder_a.SETTINGS)
        torch.cuda.empty_cache()
    verbs, nouns = da.class_names()
    pairs = da.detect(model("detectors"), feats, duration, vid)
    records = da.to_records(pairs, duration, verbs, nouns)
    da.check_contract(records, duration, verbs, nouns)
    torch.cuda.empty_cache()
    return records


def build(video, team_id, cache_dir=None, keep_actions_for_fpv=False):
    import encoder_a
    from highlights import epic_highlights, fpv_highlights
    from video_type import DECIDE_SEC

    vid = os.path.splitext(os.path.basename(video))[0]
    _, _, duration = encoder_a.probe(video)
    m3 = model("member3")
    first = clip_features(m3, video, cache_path(cache_dir, "clip30", vid), max_sec=DECIDE_SEC)
    is_cooking, p_cooking, top = model("video_type").decide(first)
    log.info("%s: %.1f s, video type %s (p_cooking %.3f)", vid, duration, "cooking" if is_cooking else top, p_cooking)

    if is_cooking:
        actions = cooking_branch(video, vid, duration, cache_dir)
        label = {"name": "cooking", "score": round(p_cooking, 4)}
        highlights = epic_highlights(actions, duration)
    else:
        feats = clip_features(m3, video, cache_path(cache_dir, "clip", vid))
        out = m3.predict_features(feats, duration, vid)
        label = out["label"]
        highlights = fpv_highlights(out, duration)
        actions = cooking_branch(video, vid, duration, cache_dir) if keep_actions_for_fpv else []
    return {"team_id": team_id, "video_id": vid, "duration_sec": r2(duration), "label": label,
            "actions": actions, "highlights": highlights}


def fallback(video, team_id):
    vid = os.path.splitext(os.path.basename(video))[0]
    try:
        import encoder_a
        duration = encoder_a.probe(video)[2]
    except Exception:
        duration = 4.0
    return {"team_id": team_id, "video_id": vid, "duration_sec": r2(duration), "label": {"name": "other", "score": 0.0},
            "actions": [], "highlights": [{"segment": [0.0, r2(min(duration, 4.0))]}]}


def check(path):
    """Organisers' validate.py on the written file, when it can be found; returns the error list."""
    for d in (HERE, os.path.join(REPO, "package"), os.path.join(REPO, "student_package")):
        if os.path.exists(os.path.join(d, "validate.py")):
            sys.path.insert(0, d)
            import validate
            cls = os.environ.get("SUPPORTED_CLASSES")
            rep, _ = validate.validate(path, None, validate.load_classes(cls) if cls and os.path.exists(cls) else None)
            return rep.errors
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--team-id", default=os.environ.get("TEAM_ID", "team_04"))
    ap.add_argument("--cache-dir")
    ap.add_argument("--keep-actions-for-fpv", action="store_true",
                    help="also run the action detector on FPVSum videos (not scored; off by default)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    video, out = os.path.abspath(a.video), os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0, code = time.time(), 0
    try:
        result = build(video, a.team_id, a.cache_dir, a.keep_actions_for_fpv)
    except Exception:
        log.error("pipeline failed for %s; writing a minimal valid file\n%s", video, traceback.format_exc())
        result, code = fallback(video, a.team_id), 1
    with open(out + ".tmp", "w") as f:
        json.dump(result, f, indent=1)
    os.replace(out + ".tmp", out)
    errors = check(out)
    if errors:
        log.error("%s fails validation: %s", out, errors)
        code = 1
    log.info("%s -> %s: label %s, %d actions, %d highlights (%.0f s)", os.path.basename(video), out,
             result["label"]["name"], len(result["actions"]), len(result["highlights"]), time.time() - t0)
    sys.exit(code)


if __name__ == "__main__":
    main()
