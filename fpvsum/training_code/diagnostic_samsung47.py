"""FINAL DIAGNOSTIC ONLY - run after the model is frozen. Results must not be used to change the model.

Scores the exported Member 3 package on Samsung's 47 downloadable FPVSum evaluation videos:
  * importance: frame-level F1 of top-15% predicted vs top-15% consensus (gt_data.mat Score),
    both on the 2-fps timeline (annotation frame i of N -> t = (i+0.5)/N * duration); Spearman rho.
  * category: accuracy of label.name vs the official FPVSum category.
Writes training/metrics/diagnostic_samsung47.json and outputs/<video_id>.json (sample predictions).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

WS = Path(__file__).resolve().parent.parent
PKG = WS / "export" / "Member3_FPVSum_Trained_Model"
sys.path.insert(0, str(PKG))
from member3_fpvsum.align import f1_top_fraction, frame_scores_to_timeline, spearman  # noqa: E402
from member3_fpvsum.infer import Member3Model  # noqa: E402

GT_NAME_FIXES = {"Jumping_mU-p2wuk90": "Jumping__mU-p2wuk90"}


def main():
    gt = sio.loadmat(WS / "original_fpvsum/fpvsum/annotation/gt_data.mat", squeeze_me=True,
                     struct_as_record=False)["gt_data"]
    gt = {GT_NAME_FIXES.get(str(e.Name), str(e.Name)): np.asarray(e.Score, float) for e in gt}
    matched = {r["samsung_id"]: r for r in csv.DictReader(open(WS / "reports/fpvsum_matched.csv", encoding="utf-8"))}
    m = Member3Model(PKG / "weights", load_visual=False)
    rng = np.random.default_rng(0)
    rows = []
    for p in sorted((WS / "training/features/samsung47").glob("*.npz")):
        d = np.load(p)
        feats, dur = d["feats"].astype(np.float32), float(d["duration"])
        out = m.predict_features(feats, dur, p.stem)
        json.dump(out, open(PKG / "outputs" / f"{p.stem}.json", "w"), indent=1)
        pred = np.array(out["importance"]["scores"])
        g = frame_scores_to_timeline(gt[p.stem], dur, len(pred), out["feature_fps"])
        rand = np.mean([f1_top_fraction(rng.random(len(pred)), g) for _ in range(20)])
        rows.append({"video_id": p.stem, "true_category": matched[p.stem]["category_label"],
                     "pred_category": out["label"]["name"], "confidence": out["label"]["score"],
                     "f1_top15": round(f1_top_fraction(pred, g), 4), "spearman": round(spearman(pred, g), 4),
                     "random_f1_top15": round(float(rand), 4)})
    acc = np.mean([r["true_category"] == r["pred_category"] for r in rows])
    res = {"WARNING": "Diagnostic on Samsung evaluation videos, computed after the model was frozen. "
                      "Not used for training, validation, model selection or tuning.",
           "n_videos": len(rows),
           "label_accuracy": round(float(acc), 4),
           "importance_f1_top15_mean": round(float(np.mean([r["f1_top15"] for r in rows])), 4),
           "importance_spearman_mean": round(float(np.mean([r["spearman"] for r in rows])), 4),
           "random_baseline_f1_top15_mean": round(float(np.mean([r["random_f1_top15"] for r in rows])), 4),
           "per_video": rows}
    (WS / "training/metrics").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(WS / "training/metrics/diagnostic_samsung47.json", "w"), indent=2)
    print({k: v for k, v in res.items() if k != "per_video"})


if __name__ == "__main__":
    main()
