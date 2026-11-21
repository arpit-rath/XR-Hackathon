"""Map gt_data.mat entries to downloaded videos and check frame-count alignment.

Writes reports/annotation_alignment.csv. Run with the workspace venv (needs scipy).
"""
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import scipy.io as sio

WS = Path(__file__).resolve().parent.parent
ANN = WS / "original_fpvsum" / "fpvsum" / "annotation"
FFPROBE = str(next((WS / "tools").glob("**/bin/ffprobe.exe")))

# gt_data.mat drops the leading underscore of one YouTube ID; this is the only name difference.
GT_NAME_FIXES = {"Jumping_mU-p2wuk90": "Jumping__mU-p2wuk90"}


def count_frames(path):
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_packets",
                        "-show_entries", "stream=nb_read_packets,avg_frame_rate,r_frame_rate",
                        "-show_entries", "format=duration", "-of", "json", str(path)],
                       capture_output=True, text=True)
    d = json.loads(r.stdout)
    s = d["streams"][0]
    n, _, dd = s["avg_frame_rate"].partition("/")
    return int(s["nb_read_packets"]), float(n) / float(dd), s["r_frame_rate"], float(d["format"]["duration"])


def main():
    gt = sio.loadmat(ANN / "gt_data.mat", squeeze_me=True, struct_as_record=False)["gt_data"]
    raw = sio.loadmat(ANN / "raw_data.mat", squeeze_me=True, struct_as_record=False)["raw_data"]
    n_annot = {}
    for x in raw:
        k = GT_NAME_FIXES.get(str(x.Name), str(x.Name))
        n_annot[k] = n_annot.get(k, 0) + 1
    gt_by = {GT_NAME_FIXES.get(str(e.Name), str(e.Name)): e for e in gt}

    rep = {r["samsung_id"]: r for r in csv.DictReader(open(WS / "reports" / "fpvsum_download_report.csv", encoding="utf-8"))}
    rows = []
    for sid, r in rep.items():
        e = gt_by.get(sid)
        row = {"samsung_id": sid, "gt_name": str(e.Name) if e is not None else "",
               "category": r["category"], "video_ok": r["verification_result"] == "PASS",
               "n_annotators": n_annot.get(sid, 0)}
        if e is not None:
            row["gt_frames"] = len(e.Score)
            row["gt_pos_frac"] = round(float(np.mean(e.Binary_score)), 3)
        if row["video_ok"]:
            nf, fps, rfps, dur = count_frames(r["file_path"])
            row.update(video_frames=nf, video_fps=round(fps, 3), r_frame_rate=rfps, video_duration=round(dur, 3))
            if e is not None:
                row["frame_ratio_gt_over_video"] = round(len(e.Score) / nf, 4)
                row["gt_fps_if_full_span"] = round(len(e.Score) / dur, 3)
        rows.append(row)

    fields = ["samsung_id", "gt_name", "category", "video_ok", "n_annotators", "gt_frames", "gt_pos_frac",
              "video_frames", "video_fps", "r_frame_rate", "video_duration", "frame_ratio_gt_over_video",
              "gt_fps_if_full_span"]
    out = WS / "reports" / "annotation_alignment.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)
    print("gt entries not in manifest:", sorted(set(gt_by) - set(rep)))
    for r in rows:
        ratio = r.get("frame_ratio_gt_over_video")
        if ratio is None or abs(ratio - 1) > 0.01:
            print(f'{r["samsung_id"]:26s} ok={r["video_ok"]!s:5s} gt={r.get("gt_frames")} vid={r.get("video_frames")} '
                  f'fps={r.get("video_fps")} ratio={ratio}')


if __name__ == "__main__":
    main()
