#!/usr/bin/env python3
"""Validator for PRISM 2026 Track 04 output JSON files (standard library only).

NOTE: the organisers' validate.py was missing from our copy of the student package. This file
re-implements the rules stated in TEAM_README.md (sections 3 and 4). It may be stricter or looser
than the official script in places; when the official one is available, run that too.

Errors (file is invalid -> scores 0 on every metric for that video):
  * the 6 top-level fields: team_id, video_id, duration_sec, label, actions, highlights
  * types: strings non-empty, numbers finite (booleans are not numbers), label is an object,
    actions / highlights are arrays
  * duration_sec > 0; label.score and every action score in [0, 1]
  * every segment is [start, end] with 0 <= start < end <= duration_sec + 0.5
  * actions: at most 2000 items, each {segment, verb, noun, score}
  * highlights: at least 1 item, sorted by start, no overlap, at least one segment >= 4.0 s
  * with a manifest: video_id is in it, matches the file name, duration_sec within +-0.5 s of it
  * with supported_classes.json: label.name is one of its labels
  * with --dir: team_id identical across files; with a manifest, every manifest video has a file
Warnings (valid, but worth a look):
  * a highlight shorter than 4.0 s (ignored during scoring)
  * total highlight length above 15% of duration_sec (soft penalty on coverage)
  * an action verb / noun outside supported_classes.json, or "other" on an EPIC video
  * timestamps with more than 2 decimals

Usage:
  python validate.py --file outputs/P01_11.json -v
  python validate.py --dir outputs/
  (--manifest / --classes default to manifest.csv / supported_classes.json next to this script)
"""
import argparse
import csv
import json
import math
import os
import sys

FIELDS = ("team_id", "video_id", "duration_sec", "label", "actions", "highlights")
MAX_ACTIONS = 2000
END_SLACK = 0.5
MIN_HIGHLIGHT = 4.0
BUDGET = 0.15
DURATION_TOL = 0.5
HERE = os.path.dirname(os.path.abspath(__file__))


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def is_str(x):
    return isinstance(x, str) and x.strip() != ""


def decimals(x):
    s = repr(float(x))
    return len(s.split(".")[1]) if "." in s and "e" not in s else 0


class Report:
    def __init__(self, name):
        self.name, self.errors, self.warnings = name, [], []

    def err(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    @property
    def ok(self):
        return not self.errors


def check_segment(seg, duration, where, rep):
    """Returns (start, end) if the segment is valid, else None (and records the error)."""
    if not isinstance(seg, list) or len(seg) != 2:
        rep.err(f"{where}: segment must be a list [start, end], got {seg!r}")
        return None
    s, e = seg
    if not (is_num(s) and is_num(e)):
        rep.err(f"{where}: segment values must be finite numbers, got {seg!r}")
        return None
    if not (0 <= s < e):
        rep.err(f"{where}: need 0 <= start < end, got {seg!r}")
        return None
    if duration is not None and e > duration + END_SLACK:
        rep.err(f"{where}: end {e} > duration_sec + {END_SLACK} ({duration + END_SLACK:.2f})")
        return None
    if decimals(s) > 2 or decimals(e) > 2:
        rep.warn(f"{where}: timestamps should have at most 2 decimals, got {seg!r}")
    return s, e


def validate(path, manifest=None, classes=None):
    rep = Report(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # unreadable or not JSON
        rep.err(f"cannot read JSON: {exc}")
        return rep, None
    if not isinstance(data, dict):
        rep.err("top level must be a JSON object")
        return rep, None

    for k in FIELDS:
        if k not in data:
            rep.err(f"missing field '{k}'")
    if not rep.ok:
        return rep, data

    if not is_str(data["team_id"]):
        rep.err("team_id must be a non-empty string")
    vid = data["video_id"]
    if not is_str(vid):
        rep.err("video_id must be a non-empty string")
    else:
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem != vid:
            rep.err(f"video_id '{vid}' does not match file name '{stem}.json'")

    duration = data["duration_sec"]
    if not is_num(duration) or duration <= 0:
        rep.err(f"duration_sec must be a finite number > 0, got {duration!r}")
        duration = None

    dataset = None
    if manifest is not None and is_str(vid):
        if vid not in manifest:
            rep.err(f"video_id '{vid}' is not in manifest.csv")
        else:
            ref, dataset = manifest[vid]
            if duration is not None and abs(duration - ref) > DURATION_TOL:
                rep.err(f"duration_sec {duration} is not within +-{DURATION_TOL} s of manifest {ref}")

    label = data["label"]
    if not isinstance(label, dict):
        rep.err("label must be an object {name, score}")
    else:
        if not is_str(label.get("name")):
            rep.err("label.name must be a non-empty string")
        elif classes is not None and label["name"] not in classes["labels"]:
            rep.err(f"label.name '{label['name']}' is not in supported_classes.json labels")
        elif dataset == "epic_kitchens_100" and label["name"] != "cooking":
            rep.warn(f"EPIC video labelled '{label['name']}' (README says use 'cooking')")
        if not (is_num(label.get("score")) and 0 <= label["score"] <= 1):
            rep.err(f"label.score must be a number in [0, 1], got {label.get('score')!r}")

    actions = data["actions"]
    if not isinstance(actions, list):
        rep.err("actions must be an array")
    else:
        if len(actions) > MAX_ACTIONS:
            rep.err(f"actions has {len(actions)} items (max {MAX_ACTIONS})")
        unknown = set()
        for i, a in enumerate(actions):
            where = f"actions[{i}]"
            if not isinstance(a, dict):
                rep.err(f"{where}: must be an object")
                continue
            missing = [k for k in ("segment", "verb", "noun", "score") if k not in a]
            if missing:
                rep.err(f"{where}: missing {missing}")
                continue
            check_segment(a["segment"], duration, where, rep)
            for k in ("verb", "noun"):
                if not is_str(a[k]):
                    rep.err(f"{where}: {k} must be a non-empty string")
                elif classes is not None and a[k] not in classes[k + "s"]:
                    unknown.add(f"{k} '{a[k]}'")
                elif dataset == "epic_kitchens_100" and a[k] == "other":
                    unknown.add(f"{k} 'other' on an EPIC video")
            if not (is_num(a["score"]) and 0 <= a["score"] <= 1):
                rep.err(f"{where}: score must be a number in [0, 1], got {a['score']!r}")
        if unknown:
            rep.warn("actions use labels outside the vocabulary: " + ", ".join(sorted(unknown)[:10]))

    highlights = data["highlights"]
    if not isinstance(highlights, list):
        rep.err("highlights must be an array")
    elif len(highlights) == 0:
        rep.err("highlights must contain at least 1 segment")
    else:
        segs = []
        for i, h in enumerate(highlights):
            where = f"highlights[{i}]"
            if not isinstance(h, dict) or "segment" not in h:
                rep.err(f"{where}: must be an object with 'segment'")
                continue
            se = check_segment(h["segment"], duration, where, rep)
            if se:
                segs.append((i, *se))
        for (i, s0, e0), (j, s1, e1) in zip(segs, segs[1:]):
            if s1 < s0:
                rep.err(f"highlights[{j}] starts before highlights[{i}]: not chronologically sorted")
            elif s1 < e0:
                rep.err(f"highlights[{i}] and highlights[{j}] overlap ({e0} > {s1})")
        if segs:
            lengths = [e - s for _, s, e in segs]
            if max(lengths) < MIN_HIGHLIGHT:
                rep.err(f"no highlight segment is >= {MIN_HIGHLIGHT} s (longest {max(lengths):.2f} s)")
            short = [i for (i, _, _), L in zip(segs, lengths) if L < MIN_HIGHLIGHT]
            if short:
                rep.warn(f"{len(short)} highlight(s) shorter than {MIN_HIGHLIGHT} s are ignored in scoring: "
                         f"indices {short[:10]}")
            if duration is not None:
                total = sum(lengths)
                if total > BUDGET * duration:
                    rep.warn(f"highlights total {total:.2f} s = {100 * total / duration:.1f}% of duration "
                             f"(budget {100 * BUDGET:.0f}%; soft penalty on coverage)")
    return rep, data


def load_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["video_id"]: (float(r["duration_sec"]), r["dataset"]) for r in csv.DictReader(f)}


def load_classes(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {"labels": set(d["labels"]), "verbs": set(d["verbs"]), "nouns": set(d["nouns"])}


def print_report(rep, verbose):
    status = "OK  " if rep.ok else "FAIL"
    print(f"[{status}] {rep.name}" + (f"  ({len(rep.warnings)} warning(s))" if rep.warnings else ""))
    for e in rep.errors[:20]:
        print(f"    error:   {e}")
    if len(rep.errors) > 20:
        print(f"    ... {len(rep.errors) - 20} more errors")
    if verbose:
        for w in rep.warnings:
            print(f"    warning: {w}")


def main():
    ap = argparse.ArgumentParser(description="Validate Track 04 output JSON files.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", help="validate one JSON file")
    g.add_argument("--dir", help="validate every *.json in a folder")
    ap.add_argument("--manifest", default=os.path.join(HERE, "manifest.csv"),
                    help="manifest.csv for video_id / duration checks ('' to skip)")
    ap.add_argument("--classes", default=os.path.join(HERE, "supported_classes.json"),
                    help="supported_classes.json for label / vocabulary checks ('' to skip)")
    ap.add_argument("-v", "--verbose", action="store_true", help="also print warnings")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest) if args.manifest and os.path.exists(args.manifest) else None
    classes = load_classes(args.classes) if args.classes and os.path.exists(args.classes) else None
    if manifest is None:
        print("note: no manifest.csv found; skipping video_id / duration checks")
    if classes is None:
        print("note: no supported_classes.json found; skipping label / vocabulary checks")

    if args.file:
        rep, _ = validate(args.file, manifest, classes)
        print_report(rep, args.verbose)
        sys.exit(0 if rep.ok else 1)

    files = sorted(os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.endswith(".json"))
    reports, team_ids = [], {}
    for path in files:
        rep, data = validate(path, manifest, classes)
        if isinstance(data, dict) and is_str(data.get("team_id")):
            team_ids.setdefault(data["team_id"], []).append(os.path.basename(path))
        reports.append(rep)
    if len(team_ids) > 1:
        majority = max(team_ids, key=lambda t: len(team_ids[t]))
        for rep in reports:
            name = os.path.basename(rep.name)
            for t, names in team_ids.items():
                if t != majority and name in names:
                    rep.err(f"team_id '{t}' differs from the rest ('{majority}')")
    for rep in reports:
        if not rep.ok or args.verbose:
            print_report(rep, args.verbose)

    missing = []
    if manifest is not None:
        have = {os.path.splitext(os.path.basename(p))[0] for p in files}
        missing = sorted(set(manifest) - have)
    n_ok = sum(r.ok for r in reports)
    print(f"\n{n_ok}/{len(reports)} files valid; "
          f"{sum(len(r.warnings) for r in reports)} warning(s)"
          + (f"; {len(missing)} manifest video(s) have no file" if manifest is not None else ""))
    if missing:
        print("missing: " + ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))
    sys.exit(0 if n_ok == len(reports) and not missing else 1)


if __name__ == "__main__":
    main()
