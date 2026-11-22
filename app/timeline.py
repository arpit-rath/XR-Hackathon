"""Turn one submission JSON (+ optional extra info) into facts the chatbot can answer from.

Every answer is computed here from the JSON, never by the language model, so timestamps and labels
are exactly what the system detected. Thresholds were fixed before testing:
  * an "event" is a detected action with score >= EVENT_SCORE, after merging repeats of the same
    verb-noun pair that overlap or are less than MERGE_GAP s apart;
  * a search for a specific action looks at everything >= SEARCH_SCORE and labels low scores.
"""
import difflib
import json
import os
import re

EVENT_SCORE, SEARCH_SCORE, MERGE_GAP = 0.15, 0.05, 1.0
HIGH, MEDIUM = 0.30, 0.15

HERE = os.path.dirname(os.path.abspath(__file__))
_VOCAB_PATH = os.path.join(os.path.dirname(HERE), "package", "supported_classes.json")

VERB_SYNONYMS = {
    "chop": "cut", "slice": "cut", "dice": "cut", "grab": "take", "get": "take", "pick": "take",
    "pickup": "take", "fetch": "take", "place": "put", "set": "put", "putdown": "put", "rinse": "wash",
    "clean": "wash", "stir": "mix", "whisk": "mix", "switch-on": "turn-on", "switchon": "turn-on",
    "switch-off": "turn-off", "switchoff": "turn-off", "start": "turn-on", "stop": "turn-off",
    "toss": "throw", "discard": "throw", "drink": "drink", "fry": "fry", "boil": "boil", "unlock": "open",
    "shut": "close", "spread": "apply", "rub": "scrub", "flipped": "flip", "peeled": "peel",
}
NOUN_SYNONYMS = {
    "refrigerator": "fridge", "faucet": "tap", "sink": "sink", "dish": "plate", "dishes": "plate",
    "mug": "cup", "cupboard": "cupboard", "cabinet": "cupboard", "stove": "hob", "cooker": "hob",
    "frying pan": "pan", "chopping board": "board:chopping", "cutting board": "board:chopping",
    "hands": "hand", "water": "water", "rubbish": "bin", "trash": "bin", "garbage": "bin",
}


def fmt_time(t):
    t = max(0.0, float(t))
    m, s = divmod(int(round(t)), 60)
    return f"{m}:{s:02d}"


def parse_time(text):
    """'2:30' / '150' / '2 min 30' / '2.5 minutes' -> seconds (float) or None."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text).strip().lower()
    m = re.fullmatch(r"(\d+):(\d{1,2})(?::(\d{1,2}))?", s)
    if m:
        a, b, c = m.groups()
        return int(a) * 3600 + int(b) * 60 + int(c) if c else int(a) * 60 + int(b)
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:m|min|mins|minutes?)\s*(?:(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?)?)?", s)
    if m:
        return float(m.group(1)) * 60 + float(m.group(2) or 0)
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?)?", s)
    if m:
        return float(m.group(1))
    return None


def noun_display(n):
    """EPIC writes compound nouns backwards: 'board:chopping' -> 'chopping board'."""
    return " ".join(reversed(n.split(":"))) if ":" in n else n


def verb_display(v):
    return v.replace("-", " ")


class Vocab:
    def __init__(self, path=_VOCAB_PATH):
        d = json.load(open(path, encoding="utf-8"))
        self.verbs = [v for v in d["verbs"] if v != "other"]
        self.nouns = [n for n in d["nouns"] if n != "other"]
        self.noun_alias = {}
        for n in self.nouns:
            for a in {n, noun_display(n), n.split(":")[0], n.replace(":", " ")}:
                self.noun_alias.setdefault(a, n)
        for a, n in NOUN_SYNONYMS.items():
            if n in self.nouns:
                self.noun_alias[a] = n

    @staticmethod
    def _stems(w):
        w = w.lower().strip().replace("_", " ")
        out = [w, w.replace(" ", "-"), w.replace(" ", "")]
        for suf, rep in (("ing", ""), ("ing", "e"), ("ed", ""), ("ed", "e"), ("es", ""), ("s", ""), ("ies", "y"),
                         ("ves", "fe"), ("ves", "f")):
            if w.endswith(suf) and len(w) > len(suf) + 2:
                out.append(w[: -len(suf)] + rep)
        if len(w) > 4 and w[-1] == w[-2]:  # cutt(ing) -> cut
            out.append(w[:-1])
        out += [x[:-1] for x in out if len(x) > 4 and x[-1] == x[-2]]
        return out

    def verb(self, word):
        if not word:
            return None
        for w in self._stems(word):
            if w in self.verbs:
                return w
            if w in VERB_SYNONYMS and VERB_SYNONYMS[w] in self.verbs:
                return VERB_SYNONYMS[w]
        first = word.lower().split()[0]
        if first != word.lower():
            return self.verb(first)
        m = difflib.get_close_matches(word.lower(), self.verbs, n=1, cutoff=0.85)
        return m[0] if m else None

    def noun(self, word):
        if not word:
            return None
        w0 = re.sub(r"^(the|a|an|some|his|her|their)\s+", "", word.lower().strip())
        for w in self._stems(w0) + [w0]:
            if w in self.noun_alias:
                return self.noun_alias[w]
        last = w0.split()[-1] if w0.split() else w0
        if last != w0:
            return self.noun(last)
        if len(w0) < 5 or self.verb(w0):  # no fuzzy guesses for short words or verbs ("open" is not "pen")
            return None
        m = difflib.get_close_matches(w0, [a for a in self.noun_alias if " " not in a], n=1, cutoff=0.88)
        return self.noun_alias[m[0]] if m else None


def merge(actions, min_score):
    """Merge repeats of the same pair that overlap or nearly touch; keep the best score."""
    acts = sorted((a for a in actions if a["score"] >= min_score), key=lambda a: (a["verb"], a["noun"], a["segment"][0]))
    out = []
    for a in acts:
        s, e = a["segment"]
        last = out[-1] if out else None
        if last and last["verb"] == a["verb"] and last["noun"] == a["noun"] and s <= last["end"] + MERGE_GAP:
            last["end"] = max(last["end"], e)
            last["score"] = max(last["score"], a["score"])
        else:
            out.append({"verb": a["verb"], "noun": a["noun"], "start": s, "end": e, "score": a["score"]})
    return sorted(out, key=lambda x: x["start"])


def confidence(score):
    return "high" if score >= HIGH else "medium" if score >= MEDIUM else "low"


def event_text(e):
    return f"{verb_display(e['verb'])} {noun_display(e['noun'])}"


class VideoFacts:
    """Everything the app shows and the chatbot answers for one video."""

    def __init__(self, result, extra=None):
        self.r, self.extra = result, extra or {}
        self.id, self.duration = result["video_id"], float(result["duration_sec"])
        self.label = result["label"]
        self.is_cooking = self.label["name"] == "cooking"
        self.highlights = [h["segment"] for h in result["highlights"]]
        acts = result.get("actions", [])
        self.events = merge(acts, EVENT_SCORE)
        self.searchable = merge(acts, SEARCH_SCORE)
        imp = self.extra.get("importance") or {}
        self.imp_t, self.imp_s = imp.get("timestamps_sec", []), imp.get("scores", [])

    # ---- answers (each returns {"text", "refs": [{"t", "label"}]}) ----
    def summary(self):
        if self.is_cooking:
            from collections import Counter
            top = Counter(event_text(e) for e in self.events).most_common(5)
            acts = ", ".join(f"{t} ({n}x)" for t, n in top) if top else "no confident actions"
            text = (f"This is a {fmt_time(self.duration)} kitchen (cooking) video, label confidence "
                    f"{self.label['score']:.2f}. I found {len(self.events)} confident actions; the most frequent are "
                    f"{acts}. The summary keeps {len(self.highlights)} highlight(s), "
                    f"{self.highlight_seconds():.0f} s in total.")
        else:
            probs = self.extra.get("category_probs") or {}
            alt = sorted(((p, k) for k, p in probs.items() if k != self.label["name"]), reverse=True)[:2]
            alt_t = (" Next most likely: " + ", ".join(f"{k} ({p:.2f})" for p, k in alt) + ".") if alt else ""
            text = (f"This is a {fmt_time(self.duration)} first-person {self.label['name']} video "
                    f"(confidence {self.label['score']:.2f}).{alt_t} The summary keeps {len(self.highlights)} "
                    f"highlight(s), {self.highlight_seconds():.0f} s in total.")
        return {"text": text, "refs": [{"t": s, "label": fmt_time(s)} for s, _ in self.highlights[:3]]}

    def highlight_seconds(self):
        return sum(e - s for s, e in self.highlights)

    def list_highlights(self):
        if not self.highlights:
            return {"text": "There are no highlights for this video.", "refs": []}
        parts = []
        for i, (s, e) in enumerate(self.highlights, 1):
            what = ""
            if self.is_cooking:
                inside = [x for x in self.events if x["start"] < e and x["end"] > s]
                if inside:
                    best = max(inside, key=lambda x: x["score"])
                    what = f" ({event_text(best)})"
            parts.append(f"{i}. {fmt_time(s)}–{fmt_time(e)}{what}")
        text = f"{len(self.highlights)} highlight(s), {self.highlight_seconds():.0f} s of {fmt_time(self.duration)}:\n" + "\n".join(parts)
        return {"text": text, "refs": [{"t": s, "label": fmt_time(s)} for s, _ in self.highlights]}

    def find_action(self, verb=None, noun=None, vocab=None):
        if not self.is_cooking:
            return self._no_actions()
        v, n = (vocab.verb(verb), vocab.noun(noun)) if vocab else (verb, noun)
        if verb and not v and noun and not n:
            return {"text": f"\"{verb} {noun}\" isn't something the detector knows.", "refs": []}
        if verb and not v:
            return {"text": f"\"{verb}\" isn't one of the 97 actions the detector knows (e.g. take, put, cut, wash, open).", "refs": []}
        if noun and not n:
            return {"text": f"\"{noun}\" isn't one of the 300 objects the detector knows.", "refs": []}
        hits = [e for e in self.searchable if (not v or e["verb"] == v) and (not n or e["noun"] == n)]
        what = " ".join(x for x in (verb_display(v) if v else "", noun_display(n) if n else "") if x)
        if not hits:
            return {"text": f"I didn't detect \"{what}\" in this video.", "refs": []}
        best = sorted(hits, key=lambda e: -e["score"])[:6]
        best.sort(key=lambda e: e["start"])
        lines = [f"- {fmt_time(e['start'])}–{fmt_time(e['end'])}: {event_text(e)} ({confidence(e['score'])} confidence)" for e in best]
        more = f"\n…and {len(hits) - len(best)} weaker match(es)." if len(hits) > len(best) else ""
        conf = [e for e in hits if e["score"] >= MEDIUM]
        head = (f"\"{what}\" happens {len(conf)} time(s) with medium or high confidence" if conf
                else f"Only low-confidence matches for \"{what}\"")
        return {"text": head + ":\n" + "\n".join(lines) + more,
                "refs": [{"t": e["start"], "label": fmt_time(e["start"])} for e in best]}

    def compare_actions(self, verbs, noun=None, vocab=None, nouns=None):
        """'Did they take or put the cereal first?': each action's first confident time, in order.
        nouns: one object per verb ("take a cup or put a plate"); else `noun` for all."""
        if not self.is_cooking:
            return self._no_actions()
        firsts, refs = [], []
        for i, v in enumerate(verbs):
            nn = (nouns[i] if nouns and i < len(nouns) else None) or noun
            n = vocab.noun(nn) if vocab and nn else nn
            hits = [e for e in self.searchable if e["verb"] == v and (not n or e["noun"] == n) and e["score"] >= MEDIUM]
            if hits:
                e = min(hits, key=lambda x: x["start"])
                firsts.append((e["start"], f"{event_text(e)} first at {fmt_time(e['start'])} ({len(hits)} time(s) in all)"))
                refs.append({"t": e["start"], "label": fmt_time(e["start"])})
            else:
                firsts.append((float("inf"), f"{verb_display(v)} {noun_display(n) if n else ''}: not detected with confidence".strip()))
        firsts.sort()
        head = f"{firsts[0][1].split(' first at')[0]} comes first." if firsts and firsts[0][0] != float("inf") else "Neither was detected with confidence."
        return {"text": head + "\n" + "\n".join(f"- {t}" for _, t in firsts), "refs": refs}

    def what_happens(self, start=None, end=None):
        t0 = parse_time(start)
        if t0 is None:
            return {"text": "Which time? For example \"what happens at 1:30\".", "refs": []}
        t1 = parse_time(end)
        if t0 > self.duration:
            return {"text": f"The video is only {fmt_time(self.duration)} long.", "refs": []}
        lo, hi = (t0, t1) if t1 is not None and t1 > t0 else (t0 - 3, t0 + 3)
        span = f"{fmt_time(t0)}–{fmt_time(t1)}" if t1 is not None and t1 > t0 else f"around {fmt_time(t0)}"
        in_hl = any(s < hi and e > lo for s, e in self.highlights)
        hl = " This part is in the highlights." if in_hl else ""
        if not self.is_cooking:
            if self.imp_s:
                idx = [i for i, t in enumerate(self.imp_t) if lo <= t <= hi] or [min(range(len(self.imp_t)), key=lambda i: abs(self.imp_t[i] - t0))]
                val = max(self.imp_s[i] for i in idx)
                rank = sum(s <= val for s in self.imp_s) / len(self.imp_s)
                return {"text": f"{span}: importance {val:.2f}, higher than {rank:.0%} of the video.{hl} "
                                f"(For {self.label['name']} videos the system scores how interesting each moment is; "
                                "it doesn't name actions.)", "refs": [{"t": t0, "label": fmt_time(t0)}]}
            return self._no_actions(f"{span}.{hl}")
        near = [e for e in self.searchable if e["start"] < hi and e["end"] > lo and e["score"] >= 0.10]
        if not near:
            return {"text": f"{span}: no confident action detected.{hl}", "refs": [{"t": t0, "label": fmt_time(t0)}]}
        near = sorted(near, key=lambda e: -e["score"])[:6]
        near.sort(key=lambda e: e["start"])
        lines = [f"- {fmt_time(e['start'])}–{fmt_time(e['end'])}: {event_text(e)} ({confidence(e['score'])})" for e in near]
        return {"text": f"{span}:{hl}\n" + "\n".join(lines),
                "refs": [{"t": e["start"], "label": fmt_time(e["start"])} for e in near]}

    def _no_actions(self, prefix=""):
        return {"text": (prefix + " " if prefix else "") +
                f"This is a {self.label['name']} video. The action detector only knows kitchen actions, so "
                "the system gives a sport label and an importance score per moment here, not named actions. "
                "Try \"show the highlights\" or \"what happens at 1:00\".", "refs": []}
