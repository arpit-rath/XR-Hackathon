"""Chatbot: understand the question, then answer from the video's facts (timeline.py).

1. Fast path (no model, instant): clear patterns such as "what happens at 2:30", "between 1:00 and
   2:00", "highlights", "what is this video about", "what sport".
2. Otherwise the local model (Ollama, llama3.2 3B) picks ONE tool and its arguments. It never writes
   the answer: the tool's text, computed from the JSON, is the answer, so timestamps are exact.
3. If Ollama isn't running, a keyword fallback finds a verb and an object from the vocabulary.
"""
import json
import os
import re
import time
import urllib.request

from timeline import Vocab, parse_time

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("CHAT_MODEL", "llama3.2")
VOCAB = Vocab()

TOOLS = [
    {"type": "function", "function": {
        "name": "find_action",
        "description": "When does an action happen? Use for questions like 'when do they cut the onion', "
                       "'did they open the fridge', 'how many times do they wash a plate'.",
        "parameters": {"type": "object", "properties": {
            "verb": {"type": "string", "description": "the action as one base verb, e.g. cut, take, wash, open"},
            "noun": {"type": "string", "description": "the object, e.g. onion, fridge, plate; empty if none"}},
            "required": ["verb"]}}},
    {"type": "function", "function": {
        "name": "what_happens",
        "description": "What happens at a time or between two times, e.g. 'what happens at 2:30'.",
        "parameters": {"type": "object", "properties": {
            "start": {"type": "string", "description": "time as m:ss, e.g. 2:30"},
            "end": {"type": "string", "description": "optional end time as m:ss"}},
            "required": ["start"]}}},
    {"type": "function", "function": {
        "name": "list_highlights",
        "description": "The highlights / key moments / summary segments of the video.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "video_summary",
        "description": "What the video is about overall: its type, sport or activity, main actions, length.",
        "parameters": {"type": "object", "properties": {}}}},
]

TIME = r"(\d{1,2}:\d{2}(?::\d{2})?|\d+(?:\.\d+)?\s*(?:s|sec|secs|seconds|m|min|mins|minutes?)(?:\s*\d+\s*(?:s|sec|secs|seconds)?)?)"


def fast_path(q):
    s = q.lower().strip()
    m = re.search(rf"between\s+{TIME}\s+(?:and|to|-)\s+{TIME}", s) or re.search(rf"from\s+{TIME}\s+(?:to|until|-)\s+{TIME}", s)
    if m:
        return "what_happens", {"start": m.group(1), "end": m.group(2)}
    m = re.search(rf"(?:at|around|near)\s+{TIME}", s) or re.fullmatch(rf"{TIME}\??", s)
    if m and not re.search(r"\bwhen\b", s):
        return "what_happens", {"start": m.group(1)}
    if re.search(r"highlight|key moment|best moment|best part|important part|summar(?:y|ise|ize)|recap|"
                 r"(?:most|more) (?:exciting|interesting|important)|exciting part|interesting part|coolest", s):
        return "list_highlights", {}
    if re.search(r"what(?:'s| is) (?:this|the) video|about\??$|what sport|which sport|what activity|what kind of video|overview|what type of video", s):
        return "video_summary", {}
    return None


PHRASAL = [  # multi-word verbs -> EPIC verb (checked before single words)
    (r"\b(?:turn|switch)\w*\b(?:\s+\w+){0,3}?\s+on\b|\b(?:turn|switch)\w*\s+on\b", "turn-on"),
    (r"\b(?:turn|switch)\w*\b(?:\s+\w+){0,3}?\s+off\b|\b(?:turn|switch)\w*\s+off\b", "turn-off"),
    (r"\bturn\w*\s+down\b", "turn-down"), (r"\blet\w*\s+go\b", "let-go"),
    (r"\bpick\w*\s+up\b|\bpick\w*\s+\w+\s+up\b", "take"), (r"\btak\w*\s+out\b", "take"),
    (r"\bput\w*\s+down\b|\bput\w*\s+\w+\s+(?:down|back)\b|\bput\w*\s+(?:\w+\s+)?away\b", "put"),
    (r"\bthrow\w*\s+(?:\w+\s+)?away\b", "throw"),
]
NOT_ACTION = {"happen", "do", "does", "did", "is", "are", "was", "be", "show", "find", "tell", "give", "see"}


SUBCLAUSE = r"\b(?:after|before|while|once|as soon as|until|since)\b[^,?.!]*"


def find_verbs(s):
    verbs = [v for pat, v in PHRASAL if re.search(pat, s)]
    if not verbs:
        verbs = [v for w in re.findall(r"[a-z]+(?:-[a-z]+)?", s) if w not in NOT_ACTION and (v := VOCAB.verb(w))]
    return list(dict.fromkeys(verbs))


def rule_action(q):
    """Verb and object read straight from the question when that is unambiguous, else None."""
    s = q.lower()
    words = re.findall(r"[a-z]+(?:-[a-z]+)?", s)
    verbs = find_verbs(s)
    if len(verbs) > 1:  # "when do they put the plate after washing it": the main clause decides
        main = find_verbs(re.sub(SUBCLAUSE, " ", s))
        if main:
            verbs = main
    noun = None
    for i in range(len(words)):
        two = " ".join(words[i:i + 2])
        if two in VOCAB.noun_alias:  # exact two-word objects first: "chopping board"
            noun = VOCAB.noun_alias[two]
            break
        w = words[i]
        n = VOCAB.noun(w) if len(w) > 2 and w not in NOT_ACTION else None
        if n and VOCAB.verb(w) not in verbs:  # "water" the verb vs the noun
            noun = n
            break
    if len(verbs) == 1:
        return "find_action", {"verb": verbs[0], "noun": noun or ""}
    if len(verbs) > 1 and re.search(r"\b(or|and|first|then|which)\b", s):  # "take or put the cereal first"
        # each verb gets the object right after it, else the shared one: "take a cup or put a plate"
        pairs = []
        for part in re.split(r"\b(?:or|and|then)\b", s):
            pv, pn = find_verbs(part), next((VOCAB.noun(w) for w in re.findall(r"[a-z]+", part)
                                             if w not in NOT_ACTION and VOCAB.noun(w) and not VOCAB.verb(w)), None)
            if pv:
                pairs.append([pv[0], pn])
        if len(pairs) > 1:
            return "compare_actions", {"verbs": [v for v, _ in pairs], "nouns": [n or noun or "" for _, n in pairs], "noun": noun or ""}
        return "compare_actions", {"verbs": verbs, "noun": noun or ""}
    if verbs:  # still several: let the model choose, but only among these
        return None, {"verbs": verbs, "noun": noun or ""}
    if not verbs and noun and re.search(r"\b(when|did|does|do|is|are|was|were|where|how many|any)\b", s):
        return "find_action", {"verb": "", "noun": noun}
    return None


def keyword_fallback(q):
    words = re.findall(r"[a-z]+(?:-[a-z]+)?", q.lower())
    verb = next((VOCAB.verb(w) for w in words if VOCAB.verb(w)), None)
    noun = None
    for i in range(len(words)):
        for j in (2, 1):
            cand = " ".join(words[i:i + j])
            if len(cand) > 2 and VOCAB.noun(cand) and cand not in ("they", "the", "do", "did"):
                noun = VOCAB.noun(cand)
                break
        if noun:
            break
    if verb or noun:
        return "find_action", {"verb": verb or "", "noun": noun or ""}
    return "video_summary", {}


def ask_model(q, facts):
    kind = "a kitchen (cooking) video with detected actions" if facts.is_cooking else \
        f"a first-person {facts.label['name']} video (no named actions, only importance scores)"
    system = (f"You route questions about one video: {kind}, {facts.duration:.0f} s long. "
              "Always call exactly one tool. Use base verbs (cut, not cutting) and single objects (onion, not onions). "
              "Times go in m:ss.")
    body = {"model": MODEL, "stream": False, "keep_alive": "30m", "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": q}], "tools": TOOLS}
    req = urllib.request.Request(f"{OLLAMA}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=30))
    calls = r.get("message", {}).get("tool_calls") or []
    if not calls:
        return None
    f = calls[0]["function"]
    args = f.get("arguments") or {}
    if isinstance(args, str):
        args = json.loads(args or "{}")
    return f["name"], args


def answer(q, facts, use_model=True):
    t0 = time.time()
    route, source = fast_path(q), "pattern"
    if route is None and not facts.is_cooking:
        route = ("find_action", {})  # FPV videos have no named actions: explain that, no model needed
    hint = {}
    if route is None:
        route, source = rule_action(q), "rules"
        if route and route[0] is None:  # several candidate verbs: the model picks one of them
            hint, route = route[1], None
    if route is None and use_model:
        try:
            route, source = ask_model(q, facts), "model"
        except Exception:
            route = None
        if route and route[0] == "find_action" and hint:
            args = dict(route[1])
            if VOCAB.verb(args.get("verb")) not in hint["verbs"]:
                args["verb"] = hint["verbs"][0]
            if not VOCAB.noun(args.get("noun")) and hint["noun"]:
                args["noun"] = hint["noun"]
            route = ("find_action", args)
    if route is None and hint:
        route, source = ("find_action", {"verb": hint["verbs"][0], "noun": hint["noun"]}), "rules"
    if route is None:
        route, source = keyword_fallback(q), "keywords"
    name, args = route
    # A model that picks find_action with words that aren't actions (e.g. "is about") -> overview instead.
    if facts.is_cooking and name == "find_action" and not VOCAB.verb(args.get("verb")) and not VOCAB.noun(args.get("noun")):
        kw = keyword_fallback(q)
        name, args = kw if kw[0] == "find_action" else ("video_summary", {})
    if name == "what_happens" and parse_time(args.get("start")) is None:
        name, args = "video_summary", {}
    if name == "compare_actions":
        out = facts.compare_actions(args["verbs"], args.get("noun") or None, VOCAB, args.get("nouns"))
    elif name == "find_action":
        out = facts.find_action(args.get("verb") or None, args.get("noun") or None, VOCAB)
    elif name == "what_happens":
        out = facts.what_happens(args.get("start"), args.get("end"))
    elif name == "list_highlights":
        out = facts.list_highlights()
    else:
        out = facts.summary()
    out.update(tool=name, args=args, source=source, ms=round((time.time() - t0) * 1000))
    return out


def warm_up():
    """Load the model into GPU memory so the first question is fast."""
    try:
        body = {"model": MODEL, "prompt": "", "keep_alive": "30m"}
        urllib.request.urlopen(urllib.request.Request(f"{OLLAMA}/api/generate", json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}), timeout=60).read()
        return True
    except Exception:
        return False


def unload():
    """Free the model's GPU memory (before running the video pipeline on a 6 GB GPU)."""
    try:
        body = {"model": MODEL, "prompt": "", "keep_alive": 0}
        urllib.request.urlopen(urllib.request.Request(f"{OLLAMA}/api/generate", json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}), timeout=30).read()
    except Exception:
        pass
