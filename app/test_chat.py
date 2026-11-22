"""Accuracy and speed check for the chatbot's question understanding.

Each case: (video, question, expected tool, expected verb, expected noun / expected time).
Correct = the same tool and, for actions, the same vocabulary verb and noun after normalisation.
usage: python test_chat.py <outputs dir> <extra dir> [--no-model]
"""
import json
import os
import sys
import time

from chat import VOCAB, answer, warm_up
from timeline import VideoFacts, parse_time

E, F = "P01_12", "Ski_-Z5a4UQmZLs"
CASES = [
    (E, "When do they open the fridge?", "find_action", "open", "fridge"),
    (E, "when does the person cut the onion", "find_action", "cut", "onion"),
    (E, "Did they pour milk?", "find_action", "pour", "milk"),
    (E, "when is the milk poured", "find_action", "pour", "milk"),
    (E, "At what time do they take a plate?", "find_action", "take", "plate"),
    (E, "how many times do they close the fridge", "find_action", "close", "fridge"),
    (E, "when do they grab the cereal", "find_action", "take", "cereal"),
    (E, "Show me when they wash their hands", "find_action", "wash", "hand"),
    (E, "when do they open a drawer", "find_action", "open", "drawer"),
    (E, "Find the moment they put the cup down", "find_action", "put", "cup"),
    (E, "when do they turn on the tap", "find_action", "turn-on", "tap"),
    (E, "did the person use a knife", "find_action", None, "knife"),
    (E, "when do they pick up the spoon", "find_action", "take", "spoon"),
    (E, "When was the refrigerator closed?", "find_action", "close", "fridge"),
    (E, "Is anything being mixed?", "find_action", "mix", None),
    # several verbs or no vocabulary words: these go to the language model
    (E, "After they take the milk, when do they pour it?", "find_action", "pour", "milk"),
    (E, "when do they put the plate after washing it", "find_action", "put", "plate"),
    (E, "did they take or put the cereal first", "compare_actions", None, "cereal"),
    (E, "When does the person close the fridge after opening it?", "find_action", "close", "fridge"),
    (E, "what happens at 2:03", "what_happens", "2:03", None),
    (E, "What's going on around 1:15?", "what_happens", "1:15", None),
    (E, "what do they do between 0:30 and 1:00", "what_happens", "0:30", "1:00"),
    (E, "what happens at 90 seconds", "what_happens", "90 seconds", None),
    (E, "show the highlights", "list_highlights", None, None),
    (E, "What are the key moments?", "list_highlights", None, None),
    (E, "What is this video about?", "video_summary", None, None),
    (E, "give me an overview", "video_summary", None, None),
    (E, "summarise the video", "list_highlights", None, None),
    (F, "What sport is this?", "video_summary", None, None),
    (F, "what happens at 1:00", "what_happens", "1:00", None),
    (F, "show the best moments", "list_highlights", None, None),
    (F, "when does he jump", "find_action", None, None),
    (F, "What is this video about?", "video_summary", None, None),
    (F, "between 0:50 and 1:10", "what_happens", "0:50", "1:10"),
]


def load(outdir, extradir, vid):
    r = json.load(open(os.path.join(outdir, vid + ".json")))
    p = os.path.join(extradir, vid + ".extra.json")
    return VideoFacts(r, json.load(open(p)) if os.path.exists(p) else None)


def main():
    outdir, extradir = sys.argv[1], sys.argv[2]
    use_model = "--no-model" not in sys.argv
    if use_model:
        t = time.time(); warm_up(); print(f"model warm-up {time.time() - t:.1f}s")
    facts = {v: load(outdir, extradir, v) for v in (E, F)}
    ok, times, srcs = 0, [], {}
    for vid, q, tool, a1, a2 in CASES:
        out = answer(q, facts[vid], use_model)
        args = out["args"]
        good = out["tool"] == tool
        if good and tool == "find_action" and facts[vid].is_cooking:
            good = (a1 is None or VOCAB.verb(args.get("verb")) == a1) and (a2 is None or VOCAB.noun(args.get("noun")) == a2)
        if good and tool == "what_happens":
            good = parse_time(args.get("start")) == parse_time(a1) and (a2 is None or parse_time(args.get("end")) == parse_time(a2))
        ok += good
        times.append(out["ms"]); srcs[out["source"]] = srcs.get(out["source"], 0) + 1
        print(f"{'OK ' if good else 'BAD'} {out['ms']:5d} ms [{out['source']:8s}] {q!r} -> {out['tool']} {args}")
        if not good:
            print("      answer:", out["text"][:160].replace("\n", " | "))
    times.sort()
    print(f"\n{ok}/{len(CASES)} understood correctly; median {times[len(times)//2]} ms, max {times[-1]} ms; by source {srcs}")


if __name__ == "__main__":
    main()
