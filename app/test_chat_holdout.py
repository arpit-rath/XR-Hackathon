"""Held-out questions: written after the rules were finished and run once, with no changes afterwards.
usage: python test_chat_holdout.py <outputs dir> <extra dir>
"""
import sys

import test_chat as T

E, E2, F = "P01_11", "P29_05", "Surfing_7bzMGVKaiCQ"
T.CASES = [
    (E, "When are the knives washed?", "find_action", "wash", "knife"),
    (E, "at what point do they dry their hands", "find_action", "dry", "hand"),
    (E, "Does the person ever close the fridge?", "find_action", "close", "fridge"),
    (E, "when is the glass rinsed", "find_action", "wash", "glass"),
    (E, "Where in the video do they put the plate away?", "find_action", "put", "plate"),
    (E, "what's happening at 5:12?", "what_happens", "5:12", None),
    (E, "tell me what happens from 2:00 to 2:30", "what_happens", "2:00", "2:30"),
    (E, "Which parts were picked as highlights?", "list_highlights", None, None),
    (E, "how many times does the person wash a fork", "find_action", "wash", "fork"),
    (E, "when do they switch the tap off", "find_action", "turn-off", "tap"),
    (E2, "when is the fridge opened", "find_action", "open", "fridge"),
    (E2, "Did they take a cup or put a plate first?", "compare_actions", None, None),
    (E2, "what happens around 15:00", "what_happens", "15:00", None),
    (E2, "when do they throw something in the bin", "find_action", "throw", "bin"),
    (E2, "What kind of video is this?", "video_summary", None, None),
    (F, "what sport is shown?", "video_summary", None, None),
    (F, "what is the most exciting part", "list_highlights", None, None),
    (F, "what happens at 0:45", "what_happens", "0:45", None),
    (F, "when does the surfer fall", "find_action", None, None),
    (F, "give me a summary of this clip", "list_highlights", None, None),
]

if __name__ == "__main__":
    T.E, T.F = E, F
    _load = T.load
    cache = {}

    def main():
        outdir, extradir = sys.argv[1], sys.argv[2]
        T.warm_up()
        facts = {v: _load(outdir, extradir, v) for v in (E, E2, F)}
        ok, times = 0, []
        for vid, q, tool, a1, a2 in T.CASES:
            out = T.answer(q, facts[vid])
            args = out["args"]
            good = out["tool"] == tool
            if good and tool == "find_action" and facts[vid].is_cooking:
                good = (a1 is None or T.VOCAB.verb(args.get("verb")) == a1) and (a2 is None or T.VOCAB.noun(args.get("noun")) == a2)
            if good and tool == "what_happens":
                good = T.parse_time(args.get("start")) == T.parse_time(a1) and (a2 is None or T.parse_time(args.get("end")) == T.parse_time(a2))
            ok += good
            times.append(out["ms"])
            print(f"{'OK ' if good else 'BAD'} {out['ms']:5d} ms [{out['source']:8s}] {q!r} -> {out['tool']} {args}")
            if not good:
                print("      answer:", out["text"][:160].replace("\n", " | "))
        times.sort()
        print(f"\nHELD-OUT: {ok}/{len(T.CASES)} understood correctly; median {times[len(times)//2]} ms, max {times[-1]} ms")

    main()
