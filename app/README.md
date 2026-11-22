# Demo app: video summary viewer and chatbot (offline)

A local web page (http://127.0.0.1:8000) that shows the system's results and answers questions about a
video. Nothing leaves the machine; it isn't part of the submission and doesn't change any output.

## Start it

Double-click `start_app.bat`, or:

```bat
cd app
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe server.py
```

Then open http://127.0.0.1:8000. Ollama must be running for the language model (it starts with
Windows; check with `ollama list`, which should show `llama3.2`). Without it, the chatbot still works
in keyword mode.

## What it does

- **Browse:** the 187 results, filtered by cooking or FPV, with the video, the label, and a timeline.
  The timeline shows the highlights in green and, for cooking videos, the detected actions in blue
  (dark = high confidence). For FPV videos it shows the importance curve in orange. Click anywhere to jump.
- **Play highlights:** plays only the highlight segments, one after another.
- **Upload video:** use the button, or drag a file onto the page. The file is saved in
  `Samsung_XR\app_data\uploads\` and analysed with the submission's own `run.sh` (in WSL), then opened.
  Expect about 1.5 min for a 45 s clip, mostly model loading, and about 2 min per 3 min of kitchen video.
- **Chat:** questions about the selected video, answered with timestamps you can click.

## How the chatbot stays fast and accurate

The answer always comes from the video's JSON (`timeline.py`); the language model never writes facts.
`chat.py` only works out which question is being asked:

1. **Patterns** (instant): times ("what happens at 2:30", "between 1:00 and 2:00"), highlights,
   "what is this video about".
2. **Rules** (instant): an action and object read from the question, including "turn the tap on",
   "pick up", "put down". Clauses like "after washing it" are ignored, and "take or put … first" is
   compared.
3. **Local model** (llama3.2 3B via Ollama, about 0.5 s): only when several actions remain ambiguous.
   It picks one of the actions found in the question, never a new one.

Measured: 34/34 on the development questions (`test_chat.py`) and 17/20 on 20 held-out questions
written afterwards and run once (`test_chat_holdout.py`). The 3 held-out misses were fixed after that
run, so 17/20 is the honest score.

**Limits:**
- The detector knows 97 kitchen verbs and 300 objects, so the chatbot can't answer about anything else.
- "Confident" means score 0.15 or higher; a search also lists low-confidence matches, labelled as such.
- FPV videos have only a sport label and an importance score per moment, so it can't say what the
  person is doing there.
