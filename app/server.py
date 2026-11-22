"""Local demo app: browse the system's results, watch the highlights, and ask questions about a video.

Runs fully offline on this machine: http://127.0.0.1:8000
    python server.py                       (settings below can be changed with environment variables)

APP_OUTPUTS   submission JSON files (default: ../../submission/team_04/outputs)
APP_EXTRA     extra details per video, <id>.extra.json (default: ../../app_data/extra)
APP_NEW       where videos analysed from the app are written (default: ../../app_data/outputs)
APP_VIDEOS    folders to look for the videos in, separated by ';' (default: E:\\EPIC_videos;E:\\FPVSum_videos)
APP_UPLOADS   where uploaded videos are saved (default: ../../app_data/uploads)
APP_PACKAGE   the team_04 folder whose run.sh analyses new videos (default: ../../submission/team_04)
APP_WSL_PY    Python environment inside WSL for run.sh (default: /root/epic/venv_int)
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chat  # noqa: E402
from timeline import VideoFacts, event_text, fmt_time  # noqa: E402

TOP = HERE.parent.parent  # the Samsung_XR folder
OUTPUTS = Path(os.environ.get("APP_OUTPUTS", TOP / "submission" / "team_04" / "outputs"))
EXTRA = Path(os.environ.get("APP_EXTRA", TOP / "app_data" / "extra"))
NEW = Path(os.environ.get("APP_NEW", TOP / "app_data" / "outputs"))
VIDEO_DIRS = [Path(p) for p in os.environ.get("APP_VIDEOS", r"E:\EPIC_videos;E:\FPVSum_videos").split(";") if p]
PACKAGE = Path(os.environ.get("APP_PACKAGE", TOP / "submission" / "team_04"))
WSL_PY = os.environ.get("APP_WSL_PY", "/root/epic/venv_int")
WSL_DISTRO = os.environ.get("APP_WSL_DISTRO", "Ubuntu-22.04")

app = FastAPI(title="Team 04 video summary")
_videos, _facts, _jobs = {}, {}, {}


def index_videos():
    """video_id -> file path, found once at start-up (and again after an analysis)."""
    for d in VIDEO_DIRS + [UPLOADS]:
        if d.exists():
            for p in d.rglob("*"):
                if p.suffix.lower() in (".mp4", ".mov", ".mkv") and p.is_file():
                    _videos.setdefault(p.stem, p)


def result_path(vid):
    for d in (NEW, OUTPUTS):
        p = d / f"{vid}.json"
        if p.exists():
            return p
    return None


def facts(vid):
    p = result_path(vid)
    if p is None:
        raise HTTPException(404, f"no result for {vid}")
    key = (vid, p.stat().st_mtime)
    if key not in _facts:
        e = EXTRA / f"{vid}.extra.json"
        _facts[key] = VideoFacts(json.load(open(p, encoding="utf-8")), json.load(open(e)) if e.exists() else None)
    return _facts[key]


@app.get("/api/videos")
def list_videos():
    ids = sorted({p.stem for d in (OUTPUTS, NEW) if d.exists() for p in d.glob("*.json")})
    out = []
    for vid in ids:
        r = json.load(open(result_path(vid), encoding="utf-8"))
        out.append({"id": vid, "label": r["label"]["name"], "score": r["label"]["score"], "duration": r["duration_sec"],
                    "highlights": len(r["highlights"]), "has_video": vid in _videos,
                    "new": (NEW / f"{vid}.json").exists()})
    return out


@app.get("/api/video/{vid}")
def video(vid: str):
    f = facts(vid)
    imp = []
    if f.imp_s:  # at most ~400 points for the chart
        step = max(1, len(f.imp_s) // 400)
        imp = [[round(f.imp_t[i], 2), round(f.imp_s[i], 3)] for i in range(0, len(f.imp_s), step)]
    return {"id": vid, "label": f.label, "duration": f.duration, "is_cooking": f.is_cooking,
            "highlights": f.highlights, "importance": imp, "has_video": vid in _videos,
            "summary": f.summary()["text"],
            "events": [{"start": e["start"], "end": e["end"], "text": event_text(e), "score": round(e["score"], 3),
                        "time": fmt_time(e["start"])} for e in f.events]}


@app.get("/api/json/{vid}")
def result_json(vid: str, download: int = 0):
    """The exact submission JSON for this video (as written by run.sh), to view or download."""
    p = result_path(vid)
    if p is None:
        raise HTTPException(404, f"no result for {vid}")
    return FileResponse(p, media_type="application/json", filename=f"{vid}.json" if download else None,
                        content_disposition_type="attachment" if download else "inline")


@app.get("/media/{vid}")
def media(vid: str):
    p = _videos.get(vid)
    if not p:
        raise HTTPException(404, "video file not found")
    return FileResponse(p, media_type="video/mp4")


class Ask(BaseModel):
    id: str
    q: str


@app.post("/api/chat")
def ask(body: Ask):
    if not body.q.strip():
        raise HTTPException(400, "empty question")
    return chat.answer(body.q, facts(body.id))


class Analyze(BaseModel):
    path: str


def to_wsl(p):
    p = str(Path(p).resolve())
    return f"/mnt/{p[0].lower()}{p[2:].replace(chr(92), '/')}" if len(p) > 2 and p[1] == ":" else p


def run_job(job, video):
    vid = video.stem
    NEW.mkdir(parents=True, exist_ok=True)
    EXTRA.mkdir(parents=True, exist_ok=True)
    out, extra = NEW / f"{vid}.json", EXTRA / f"{vid}.extra.json"
    chat.unload()  # free the language model's GPU memory for the pipeline (6 GB GPU)
    cmd = (f". '{WSL_PY}/bin/activate' && '{to_wsl(PACKAGE / 'run.sh')}' '{to_wsl(video)}' '{to_wsl(out)}' "
           f"--extra '{to_wsl(extra)}'")
    t0 = time.time()
    try:
        r = subprocess.run(["wsl", "-d", WSL_DISTRO, "-u", "root", "--exec", "bash", "-c", cmd],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        ok = r.returncode == 0 and out.exists()
        _jobs[job].update(status="done" if ok else "failed", seconds=round(time.time() - t0),
                          log=(r.stdout + r.stderr)[-2000:], id=vid)
        _videos[vid] = video
    except Exception as e:  # WSL missing, etc.
        _jobs[job].update(status="failed", log=str(e))
    threading.Thread(target=chat.warm_up, daemon=True).start()


UPLOADS = Path(os.environ.get("APP_UPLOADS", TOP / "app_data" / "uploads"))
VIDEO_EXT = (".mp4", ".mov", ".mkv")


def busy():
    return any(j["status"] == "running" for j in _jobs.values())


def start_job(p):
    job = uuid.uuid4().hex[:8]
    _jobs[job] = {"status": "running", "id": p.stem, "started": time.time()}
    threading.Thread(target=run_job, args=(job, p), daemon=True).start()
    return job


@app.post("/api/upload")
async def upload(request: Request, name: str):
    """The browser sends the file itself as the request body; it is saved locally, then analysed."""
    stem, ext = os.path.splitext(os.path.basename(name))
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "video"
    if ext.lower() not in VIDEO_EXT:
        raise HTTPException(400, "please upload an .mp4, .mov or .mkv file")
    if busy():
        raise HTTPException(409, "another video is being analysed; try again when it finishes")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    dst = UPLOADS / f"{stem}{ext.lower()}"
    tmp = dst.with_suffix(dst.suffix + ".part")
    size = 0
    with open(tmp, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
            size += len(chunk)
    if size == 0:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, "the file was empty")
    os.replace(tmp, dst)
    return {"job": start_job(dst), "id": dst.stem, "mb": round(size / 2 ** 20, 1)}


@app.post("/api/analyze")
def analyze(body: Analyze):
    p = Path(body.path.strip().strip('"'))
    if not p.is_file():
        raise HTTPException(400, f"not a file: {p}")
    if busy():
        raise HTTPException(409, "another video is being analysed")
    return {"job": start_job(p)}


@app.get("/api/analyze/{job}")
def job_status(job: str):
    j = _jobs.get(job)
    if not j:
        raise HTTPException(404, "no such job")
    return {**j, "elapsed": round(time.time() - j.get("started", time.time()))}


@app.get("/api/status")
def status():
    return {"model": chat.MODEL, "model_ready": chat.warm_up(), "videos_found": len(_videos)}


app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    index_videos()
    threading.Thread(target=chat.warm_up, daemon=True).start()
    print(f"{len(_videos)} video files found; open http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("APP_PORT", 8000)), log_level="warning")
