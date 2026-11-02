"""Pocket Lab: film an everyday experiment, let SAM 3.1 track it, measure real physics, level up.

Run locally:  uvicorn app:app --reload
"""
import base64
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np
from fastapi import Body, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

import game
import physics as P
import tracker
from tracker import UserError

PER_IP_DAILY = int(os.getenv("PER_IP_DAILY", "10"))
GLOBAL_DAILY = int(os.getenv("GLOBAL_DAILY", "200"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "120"))
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "2000"))  # about $0.40 of SAM per run at most
BASE = Path(__file__).parent
app = FastAPI(title="Pocket Lab")
game.init()

pool = ThreadPoolExecutor(max_workers=1)  # one video at a time keeps memory low on a small server
JOBS = {}
_lock = threading.Lock()
_day, _used_ip, _used_global = None, {}, 0


# ---- usage limits ----------------------------------------------------------
def client_ip(request: Request):
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


def consume(ip):
    global _day, _used_ip, _used_global
    with _lock:
        today = datetime.now(timezone.utc).date()
        if today != _day:
            _day, _used_ip, _used_global = today, {}, 0
        if _used_global >= GLOBAL_DAILY:
            return "Pocket Lab has hit its free daily limit. Come back tomorrow!"
        if _used_ip.get(ip, 0) >= PER_IP_DAILY:
            return f"You've used your {PER_IP_DAILY} free runs for today. Come back tomorrow!"
        _used_ip[ip] = _used_ip.get(ip, 0) + 1
        _used_global += 1
    return ""


def refund(ip):
    global _used_global
    with _lock:
        if _used_ip.get(ip, 0) > 0:
            _used_ip[ip] -= 1
            _used_global = max(0, _used_global - 1)


def err(status, message):
    return JSONResponse({"status": "error", "message": message}, status_code=status)


# ---- coach and art (Meta models) ---------------------------------------------
def coach_text(m, res, name):
    """A short, kind explanation from Muse Spark. Falls back to a plain sentence."""
    facts = (f"Mission: {m['title']}. Measured gravity g = {res['g']:.2f} m/s2 (accepted 9.81), "
             f"error {res['error']:.1f}%, medal: {res['medal'] or 'none'}. ")
    if res.get("period"):
        facts += f"Swing period {res['period']:.2f} s. "
    if res.get("restitution"):
        facts += f"Bounce keeps {res['restitution'] * 100:.0f}% of its speed each bounce. "
    prompt = ("You are Pocket Lab's friendly lab coach for a curious student called " + name + ". "
              + facts + "In 3 short sentences: say what the result shows, give the most likely reason for any gap "
              "(for example string weight, air drag, the ruler, or the video frame rate), and give one tip for the next try. "
              "Plain words, no markdown, no emoji.")
    try:
        from openai import OpenAI
        client = OpenAI(base_url=os.getenv("COACH_BASE_URL", "https://api.meta.ai/v1"),
                        api_key=os.getenv("COACH_API_KEY") or os.environ["MODEL_API_KEY"], timeout=25)
        model = os.getenv("COACH_MODEL", "muse-spark-1.3")
        try:
            r = client.chat.completions.create(model=model, max_tokens=220,
                                               messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content.strip(), model
        except Exception:
            r = client.responses.create(model=model, input=prompt)
            return r.output_text.strip(), model
    except Exception as e:
        print("coach failed:", repr(e))
        gap = res["error"]
        tip = "Try a longer clip" if m["id"] == "pendulum" else "Try filming at a higher frame rate"
        return (f"You measured g = {res['g']:.2f} m/s², about {gap:.1f}% from the accepted 9.81. "
                f"Small gaps come from the ruler, the frame rate and air drag. {tip} and measure your object carefully.", "offline")


ART_PROMPTS = {
    "pendulum": "a glowing golden pendulum swinging over a dark navy lab, soft light trails, elegant trading card art, no text",
    "drop": "a glowing orb falling through a dark navy lab with motion streaks and sparks, trading card art, no text",
    "bounce": "a glowing ball bouncing on a reflective floor in a dark navy lab, light arcs, trading card art, no text",
}


@app.get("/api/art/{mission}")
def art(mission: str):
    """One card background per mission, made once with Muse Image and cached."""
    if mission not in ART_PROMPTS:
        return Response(status_code=404)
    f = game.DATA_DIR / f"art_{mission}.png"
    if not f.exists():
        if tracker.MOCK or not os.getenv("MODEL_API_KEY"):
            return Response(status_code=404)
        try:
            r = tracker._client().images.generate(model="muse-image-1.0", prompt=ART_PROMPTS[mission], size="1024x1024")
            item = r.data[0]
            data = base64.b64decode(item.b64_json) if getattr(item, "b64_json", None) else None
            if data is None and getattr(item, "url", None):
                import urllib.request
                data = urllib.request.urlopen(item.url, timeout=30).read()
            f.write_bytes(data)
        except Exception as e:
            print("art failed:", repr(e))
            return Response(status_code=404)
    return Response(f.read_bytes(), media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


# ---- the experiment pipeline ---------------------------------------------------
def downsample(t, v, n=160):
    idx = np.linspace(0, len(t) - 1, min(n, len(t))).astype(int)
    return [[round(float(t[i]), 3), round(float(v[i]), 4)] for i in idx]


def run_experiment(job, path, p):
    m = game.MISSIONS[p["mission"]]
    job["stage"], job["pct"] = "Checking your video", 5
    dur, native_fps = tracker.probe(path)
    fps = p["fps"] or native_fps
    if dur * native_fps > MAX_FRAMES:
        raise UserError(f"That clip is a bit long: about {int(dur * native_fps)} frames, and the limit is {MAX_FRAMES}. "
                        f"Trim it to about {int(MAX_FRAMES / native_fps)} seconds.")
    job["stage"], job["pct"] = "Getting your video ready", 15
    mp4 = path + ".mp4"
    tracker.convert(path, mp4, native_fps)
    job["stage"], job["pct"] = "SAM 3.1 is tracking your object", 30
    if tracker.MOCK:
        time.sleep(2)
        rows, n_sam = tracker.mock_rows(p["mission"], fps, p["size_m"], p.get("length_m"))
        n_sam = None
    else:
        rows, n_sam = tracker.track(mp4, p["prompt"])
    header_frames = dur * native_fps
    ratio = tracker.frame_ratio(header_frames, n_sam)
    debug = {"file_seconds": round(dur, 2), "file_fps": round(native_fps, 2), "file_frames_est": int(header_frames),
             "sam_frames": n_sam, "tracked_rows": len(rows), "first_frame": rows[0]["frame"] if rows else None,
             "last_frame": rows[-1]["frame"] if rows else None, "time_scale": ratio, "fps_used": fps}
    print("debug:", debug)
    if len(rows) < 12:
        raise UserError("SAM only saw your object for a moment. Keep it in view and film against a plain background.")
    job["stage"], job["pct"] = "Doing the physics", 85
    ruler = "height" if p["mission"] == "drop" else "min"
    t, x, y, scale = P.to_series(rows, fps / ratio, p["size_m"], ruler)

    res = {"mission": p["mission"], "frames": len(rows), "fps": fps, "frame_ratio": ratio, "debug": debug, "seconds": round(float(t[-1]), 2),
           "inputs": {"object": p["prompt"], "size_cm": round(p["size_m"] * 100, 1), "length_cm": round((p.get("length_m") or 0) * 100, 1)}}
    if p["mission"] == "pendulum":
        r = P.analyze_pendulum(t, x, p["length_m"])
        if "error" in r:
            raise UserError(r["error"])
        res.update(g=r["g"], g_sd=r["g_sd"], period=r["period"], swings=r["swings"], series=downsample(t, x), axis="Sideways position (m)")
        predicted_value, pred_ok = r["period"], None
    elif p["mission"] == "drop":
        r = P.analyze_drop(t, y)
        if "error" in r:
            raise UserError(r["error"])
        res.update(g=r["g"], fall_height=r["fall_height"], fall_time=r["fall_time"], series=downsample(t, y), axis="Height (m)",
                   fit={"t0": r["t0"], "y_top": r["y_top"]})
    else:
        r = P.analyze_bounce(t, y)
        if "g" not in r:
            raise UserError("Couldn't find a clean bounce. Film side-on with the whole bounce and the floor in view.")
        res.update(g=r["g"], g_sd=r["g_sd"], bounces=len(r["flights"]), restitution=r.get("e"), series=downsample(t, y), axis="Height (m)",
                   estimates=[round(v, 2) for v in r["g_estimates"]])
    error, medal = P.rank(res["g"], P.G_TRUE)
    res["error"], res["medal"] = error, medal

    # Did the prediction land close? Pendulum: the swing time. Others: gravity.
    pred = p["prediction"]
    if p["mission"] == "pendulum":
        pred_ok = abs(pred - res["period"]) / res["period"] <= 0.15
    else:
        pred_ok = abs(pred - P.G_TRUE) / P.G_TRUE <= 0.2
    res["prediction"] = {"guess": pred, "ok": bool(pred_ok), "kind": m["prediction"]["kind"],
                         "actual": res["period"] if p["mission"] == "pendulum" else P.G_TRUE}
    job["stage"], job["pct"] = "Your lab coach is writing feedback", 93
    res["xp"] = game.record_run(p["player_id"], p["mission"], res["g"], error, medal, pred, pred_ok)
    pl = game.get_player(p["player_id"])
    res["coach"], res["coach_model"] = coach_text(m, res, pl["name"])
    res["player"] = pl
    return res


def worker(job_id, path, p, ip):
    job = JOBS[job_id]
    try:
        job["result"] = run_experiment(job, path, p)
        job["status"], job["pct"], job["stage"] = "done", 100, "Done"
    except UserError as e:
        refund(ip)
        job["status"], job["message"] = "error", str(e)
    except Exception as e:
        print("job failed:", repr(e))
        refund(ip)
        job["status"], job["message"] = "error", "Something went wrong analyzing that video. Please try again."
    finally:
        for f in (path, path + ".mp4"):
            try:
                os.remove(f)
            except OSError:
                pass


@app.post("/api/jobs")
def create_job(request: Request, video: UploadFile = File(...), player_id: str = Form(...), mission: str = Form(...),
               size_cm: float = Form(...), length_cm: float = Form(0), fps: float = Form(0),
               prompt: str = Form(""), prediction: float = Form(...)):
    m = game.MISSIONS.get(mission)
    if not m:
        return err(400, "Unknown mission.")
    if not game.get_player(player_id):
        return err(400, "Please pick a lab name first.")
    if not 0.3 <= size_cm <= 300:
        return err(400, "Check the object size. Enter it in centimetres.")
    if m["needs_length"] and not 10 <= length_cm <= 400:
        return err(400, "Check the string length. Enter it in centimetres (10 to 400).")
    if fps and not 10 <= fps <= 480:
        return err(400, "Frame rate should be between 10 and 480.")
    word = re.sub(r"[^a-zA-Z \-]", "", prompt).strip()[:30] or m["prompt"]

    ip = client_ip(request)
    msg = consume(ip)
    if msg:
        return err(429, msg)
    tmp = tempfile.mkdtemp(prefix="pl_")
    path = os.path.join(tmp, "upload")
    size = 0
    with open(path, "wb") as f:
        while chunk := video.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                f.close()
                shutil.rmtree(tmp, ignore_errors=True)
                refund(ip)
                return err(400, f"That video is too large (max {MAX_UPLOAD_MB} MB). Trim it or record a shorter clip.")
            f.write(chunk)
    job_id = uuid.uuid4().hex
    now = time.time()
    for k in [k for k, v in JOBS.items() if now - v["created"] > 3600]:
        JOBS.pop(k, None)
    JOBS[job_id] = {"status": "running", "stage": "Uploading", "pct": 2, "created": now}
    params = {"mission": mission, "player_id": player_id, "size_m": size_cm / 100, "length_m": length_cm / 100,
              "fps": fps, "prompt": word, "prediction": prediction}
    pool.submit(worker, job_id, path, params, ip)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return err(404, "That run expired. Please try again.")
    return {k: job.get(k) for k in ("status", "stage", "pct", "message", "result")}


# ---- players, leaderboard ---------------------------------------------------------
@app.get("/api/missions")
def missions():
    ms = [dict(m, quiz=game.quiz_public(m["id"])) for m in game.MISSIONS.values()]
    return {"missions": ms, "locked": game.LOCKED_WORLDS, "ranks": game.RANKS,
            "mock": tracker.MOCK}


@app.post("/api/player")
def player(name: str = Form(...), crew: str = Form(""), player_id: str = Form("")):
    if player_id and game.get_player(player_id):
        game.update_player(player_id, name, crew)
        return game.get_player(player_id)
    return game.get_player(game.create_player(name, crew))


@app.get("/api/player/{pid}")
def get_player(pid: str):
    p = game.get_player(pid)
    return p or err(404, "Unknown player.")


@app.post("/api/quiz")
def quiz(payload: dict = Body(...)):
    pid, mission, answers = payload.get("player_id", ""), payload.get("mission", ""), payload.get("answers") or {}
    if not game.get_player(pid):
        return err(400, "Please pick a lab name first.")
    out = game.grade_quiz(pid, mission, {str(k): v for k, v in answers.items()})
    if out is None:
        return err(400, "Unknown mission.")
    out["player"] = game.get_player(pid)
    return out


@app.post("/api/arcade")
def arcade(player_id: str = Form(...), hits: int = Form(...)):
    if not game.get_player(player_id):
        return err(400, "Please pick a lab name first.")
    out = game.arcade_award(player_id, hits)
    out["player"] = game.get_player(player_id)
    return out


@app.get("/api/leaderboard")
def leaderboard(crew: str = ""):
    return {"rows": game.leaderboard(crew)}


@app.get("/api/usage")
def usage(request: Request):
    return {"per_ip_daily": PER_IP_DAILY}


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")
