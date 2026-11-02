"""Pocket Lab game layer: missions, medals, XP, ranks, players and the leaderboard."""
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB = DATA_DIR / "pocketlab.db"
_lock = threading.Lock()

MEDAL_XP = {"bronze": 60, "silver": 100, "gold": 150}
MEDAL_RANK = {None: 0, "bronze": 1, "silver": 2, "gold": 3}
ATTEMPT_XP, MAX_ATTEMPT_XP_RUNS, PREDICTION_XP = 15, 5, 25
RANKS = [(0, "Lab Rookie"), (100, "Curious Mind"), (300, "Experimenter"),
         (700, "Scientist"), (1500, "Professor"), (3000, "Nobel Dreamer")]

MISSIONS = {
    "pendulum": {
        "id": "pendulum", "world": "physics", "emoji": "🔑", "status": "live",
        "title": "Swing Time", "tagline": "Hang anything on a string and let it swing.",
        "steps": ["Tie a small object (a key on a lanyard works) to a string and hold the top still.",
                  "Pull it back a little and let go. Don't push it.",
                  "Film from the side for 15 to 25 seconds, keeping the whole swing in view.",
                  "Measure the string from the pinch point to the middle of the object."],
        "size_label": "How wide is the object? (cm, the shorter side)", "size_default": 2.5,
        "needs_length": True, "length_label": "String length (cm), from the pinch point to the middle of the object",
        "prediction": {"kind": "period", "label": "How many seconds will one full swing take?", "unit": "s",
                       "min": 0.3, "max": 4, "step": 0.1, "default": 1.5},
        "constant": {"name": "Gravity from a pendulum", "symbol": "g", "unit": "m/s²"},
        "prompt": "key", "fps_hint": "Normal video is fine. Use 60 fps if your phone offers it.",
        "learn": "A pendulum's swing time depends only on its length and gravity: g = 4π²L / T².",
    },
    "drop": {
        "id": "drop", "world": "physics", "emoji": "🧴", "status": "beta",
        "title": "Drop Zone", "tagline": "Let go of something and time its fall.",
        "steps": ["Hold something small and solid (a bottle, a ball of socks) about shoulder height.",
                  "Film side-on in Slo-mo (120 or 240 fps) and keep the whole fall in view, floor included.",
                  "Open your hand. Don't throw it.",
                  "Measure how tall the object is (the up-down size)."],
        "size_label": "How tall is the object? (cm, the up-down size)", "size_default": 13,
        "needs_length": False,
        "prediction": {"kind": "g", "label": "How fast does Earth pull things down? (m/s every second)", "unit": "m/s²",
                       "min": 1, "max": 25, "step": 0.5, "default": 10},
        "constant": {"name": "Gravity from a falling object", "symbol": "g", "unit": "m/s²"},
        "prompt": "bottle", "fps_hint": "Pick the fps you filmed at. Slo-mo is usually 120 or 240.",
        "learn": "Anything dropped from rest falls a distance of ½ g t², so speed grows the whole way down.",
    },
    "bounce": {
        "id": "bounce", "world": "physics", "emoji": "🏀", "status": "beta",
        "title": "Bounce Lab", "tagline": "Bounce a ball and find out how bouncy it is.",
        "steps": ["Drop a ball onto a hard floor from about waist height.",
                  "Film side-on for 3 or 4 bounces, keeping the floor in view.",
                  "Measure the ball's width."],
        "size_label": "How wide is the ball? (cm)", "size_default": 6.7,
        "needs_length": False,
        "prediction": {"kind": "g", "label": "How fast does Earth pull things down? (m/s every second)", "unit": "m/s²",
                       "min": 1, "max": 25, "step": 0.5, "default": 10},
        "constant": {"name": "Gravity from a bouncing ball", "symbol": "g", "unit": "m/s²"},
        "prompt": "ball", "fps_hint": "60 fps or more gives cleaner bounces.",
        "learn": "Each flight is a parabola, and each bounce keeps a fixed share of the speed.",
    },
}

LOCKED_WORLDS = [
    {"id": "bio", "name": "Biology", "emoji": "🌱", "ideas": ["Sprout Race", "Ant Highway"]},
    {"id": "chem", "name": "Chemistry", "emoji": "🧪", "ideas": ["Fizz Speed", "Ice Melt"]},
    {"id": "space", "name": "Space", "emoji": "🪐", "ideas": ["Shadow Clock", "Moon Tracker"]},
]


def rank_for(xp):
    cur, nxt = RANKS[0], None
    for i, (need, name) in enumerate(RANKS):
        if xp >= need:
            cur = (need, name)
            nxt = RANKS[i + 1] if i + 1 < len(RANKS) else None
    return {"name": cur[1], "floor": cur[0], "next_name": nxt[1] if nxt else None,
            "next_at": nxt[0] if nxt else None}


def _db():
    c = sqlite3.connect(DB, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def init():
    with _lock, _db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS players(id TEXT PRIMARY KEY, name TEXT, crew TEXT, xp INTEGER DEFAULT 0, created REAL);
        CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY AUTOINCREMENT, player_id TEXT, mission TEXT, ts REAL,
            measured REAL, error REAL, medal TEXT, xp INTEGER, predicted REAL, detail TEXT);
        CREATE INDEX IF NOT EXISTS runs_player ON runs(player_id);
        CREATE TABLE IF NOT EXISTS quiz_done(player_id TEXT, qid TEXT, PRIMARY KEY(player_id, qid));
        CREATE TABLE IF NOT EXISTS arcade(player_id TEXT, ts REAL, xp INTEGER);
        """)


def clean(text, n):
    return re.sub(r"[^\w .'\-]", "", (text or "")).strip()[:n]


def create_player(name, crew):
    pid = uuid.uuid4().hex
    with _lock, _db() as c:
        c.execute("INSERT INTO players VALUES(?,?,?,0,?)", (pid, clean(name, 20) or "Scientist", clean(crew, 16).upper(), time.time()))
    return pid


def update_player(pid, name, crew):
    with _lock, _db() as c:
        c.execute("UPDATE players SET name=?, crew=? WHERE id=?", (clean(name, 20) or "Scientist", clean(crew, 16).upper(), pid))


def get_player(pid):
    with _lock, _db() as c:
        p = c.execute("SELECT * FROM players WHERE id=?", (pid,)).fetchone()
        if not p:
            return None
        runs = c.execute("SELECT * FROM runs WHERE player_id=? ORDER BY ts DESC", (pid,)).fetchall()
    bests = {}
    for r in runs:
        b = bests.get(r["mission"])
        key = (MEDAL_RANK[r["medal"]], -(r["error"] if r["error"] is not None else 999))
        if b is None or key > b["key"]:
            bests[r["mission"]] = {"key": key, "measured": r["measured"], "error": r["error"], "medal": r["medal"], "ts": r["ts"]}
    for b in bests.values():
        b.pop("key")
    return {"id": p["id"], "name": p["name"], "crew": p["crew"], "xp": p["xp"], "rank": rank_for(p["xp"]),
            "bests": bests, "runs": len(runs)}


def record_run(pid, mission, measured, error, medal, predicted, prediction_ok, detail=""):
    """Score a run. Returns the XP breakdown. Medal XP only pays for beating your own best."""
    with _lock, _db() as c:
        past = c.execute("SELECT medal FROM runs WHERE player_id=? AND mission=?", (pid, mission)).fetchall()
        best_before = max((MEDAL_RANK[r["medal"]] for r in past), default=0)
        parts = []
        if len(past) < MAX_ATTEMPT_XP_RUNS:
            parts.append(("Finished a run", ATTEMPT_XP))
            if prediction_ok:
                parts.append(("Great prediction", PREDICTION_XP))
        paid_before = sum(MEDAL_XP.get(n, 0) for n, r in MEDAL_RANK.items() if n and r == best_before)
        if medal and MEDAL_RANK[medal] > best_before:
            parts.append((f"New best: {medal} medal", MEDAL_XP[medal] - paid_before))
        total = sum(v for _, v in parts)
        c.execute("INSERT INTO runs(player_id,mission,ts,measured,error,medal,xp,predicted,detail) VALUES(?,?,?,?,?,?,?,?,?)",
                  (pid, mission, time.time(), measured, error, medal, total, predicted, detail))
        before = c.execute("SELECT xp FROM players WHERE id=?", (pid,)).fetchone()["xp"]
        c.execute("UPDATE players SET xp = xp + ? WHERE id=?", (total, pid))
    return {"parts": [{"label": a, "xp": b} for a, b in parts], "total": total,
            "xp_before": before, "xp_after": before + total,
            "rank_before": rank_for(before)["name"], "rank_after": rank_for(before + total)["name"]}


def leaderboard(crew="", limit=25):
    crew = clean(crew, 16).upper()
    with _lock, _db() as c:
        if crew:
            rows = c.execute("SELECT id,name,crew,xp FROM players WHERE crew=? AND xp>0 ORDER BY xp DESC LIMIT ?", (crew, limit)).fetchall()
        else:
            rows = c.execute("SELECT id,name,crew,xp FROM players WHERE xp>0 ORDER BY xp DESC LIMIT ?", (limit,)).fetchall()
    return [{"id": r["id"], "name": r["name"], "crew": r["crew"], "xp": r["xp"], "rank": rank_for(r["xp"])["name"]} for r in rows]


# ---- Science check (quiz) -----------------------------------------------------
QUIZ_XP = 10
QUIZ = {
    "pendulum": [
        {"id": "pen1", "q": "You make the string LONGER. What happens to the swing?",
         "options": ["It swings slower", "It swings faster", "It stays exactly the same"], "answer": 0,
         "why": "A longer string means a longer trip for every swing, so each swing takes more time. Like a longer slide takes longer to ride!"},
        {"id": "pen2", "q": "What keeps pulling the swinging object back toward the middle?",
         "options": ["Wind", "Gravity", "A magnet"], "answer": 1,
         "why": "Gravity pulls everything down. When the object is off to one side, gravity pulls it back toward the middle."},
        {"id": "pen3", "q": "You swap the key for a heavier one. How does the swing time change?",
         "options": ["Heavier swings much faster", "Heavier swings much slower", "It stays about the same"], "answer": 2,
         "why": "Gravity pulls heavy things harder, but heavy things are also harder to move. The two cancel out, so only the string length matters."},
    ],
    "drop": [
        {"id": "drp1", "q": "A heavy ball and a light ball fall from the same height (no air). Which lands first?",
         "options": ["The heavy one", "The light one", "They land together"], "answer": 2,
         "why": "Gravity speeds every object up at the same rate, heavy or light. Galileo figured this out about 400 years ago."},
        {"id": "drp2", "q": "As something falls, its speed...",
         "options": ["Gets faster and faster", "Stays the same", "Gets slower"], "answer": 0,
         "why": "Gravity keeps pulling the whole way down. On Earth it adds about 9.8 meters per second of speed every second."},
        {"id": "drp3", "q": "What do we call the pull that makes things fall to the ground?",
         "options": ["Friction", "Magnetism", "Gravity"], "answer": 2,
         "why": "Gravity is the pull between big things like Earth and everything on it. Earth is so big its pull is strong."},
    ],
    "bounce": [
        {"id": "bnc1", "q": "Why does a ball bounce a little lower each time?",
         "options": ["It loses some energy as sound and heat", "Gravity gets stronger", "The floor gets softer"], "answer": 0,
         "why": "Each bounce squishes the ball a bit. Some energy turns into sound and warmth, so there is less left to go up."},
        {"id": "bnc2", "q": "At the very top of a bounce, how fast is the ball moving up?",
         "options": ["Super fast", "Not at all, just for a moment", "Faster than at the bottom"], "answer": 1,
         "why": "Gravity slows the ball on the way up until it stops for an instant at the top, then it starts falling."},
        {"id": "bnc3", "q": "You bounce the same ball on the Moon, where gravity is weaker. It bounces...",
         "options": ["Lower", "The same", "Higher"], "answer": 2,
         "why": "With weaker gravity, the ball is slowed down less, so it climbs higher and stays up longer."},
    ],
}


def quiz_public(mission):
    return [{"id": q["id"], "q": q["q"], "options": q["options"]} for q in QUIZ.get(mission, [])]


def grade_quiz(pid, mission, answers):
    qs = QUIZ.get(mission)
    if not qs:
        return None
    gained, results = 0, []
    with _lock, _db() as c:
        done = {r["qid"] for r in c.execute("SELECT qid FROM quiz_done WHERE player_id=?", (pid,))}
        for q in qs:
            ok = answers.get(q["id"]) == q["answer"]
            first = ok and q["id"] not in done
            if first:
                c.execute("INSERT INTO quiz_done VALUES(?,?)", (pid, q["id"]))
                gained += QUIZ_XP
            results.append({"id": q["id"], "ok": ok, "answer": q["answer"], "why": q["why"], "xp": QUIZ_XP if first else 0})
        if gained:
            c.execute("UPDATE players SET xp = xp + ? WHERE id=?", (gained, pid))
    return {"results": results, "gained": gained, "correct": sum(r["ok"] for r in results)}


# ---- Arcade ---------------------------------------------------------------------
ARCADE_XP_PER_HIT, ARCADE_DAILY_SESSIONS = 4, 3


def arcade_award(pid, hits):
    hits = max(0, min(int(hits), 5))
    with _lock, _db() as c:
        n = c.execute("SELECT COUNT(*) FROM arcade WHERE player_id=? AND ts>?", (pid, time.time() - 86400)).fetchone()[0]
        xp = hits * ARCADE_XP_PER_HIT if n < ARCADE_DAILY_SESSIONS else 0
        c.execute("INSERT INTO arcade VALUES(?,?,?)", (pid, time.time(), xp))
        if xp:
            c.execute("UPDATE players SET xp = xp + ? WHERE id=?", (xp, pid))
    return {"xp": xp, "capped": n >= ARCADE_DAILY_SESSIONS}
