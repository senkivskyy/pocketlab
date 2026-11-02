"""Video prep and SAM 3.1 tracking for Pocket Lab."""
import os
import re
import shutil
import subprocess

import imageio_ffmpeg
import numpy as np

FFMPEG = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
MOCK = os.getenv("POCKETLAB_MOCK") == "1"

FRAME_RE = re.compile(r"<(\d+)f>")
BOX_RE = re.compile(r"(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)\|>")


class UserError(Exception):
    """A problem the person can fix (shown to them as is)."""


def probe(path):
    """(duration seconds, frames per second) of the file, read from ffmpeg's header dump."""
    out = subprocess.run([FFMPEG, "-hide_banner", "-i", path], capture_output=True, text=True, timeout=30).stderr
    d = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", out)
    f = re.search(r"(\d+(?:\.\d+)?)\s*fps", out)
    if not d or "Video:" not in out:
        raise UserError("That doesn't look like a video. Try an MP4 or MOV from your phone.")
    dur = int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3))
    return dur, float(f.group(1)) if f else 30.0


def convert(src, dst):
    """Re-encode to a small H.264 MP4 (SAM needs it) without dropping or adding frames.

    Settings are chosen to keep memory low on a small server: one decoder thread,
    a modest output size and the lightest x264 preset.
    """
    base = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-threads", "1", "-i", src, "-an",
            "-vf", "scale='min(720,iw)':-2", "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "28", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    for sync in (["-fps_mode", "passthrough"], ["-vsync", "0"]):
        r = subprocess.run(base + sync + [dst], capture_output=True, text=True, timeout=240)
        if r.returncode == 0 and os.path.exists(dst):
            return
    raise UserError("We couldn't read that video. Try recording again, or export it as MP4.")


def parse_boxes(text):
    pieces = FRAME_RE.split(text)
    frames = {}
    for i in range(1, len(pieces) - 1, 2):
        frame = int(pieces[i])
        for m in BOX_RE.finditer(pieces[i + 1]):
            obj = int(m.group(1))
            x1, y1, x2, y2 = (int(m.group(k)) for k in (2, 3, 4, 5))
            frames.setdefault(frame, {})[obj] = (x1, y1, x2, y2)
    return frames


def _client():
    from openai import OpenAI
    return OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"], timeout=300)


def track(mp4_path, prompt):
    """Send the video to SAM 3.1 and return rows for the object seen in the most frames."""
    client = _client()
    with open(mp4_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="user_data")
    frames, buf = {}, ""

    def drain(final=False):
        """Keep only the small boxes; drop the big mask data as soon as a frame is complete."""
        nonlocal buf
        marks = list(FRAME_RE.finditer(buf))
        if not final and len(marks) < 2:
            return
        cut = len(buf) if final else marks[-1].start()
        for fr, objs in parse_boxes(buf[:cut]).items():
            frames.setdefault(fr, {}).update(objs)
        buf = buf[cut:]

    try:
        with client.responses.stream(
            model="sam-3.1",
            input=[{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_video", "file_id": uploaded.id}]}],
            metadata={"mask_encoding": "one_bit"},
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    buf += event.delta
                    if len(buf) > 200_000:
                        drain()
        drain(final=True)
    finally:
        try:
            client.files.delete(uploaded.id)
        except Exception:
            pass
    counts = {}
    for fr in frames.values():
        for o in fr:
            counts[o] = counts.get(o, 0) + 1
    if not counts:
        raise UserError(f"SAM couldn't find a \"{prompt}\" in your video. Try one simple word (like ball or bottle) "
                        "and film against a plain background.")
    best = max(counts, key=counts.get)
    rows = []
    for frame in sorted(frames):
        if best in frames[frame]:
            x1, y1, x2, y2 = frames[frame][best]
            rows.append({"frame": frame, "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
                         "width": x2 - x1, "height": y2 - y1})
    return rows, len(frames)


def mock_rows(mission, fps, size_m, length_m):
    """Fake tracking so the app can be tried without an API key (POCKETLAB_MOCK=1)."""
    rng = np.random.default_rng(7)
    scale = size_m / 100.0
    if mission == "pendulum":
        t = np.arange(0, 14, 1 / fps)
        T = 2 * np.pi * np.sqrt((length_m or 0.43) / 9.55)
        x = 0.05 * np.sin(2 * np.pi * t / T) * np.exp(-t / 30)
        y = np.full_like(t, 0.3)
    elif mission == "drop":
        t = np.arange(0, 1.6, 1 / fps)
        h = 0.9
        fall = (2 * h / 9.2) ** 0.5
        y = np.where(t < 0.5, h, np.maximum(h - 0.5 * 9.2 * (t - 0.5) ** 2, 0.0))
        x = np.zeros_like(t)
        y = np.where(t > 0.5 + fall, 0.0, y)
    else:
        t = np.arange(0, 3, 1 / fps)
        y, vel, pos, dt = [], 0.0, 1.0, 1 / 2000
        k = 0
        for ti in t:
            while k * dt < ti:
                vel -= 9.4 * dt
                pos += vel * dt
                if pos < 0:
                    pos, vel = 0.0, -vel * 0.75
                k += 1
            y.append(pos)
        y, x = np.array(y), np.zeros_like(t)
    rows = []
    for i, (xi, yi) in enumerate(zip(x, y)):
        rows.append({"frame": i, "cx": 500 + xi / scale + rng.normal(0, 0.7),
                     "cy": 900 - yi / scale + rng.normal(0, 0.7), "width": 100.0, "height": 100.0})
    return rows, len(rows)
