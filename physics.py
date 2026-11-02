"""Pocket Lab physics: turn a tracked path (pixels per frame) into real measurements.

Input is a list of rows from track.csv: frame, cx, cy, width, height (all in pixels).
Image y grows downward, so height above the floor is (floor_y - cy).
"""
import math

import numpy as np

G_TRUE = 9.80665


def to_series(rows, fps, object_size_m, ruler="min"):
    """Rows -> time (s), x (m), height-up y (m), using the object's own size as the ruler.

    ruler="min"    : the shorter side of the box (best for swings, balls)
    ruler="height" : the up-down side of the box (best for things that fall)
    """
    rows = sorted(rows, key=lambda r: r["frame"])
    frames = np.array([r["frame"] for r in rows], float)
    cx = np.array([r["cx"] for r in rows], float)
    cy = np.array([r["cy"] for r in rows], float)
    if ruler == "height":
        size_px = np.median([r["height"] for r in rows])
    else:
        size_px = np.median([min(r["width"], r["height"]) for r in rows])
    scale = object_size_m / size_px  # metres per pixel
    t = (frames - frames[0]) / fps
    return t, (cx - cx[0]) * scale, (cy.max() - cy) * scale, scale


def _local_peaks(values, k=2, min_prominence=0.0):
    """Indices that are the largest value within +/- k samples."""
    out = []
    for i in range(k, len(values) - k):
        window = values[i - k: i + k + 1]
        if values[i] == window.max() and values[i] - window.min() >= min_prominence:
            out.append(i)
    return out


def analyze_bounce(t, y):
    """A ball bouncing: gravity from each flight, and bounciness from flight times.

    Returns dict with g_estimates (one per flight), g (mean), g_sd, e (restitution), flights.
    """
    # Floor impacts are the lowest points. Look for minima of height.
    lows = _local_peaks(-y, k=2, min_prominence=0.05 * (y.max() - y.min()))
    flights = []
    for a, b in zip(lows, lows[1:]):
        if b - a < 6:  # too short to fit a parabola
            continue
        trim = max(1, (b - a) // 8)  # ignore frames touching the floor
        ts, ys = t[a + trim: b - trim + 1], y[a + trim: b - trim + 1]
        if len(ts) < 5:
            continue
        c2, c1, c0 = np.polyfit(ts - ts[0], ys, 2)
        if c2 >= 0:
            continue  # not a downward parabola, so not a flight
        flights.append({"t0": float(t[a]), "t1": float(t[b]), "duration": float(t[b] - t[a]), "g": float(-2 * c2)})
    gs = [f["g"] for f in flights]
    result = {"flights": flights, "g_estimates": gs}
    if gs:
        result["g"] = float(np.mean(gs))
        result["g_sd"] = float(np.std(gs))
    if len(flights) >= 2:
        ratios = [b["duration"] / a["duration"] for a, b in zip(flights, flights[1:])]
        result["e"] = float(np.mean(ratios))  # flight time ratio equals the restitution
    return result


def analyze_drop(t, y):
    """An object dropped from rest: y(t) = y_top - 1/2 g (t - t0)^2.

    Because it starts at rest we only have two unknowns, the release time t0 and g,
    which is much steadier than fitting a free parabola to a short fall.
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    span = y.max() - y.min()
    at_top = np.where(y >= y.max() - 0.01 * span)[0]
    top = int(at_top[-1])
    y_top = float(np.median(y[at_top]))
    below = np.where(y[top:] <= y.min() + 0.04 * span)[0]
    if len(below) == 0:
        return {"error": "Could not find the object landing. Keep the whole fall in the video."}
    bottom = top + int(below[0])
    if bottom - top < 8:
        return {"error": "Too few frames of the fall. Film in Slo-mo (120 or 240 fps) or drop from higher."}
    # Use the fall itself, but not the last frames where the landing slows the object.
    end = bottom - max(1, (bottom - top) // 10)
    ts, ds = t[top - 3 if top >= 3 else 0: end + 1], (y_top - y)[top - 3 if top >= 3 else 0: end + 1]
    best = None
    dt = float(np.median(np.diff(t)))
    for t0 in np.arange(t[top] - 0.12, t[top] + 0.12, dt / 4):
        tau = np.clip(ts - t0, 0, None)
        denom = float(np.sum(tau ** 4))
        if denom == 0:
            continue
        g = 2 * float(np.sum(ds * tau ** 2)) / denom
        resid = float(np.sum((ds - 0.5 * g * tau ** 2) ** 2))
        if best is None or resid < best[0]:
            best = (resid, g, t0)
    if best is None or best[1] <= 0:
        return {"error": "The path did not look like a fall. Film side-on."}
    resid, g, t0 = best
    rms = math.sqrt(resid / len(ts))
    return {"g": float(g), "g_sd": float(rms / max(span, 1e-9) * g), "frames": int(len(ts)),
            "fall_height": float(y[top] - y[bottom]), "fall_time": float(t[bottom] - t0),
            "t0": float(t0), "y_top": float(y_top)}


def _dominant_period(t, x, fmin=0.2, fmax=5.0):
    """Strongest repeating period in x(t), from the FFT peak (robust to jitter and gaps)."""
    dt = float(np.median(np.diff(t)))
    grid = np.arange(t[0], t[-1], dt)
    xs = np.interp(grid, t, x)  # fill dropped frames
    # Remove slow drift (a moving mean about 3 s wide) so only the swing is left.
    win = max(3, int(3.0 / dt))
    kernel = np.ones(win) / win
    xs = xs - np.convolve(np.pad(xs, win // 2, mode="edge"), kernel, mode="same")[win // 2: win // 2 + len(xs)]
    xs = xs * np.hanning(len(xs))
    n = 1 << int(np.ceil(np.log2(len(xs) * 16)))
    spec = np.abs(np.fft.rfft(xs, n))
    freqs = np.fft.rfftfreq(n, dt)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not band.any() or spec[band].max() == 0:
        return None
    i = int(np.argmax(np.where(band, spec, 0)))
    if 0 < i < len(spec) - 1:  # sharpen the peak between FFT bins
        a, b, c = np.log(spec[i - 1] + 1e-12), np.log(spec[i] + 1e-12), np.log(spec[i + 1] + 1e-12)
        shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0
        f = freqs[i] + shift * (freqs[1] - freqs[0])
    else:
        f = freqs[i]
    return 1.0 / f


def analyze_pendulum(t, x, length_m):
    """A swinging pendulum: period from the dominant frequency, then g = 4 pi^2 L / T^2."""
    t, x = np.asarray(t, float), np.asarray(x, float)
    if t[-1] - t[0] < 4.0:
        return {"error": "Need a longer video: at least a few swings."}
    T = _dominant_period(t, x)
    if T is None:
        return {"error": "Could not find a swinging motion."}
    # Check consistency: measure the period again on three slices of the video.
    thirds = np.array_split(np.arange(len(t)), 3)
    periods = []
    for idx in thirds:
        if len(idx) > 8 and t[idx[-1]] - t[idx[0]] >= 3 * T:
            p = _dominant_period(t[idx], x[idx])
            if p:
                periods.append(p)
    g_each = [4 * math.pi ** 2 * length_m / (p ** 2) for p in periods]
    return {
        "period": float(T),
        "period_sd": float(np.std(periods)) if len(periods) > 1 else 0.0,
        "g": float(4 * math.pi ** 2 * length_m / T ** 2),
        "g_sd": float(np.std(g_each)) if len(g_each) > 1 else 0.0,
        "swings": int((t[-1] - t[0]) / T),
    }


def rank(measured, true, bands=((0.03, "gold"), (0.08, "silver"), (0.15, "bronze"))):
    """How close was the measurement? Returns (percent error, medal or None)."""
    err = abs(measured - true) / true
    for limit, medal in bands:
        if err <= limit:
            return err * 100, medal
    return err * 100, None
