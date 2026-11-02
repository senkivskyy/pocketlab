"""Check the physics on simulated videos where we know the true answer."""
import math
import numpy as np
import physics as P

rng = np.random.default_rng(1)


def fake_rows(t, x_m, y_m, scale, size_px, floor_px=900):
    rows = []
    for i, (ti, xi, yi) in enumerate(zip(t, x_m, y_m)):
        rows.append({"frame": i, "cx": 500 + xi / scale + rng.normal(0, 0.7),
                     "cy": floor_px - yi / scale + rng.normal(0, 0.7),
                     "width": size_px, "height": size_px})
    return rows


def bounce(g=9.81, e=0.75, h0=1.0, fps=30, seconds=3.0):
    t = np.arange(0, seconds, 1 / fps)
    y, v, h = [], 0.0, h0
    pos, vel, dt = h0, 0.0, 1 / 2000
    tt = np.arange(0, seconds, dt)
    ys = np.empty_like(tt)
    for i in range(len(tt)):
        vel -= g * dt
        pos += vel * dt
        if pos < 0:
            pos, vel = -pos * e, -vel * e
        ys[i] = pos
    return t, np.interp(t, tt, ys)


for fps in (30, 60):
    t, y = bounce(fps=fps)
    scale = 0.067 / 40  # tennis ball: 6.7 cm = 40 px
    rows = fake_rows(t, np.zeros_like(t), y, scale, 40)
    tt, x, yy, sc = P.to_series(rows, fps, 0.067)
    r = P.analyze_bounce(tt, yy)
    print(f"bounce {fps}fps: g={r.get('g'):.2f} +/- {r.get('g_sd'):.2f}  e={r.get('e'):.2f}  flights={len(r['flights'])}  (true g 9.81, e 0.75)")
    assert abs(r["g"] - 9.81) / 9.81 < 0.10, r
    assert abs(r["e"] - 0.75) < 0.08, r

for L in (0.47, 0.8):
    fps = 60
    t = np.arange(0, 19, 1 / fps)
    T = 2 * math.pi * math.sqrt(L / 9.81)
    amp = L * math.sin(math.radians(12)) * np.exp(-t / 25)          # swing slowly dies down
    x = amp * np.sin(2 * math.pi * t / T) + 0.002 * t              # and the hand drifts
    x = x + rng.normal(0, 0.004, len(t))                            # key jitter
    spikes = rng.random(len(t)) < 0.05
    x[spikes] += rng.normal(0, 0.03, spikes.sum())                  # tracking glitches
    keep = rng.random(len(t)) > 0.13                                # SAM missed some frames
    rows = fake_rows(t[keep], x[keep], np.full(keep.sum(), 0.3), 0.001, 25)
    for r, f in zip(rows, np.flatnonzero(keep)):
        r["frame"] = f
    tt, xx, yy, sc = P.to_series(rows, fps, 0.025)
    r = P.analyze_pendulum(tt, xx, L)
    print(f"pendulum L={L}: g={r['g']:.2f} +/- {r['g_sd']:.2f}  T={r['period']:.3f}s (true {T:.3f})")
    assert abs(r["g"] - 9.81) / 9.81 < 0.05, r

for fps, h0_, soft in ((120, 0.5, True), (120, 1.5, False), (240, 1.5, False)):
    ts = np.arange(0, 2.0, 1 / fps)
    fall = np.sqrt(2 * h0_ / 9.81)
    ys = np.where(ts < 0.5, h0_, np.maximum(h0_ - 0.5 * 9.81 * (ts - 0.5) ** 2, 0.0))
    if soft:  # landing on a bed: the object sinks a little instead of stopping dead
        ys = np.where(ts > 0.5 + fall, -0.02 * (1 - np.exp(-(ts - 0.5 - fall) * 30)), ys)
        ys = ys - ys.min()
    ys = ys + rng.normal(0, 0.003, len(ts))
    rows = fake_rows(ts, np.zeros_like(ts), ys, 0.076 / 475, 475)
    tt, x, yy, sc = P.to_series(rows, fps, 0.076)
    r = P.analyze_drop(tt, yy)
    print(f"drop {fps}fps from {h0_} m: g={r['g']:.2f}  frames={r['frames']}  (true 9.81)")
    assert abs(r["g"] - 9.81) / 9.81 < 0.08, r

print(P.rank(9.7, P.G_TRUE), P.rank(8.0, P.G_TRUE))
print("all physics tests passed")
