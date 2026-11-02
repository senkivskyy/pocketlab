"""Turn track.csv (from sam_test.py) into a measurement.

Drop:      python3 analyze.py track.csv drop     --fps 240 --size 0.08
Bounce:    python3 analyze.py track.csv bounce   --fps 30 --size 0.067
Pendulum:  python3 analyze.py track.csv pendulum --fps 30 --size 0.03 --length 0.5

--fps     frames per second of the video (iPhone is usually 30; slow motion is 120 or 240)
--size    real size of the tracked object in metres (tennis ball = 0.067)
--length  pendulum only: string length in metres, from the pivot to the middle of the weight
"""
import argparse
import csv

import physics as P


def load(path, obj=0):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if int(r["object"]) == obj:
                rows.append({k: float(r[k]) for k in ("frame", "cx", "cy", "width", "height")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("mission", choices=["bounce", "drop", "pendulum"])
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--size", type=float, required=True)
    ap.add_argument("--length", type=float)
    a = ap.parse_args()

    rows = load(a.csv)
    if len(rows) < 10:
        raise SystemExit("Not enough tracked frames. Try a clearer video.")
    t, x, y, scale = P.to_series(rows, a.fps, a.size)
    print(f"{len(rows)} frames, {t[-1]:.1f} s, scale {scale * 1000:.2f} mm per pixel")

    if a.mission == "bounce":
        r = P.analyze_bounce(t, y)
        if "g" not in r:
            raise SystemExit("Could not find a clean bounce. Film side-on with the whole bounce in view.")
        err, medal = P.rank(r["g"], P.G_TRUE)
        print(f"Flights found: {len(r['flights'])}")
        for i, g in enumerate(r["g_estimates"], 1):
            print(f"  trial {i}: g = {g:.2f} m/s^2")
        print(f"Gravity: {r['g']:.2f} +/- {r['g_sd']:.2f} m/s^2  (accepted 9.81)  error {err:.1f}%  medal: {medal}")
        if "e" in r:
            print(f"Bounciness (restitution): {r['e']:.2f}")
    elif a.mission == "drop":
        r = P.analyze_drop(t, y)
        if "error" in r:
            raise SystemExit(r["error"])
        err, medal = P.rank(r["g"], P.G_TRUE)
        print(f"Fall: {r['fall_height']:.2f} m in {r['fall_time']:.2f} s, {r['frames']} frames used")
        print(f"Gravity: {r['g']:.2f} m/s^2  (accepted 9.81)  error {err:.1f}%  medal: {medal}")
    else:
        if not a.length:
            raise SystemExit("Pendulum needs --length (metres).")
        r = P.analyze_pendulum(t, x, a.length)
        if "error" in r:
            raise SystemExit(r["error"])
        err, medal = P.rank(r["g"], P.G_TRUE)
        print(f"Period: {r['period']:.3f} +/- {r['period_sd']:.3f} s over {r['swings']} swings")
        print(f"Gravity: {r['g']:.2f} +/- {r['g_sd']:.2f} m/s^2  (accepted 9.81)  error {err:.1f}%  medal: {medal}")


if __name__ == "__main__":
    main()
