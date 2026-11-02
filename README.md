# Pocket Lab

**Turn your phone into a science lab. Film an everyday experiment, let Meta's SAM 3.1 track it, measure real physics, and level up.**

- **Live app:** https://pocketlab-cmr9.onrender.com/
- **Demo video:** YOUR_YOUTUBE_URL_HERE
- **Event:** Meta Global AI Developer Hackathon

## How it works

1. **Predict.** Guess the answer first. Good guesses earn bonus XP.
2. **Film.** Swing a key on a string, drop a bottle, bounce a ball. Any phone video works, including slow motion.
3. **Track.** Meta's **SAM 3.1** follows your object through every frame of the video.
4. **Measure.** Code turns the tracked path into real numbers. The object's own size is the ruler, and the frame rate is the stopwatch. Pocket Lab measures gravity (g) three different ways and compares it to the accepted 9.81 m/s².
5. **Level up.** Earn gold, silver or bronze medals (within 3%, 8%, 15%), XP, ranks and a shareable constant card. Climb the leaderboard, alone or with a crew code for your class or friends.

## Meta models used

| What | Model |
| --- | --- |
| Tracks the object through the video | `sam-3.1` (Meta Model API) |
| Lab coach that explains your result | `muse-spark-1.3`, or Muse Spark Open (`meta-models/muse-spark-open-1026`) through Baseten, set with `COACH_*` variables |
| Card artwork, one image per mission | `muse-image-1.0` |

## The physics

- **Swing Time:** period from the strongest frequency in the swing (FFT), then g = 4π²L / T².
- **Drop Zone:** an object dropped from rest follows y = ½ g (t − t₀)². Fitting the release time and g together keeps short falls steady.
- **Bounce Lab:** each flight between bounces is a parabola, and the ratio of flight times gives bounciness.

`test_physics.py` runs simulated experiments with noise, dropped frames and tracking glitches (`python3 test_physics.py`).

## Run it locally

You need Python 3.10+ and a Meta Model API key.

```bash
git clone <this repo>
cd pocketlab
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # paste your key into .env
uvicorn app:app --reload
```

Open http://127.0.0.1:8000. To try the app without an API key, run `POCKETLAB_MOCK=1 uvicorn app:app --reload`, which fakes the SAM step.

## Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `MODEL_API_KEY` | Meta Model API key | required |
| `PER_IP_DAILY` / `GLOBAL_DAILY` | Free runs per visitor and for everyone, per day | 10 / 200 |
| `MAX_FRAMES` | Longest clip allowed, in frames (caps SAM cost) | 2000 |
| `COACH_BASE_URL`, `COACH_MODEL`, `COACH_API_KEY` | Run the coach on open weights | Meta API, `muse-spark-1.3` |
| `DATA_DIR` | Where the leaderboard database lives | `./data` |

## Project layout

```
app.py             FastAPI server, upload jobs, usage limits, coach and card art
tracker.py         Video prep (ffmpeg) and SAM 3.1 tracking
physics.py         Turns tracked boxes into g, swing time, bounciness
game.py            Missions, medals, XP, ranks, leaderboard (SQLite)
static/index.html  Mobile-first web app
```

## Limitations

- Drop and bounce are beta: they are less accurate than the pendulum and need a clean, side-on clip.
- Results depend on measuring the object carefully and entering the true frame rate.
- On the free host the leaderboard database resets when the server restarts.

## License

MIT. See [LICENSE](LICENSE).
