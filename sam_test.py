"""Pocket Lab, step 1: send a short video to SAM 3.1 and see what comes back.

Usage:  python3 sam_test.py clip.mov "ball"
The second argument is ONE short noun phrase naming the thing to track.
"""
import csv
import os
import re
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"])

FRAME_RE = re.compile(r"<(\d+)f>")
BOX_RE = re.compile(r"(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)\|>")


def parse_boxes(text):
    """Turn SAM's text stream into {frame: {object_id: (x1, y1, x2, y2)}}."""
    pieces = FRAME_RE.split(text)  # ['', '0', chunk, '1', chunk, ...]
    frames = {}
    for i in range(1, len(pieces) - 1, 2):
        frame = int(pieces[i])
        for m in BOX_RE.finditer(pieces[i + 1]):
            obj = int(m.group(1))
            x1, y1, x2, y2 = (int(m.group(k)) for k in (2, 3, 4, 5))
            frames.setdefault(frame, {})[obj] = (x1, y1, x2, y2)
    return frames


def track_video(path, prompt):
    with open(path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="user_data")
    print("Uploaded. Tracking...")
    text = ""
    with client.responses.stream(
        model="sam-3.1",
        input=[{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_video", "file_id": uploaded.id},
            ],
        }],
        metadata={"mask_encoding": "one_bit"},  # smaller output
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                text += event.delta
    return text


def main():
    if len(sys.argv) < 3:
        sys.exit('Usage: python3 sam_test.py clip.mov "ball"')
    raw = track_video(sys.argv[1], sys.argv[2])
    with open("sam_raw.txt", "w") as f:
        f.write(raw)

    frames = parse_boxes(raw)
    if not frames:
        print("No objects found. Look at sam_raw.txt, or try a different noun (e.g. 'red ball').")
        return
    objects = sorted({o for f in frames.values() for o in f})
    print(f"Frames with detections: {len(frames)}  |  objects: {objects}")
    print(f"Rough cost: ${len(frames) / 1000 * 0.20:.4f} (at $0.20 per 1,000 frames)")

    with open("track.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "object", "cx", "cy", "width", "height"])
        for frame in sorted(frames):
            for obj, (x1, y1, x2, y2) in frames[frame].items():
                w.writerow([frame, obj, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
    first = sorted(frames)[:5]
    for fr in first:
        print(" frame", fr, frames[fr])
    print("Saved track.csv (the path of each object) and sam_raw.txt (SAM's raw answer).")


if __name__ == "__main__":
    main()
