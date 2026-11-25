"""The landing page's hero video, recorded in the real Rerun 0.40 web viewer from the gallery's own recordings.

    uv run python tools/hero_video.py website/media/hero.mp4 --plan website/hero.json

Writes each segment's gallery recording, records them in headless Chrome (tools/hero.cjs), cuts the viewer's loading
out of the recording with ffmpeg, and writes OUT (H.264, muted, yuv420p, faststart: plays inline everywhere) and
OUT with .webp (the poster: a frame of the first segment, the fallback where video cannot play).

The plan (JSON): {"segments": [{"key": "trajectory/JointTrajectory", "seconds": 8, "play": [0, 5.4],
"clicks": [[at_s, x, y]], "drags": [[at_s, x0, y0, x1, y1, ms]]}], "poster": [segment, at_s], "crf": 30}. Coordinates are
in the 1280x800 viewer. A key is any gallery entry's key or slug (a plugin's use case too, when it is installed).

WHY recorded, never animated: the hero shows what the viewer really does with the recording a visitor can open.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# WHY the recordings are written by a child process: this script then holds no Rerun / numpy while Chrome records, and
# on a loaded machine that difference decided whether the renderer was OOM-killed (2026-11-25)
WRITE = """
import json, sys
from pathlib import Path
from robot_msg_viz.gallery import entries
by_key = {e.key: e for e in entries()} | {e.slug: e for e in entries()}
plan, scratch = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
paths = [str(by_key[seg["key"]].write(scratch / f"{k}-{by_key[seg['key']].slug}.rrd"))
         for k, seg in enumerate(plan["segments"])]
print(json.dumps(paths))
"""


def ffmpeg(*args: str) -> None:
    """ffmpeg, retried: on a loaded machine the OOM killer took a 330 MB encode now and then (2026-11-25)."""
    for attempt in range(4):
        run = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], capture_output=True, text=True)
        if run.returncode == 0:
            return
        print(f"ffmpeg attempt {attempt + 1} failed ({run.returncode}): {run.stderr.strip()[:200]}", file=sys.stderr)
    raise RuntimeError(f"ffmpeg failed 4 times: {' '.join(args[:6])} …")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path, help="the .mp4 to write (its poster: the same name with .webp)")
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        written = subprocess.run([sys.executable, "-c", WRITE, str(args.plan.resolve()), str(scratch)],
                                 capture_output=True, text=True, check=True)  # fmt: skip
        rrds = json.loads(written.stdout.strip().splitlines()[-1])
        segments = [{**seg, "rrd": rrd} for seg, rrd in zip(plan["segments"], rrds, strict=True)]
        # WHY one browser per segment (2026-11-25): with the four recordings in one page the renderer was OOM-killed
        # on a loaded machine; a fresh browser per segment frees each viewer before the next one loads
        parts, marks_all = [], []
        for i, seg in enumerate(segments):
            job = scratch / f"plan{i}.json"
            job.write_text(json.dumps({"video_dir": str(scratch / f"video{i}"), "width": 1280, "height": 800,
                                       "segments": [seg]}))  # fmt: skip
            for attempt in range(3):  # a renderer killed for memory is retried, not a reason to give up
                run = subprocess.run(["node", str(REPO / "tools" / "hero.cjs"), str(job)], capture_output=True,
                                     text=True)  # fmt: skip
                if run.returncode == 0:
                    break
                print(f"{seg['key']}: attempt {attempt + 1} failed: {run.stderr.strip().splitlines()[:1]}", file=sys.stderr)
            else:
                print(run.stderr, file=sys.stderr)
                return run.returncode
            result = json.loads(run.stdout.strip().splitlines()[-1])
            marks = result["segments"][0]
            marks_all.append(marks)
            flag = "errors: " + "; ".join(marks["errors"]) if marks["errors"] else "ok"
            print(f"{seg['key']:32} {marks['start']:6.2f}-{marks['end']:6.2f} s  {flag}")
            part = scratch / f"part{i}.mp4"
            ffmpeg(
                "-i", result["video"], "-ss", f"{marks['start']:.3f}",
                "-to", f"{marks['end']:.3f}", "-vf", "fps=25,format=yuv420p", "-c:v", "libx264", "-preset", "slow",
                # WHY two threads: x264's default (one per core, 24 here) multiplied its memory, and a loaded
                # machine's OOM killer took the encode (2026-11-25)
                "-threads", "2", "-crf", str(plan.get("crf", 30)), "-an", str(part),
            )  # fmt: skip
            parts.append(part)
        listing = scratch / "parts.txt"
        listing.write_text("".join(f"file '{part}'\n" for part in parts))
        ffmpeg("-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(out))
        segment, at = plan.get("poster", [0, 1.0])
        offset = sum(m["end"] - m["start"] for m in marks_all[:segment]) + at
        ffmpeg("-ss", f"{offset:.3f}", "-i", str(out), "-frames:v", "1", "-c:v", "libwebp", "-quality", "82",
               str(out.with_suffix(".webp")))
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,size", "-of", "json", str(out)],
                           capture_output=True, text=True, check=True)  # fmt: skip
    fmt = json.loads(probe.stdout)["format"]
    print(f"{out}: {float(fmt['duration']):.1f} s, {int(fmt['size']) / 1e6:.2f} MB; poster {out.with_suffix('.webp')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
