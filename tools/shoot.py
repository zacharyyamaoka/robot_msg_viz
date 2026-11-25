"""Write every gallery recording and shoot it in the real Rerun web viewer; report blank or broken stills.

    uv run python tools/shoot.py OUT [--only KEY ...]

Writes OUT/rrd/<slug>.rrd and OUT/shots/<slug>.png for each supported gallery entry (types, and the use cases
of every installed plugin), then OUT/shots.json: per entry the viewer's errors, whether it panicked, and the
still's pixel spread (a still whose spread is under BLANK_STD is a blank view, not a picture).
WHY a pixel spread and not "the page loaded": a view pointed at the wrong entity renders an empty grid that
loads without error.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from robot_msg_viz.gallery import entries

REPO = Path(__file__).resolve().parents[1]
BLANK_STD = 6.0  # grey levels: Rerun's empty dark canvas with its grid measures ~2-4


def pixel_spread(png: Path) -> float:
    from PIL import Image  # matplotlib's dependency, so present wherever robot-msgs[rerun] is

    pixels = np.asarray(Image.open(png).convert("L"), dtype=float)
    return float(pixels.std())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    (out / "rrd").mkdir(parents=True, exist_ok=True)
    (out / "shots").mkdir(parents=True, exist_ok=True)

    jobs = []
    for entry in entries():
        if not entry.supported or entry.write is None:
            continue
        if args.only and entry.key not in args.only and entry.slug not in args.only:
            continue
        rrd = entry.write(out / "rrd" / f"{entry.slug}.rrd")
        jobs.append({"rrd": str(rrd), "out": str(out / "shots" / f"{entry.slug}.png"), "time": "end", "key": entry.key})

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(jobs, handle)
    run = subprocess.run(["node", str(REPO / "tools" / "shoot.cjs"), handle.name], capture_output=True, text=True)
    if run.returncode != 0:
        print(run.stderr, file=sys.stderr)
        return run.returncode
    results = {}
    for line, job in zip((l for l in run.stdout.splitlines() if l.startswith("{")), jobs, strict=False):
        row = json.loads(line)
        png = Path(row["out"])
        row["spread"] = pixel_spread(png) if png.exists() else 0.0
        row["blank"] = row["spread"] < BLANK_STD
        row["errors"] = [e for e in row["errors"] if "favicon" not in e]
        results[job["key"]] = row
        flag = "OK " if row["ok"] and not row["blank"] and not row["errors"] else "BAD"
        print(f"{flag} {job['key']:40} spread={row['spread']:.1f} errors={len(row['errors'])}")
    (out / "shots.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
