"""Write every gallery recording and shoot it in the real Rerun web viewer; report blank or broken stills.

    uv run python tools/shoot.py OUT [--only KEY ...]

Writes OUT/rrd/<slug>.rrd and OUT/shots/<slug>.png for each supported gallery entry (types, and the use cases
of every installed plugin), then OUT/shots.json: per entry the viewer's errors, whether it panicked, and how much
of the view the data draws (``drawn``; under DRAWN_MIN the still is ``blank``).

How ``drawn`` is measured: the same recording is also shot with every data entity filtered out
(``rerun rrd filter --drop-entity``, so the blueprint, the viewer and the theme are identical), and ``drawn`` is
the share of view pixels (menu bar and time panel cropped) whose colour the data changed.
WHY not the still's own pixel spread (the first ruler, 2026-11-24): in the dark theme an empty 3D grid (8.3) and an
empty plot (10.8) out-scored real sparse views such as a Pose (9.6) or a metrics table, so spread could not tell a
blank view from a picture. A blank view and its data-less twin are the same image whatever the theme or chrome.
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
from robot_msg_viz.ruler import read, rerun_cli

REPO = Path(__file__).resolve().parents[1]
# Share of view pixels the data must change. Calibrated 2026-11-24: two shots of one recording differ by 0.0 %
# (the viewer renders deterministically), the sparsest real view (a lone Pose) changes 0.22 %, so 0.05 % (~460 px).
DRAWN_MIN = 0.0005
CROP_TOP, CROP_BOTTOM = 40, 40  # px: Rerun's menu bar and the collapsed time panel, identical in both stills
CHANGED = 48  # summed |ΔR|+|ΔG|+|ΔB| above which a pixel counts as changed (below: antialiasing noise)


def pixel_spread(png: Path) -> float:
    """Kept as a diagnostic only (see the module docstring for why it is not the gate)."""
    from PIL import Image  # matplotlib's dependency, so present wherever robot-msgs[rerun] is

    return float(np.asarray(Image.open(png).convert("L"), dtype=float).std())


def drawn_share(png: Path, blank_png: Path) -> float:
    from PIL import Image

    a = np.asarray(Image.open(png).convert("RGB"), dtype=int)[CROP_TOP:-CROP_BOTTOM]
    b = np.asarray(Image.open(blank_png).convert("RGB"), dtype=int)[CROP_TOP:-CROP_BOTTOM]
    if a.shape != b.shape:
        return 0.0
    return float((np.abs(a - b).sum(axis=2) > CHANGED).mean())


def data_less_copy(rrd: Path, out: Path) -> Path:
    """The recording with every data entity dropped: the same blueprint over nothing."""
    paths = sorted(read(rrd).data_paths)
    args = [*rerun_cli(), "rrd", "filter", str(rrd), "-o", str(out)]
    for path in paths:
        args += ["--drop-entity", path]
    subprocess.run(args, check=True, capture_output=True, timeout=120)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    for sub in ("rrd", "rrd/blank", "shots", "shots/blank"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    jobs, keys = [], []
    for entry in entries():
        if not entry.supported or entry.write is None:
            continue
        if args.only and entry.key not in args.only and entry.slug not in args.only:
            continue
        rrd = entry.write(out / "rrd" / f"{entry.slug}.rrd")
        blank = data_less_copy(rrd, out / "rrd" / "blank" / f"{entry.slug}.rrd")
        keys.append(entry.key)
        jobs.append({"rrd": str(rrd), "out": str(out / "shots" / f"{entry.slug}.png"), "time": "end"})
        jobs.append({"rrd": str(blank), "out": str(out / "shots" / "blank" / f"{entry.slug}.png"), "time": "end"})

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(jobs, handle)
    run = subprocess.run(["node", str(REPO / "tools" / "shoot.cjs"), handle.name], capture_output=True, text=True)
    if run.returncode != 0:
        print(run.stderr, file=sys.stderr)
        return run.returncode
    rows = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
    results = {}
    for i, key in enumerate(keys):
        row, twin = (rows[2 * i], rows[2 * i + 1]) if 2 * i + 1 < len(rows) else ({}, {})
        png, blank_png = Path(row.get("out", "")), Path(twin.get("out", ""))
        row.setdefault("ok", False)
        row.setdefault("errors", ["no still"])
        row["spread"] = pixel_spread(png) if png.is_file() else 0.0
        row["drawn"] = drawn_share(png, blank_png) if png.is_file() and blank_png.is_file() else 0.0
        row["blank"] = row["drawn"] < DRAWN_MIN
        row["blank_still"] = str(blank_png) if blank_png.is_file() else None
        results[key] = row
        flag = "OK " if row["ok"] and not row["blank"] and not row["errors"] else "BAD"
        print(f"{flag} {key:40} drawn={100 * row['drawn']:.2f}% errors={len(row['errors'])}")
    (out / "shots.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
