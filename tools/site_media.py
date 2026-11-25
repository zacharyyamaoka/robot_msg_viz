"""Turn the shooter's stills into a site's committed media: ``<slug>.webp`` and ``<slug>-480.webp`` per entry.

    uv run python tools/site_media.py SHOTS_DIR website/media/shots --kind type
    uv run python tools/site_media.py SHOTS_DIR /home/bam/robot_dashboards/website/media/shots --kind use_case

Converts ``SHOTS_DIR/<slug>.png`` (tools/shoot.py's output) for every supported gallery entry of that kind, with the
builder's own `convert` (full size + a 480 px card without Rerun's menu bar), and removes stills of entries that are
not of that kind. WHY the kind: robot_msg_viz's repo is public and a plugin may be private, so its stills never land
here (tests/test_site_public.py checks it), and a plugin's site gets only its own.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from robot_msg_viz.gallery import entries
from robot_msg_viz.site.build import convert, find_png


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("shots", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--kind", choices=("type", "use_case"), required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    wanted = [e for e in entries() if e.kind == args.kind and e.supported and e.write is not None]
    missing = []
    with tempfile.TemporaryDirectory() as scratch:
        for entry in wanted:
            png = find_png(args.shots, entry.slug)
            if png is None or png.suffix != ".png":
                missing.append(entry.slug)
                continue
            shot, thumb, _ = convert(png, Path(scratch), entry.slug)
            for name in (shot, thumb):
                (out / Path(name).name).write_bytes((Path(scratch) / name).read_bytes())
    keep = {f"{e.slug}.webp" for e in wanted} | {f"{e.slug}-480.webp" for e in wanted}
    for stray in sorted(p for p in out.iterdir() if p.name not in keep):
        stray.unlink()
        print(f"removed {stray.name} (not a supported {args.kind})")
    total = sum(p.stat().st_size for p in out.iterdir())
    print(f"{len(wanted) - len(missing)} of {len(wanted)} {args.kind} stills -> {out} ({total / 2**20:.1f} MiB)")
    if missing:
        print(f"no still for: {' '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
