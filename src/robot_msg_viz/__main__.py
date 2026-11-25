"""robot-msg-viz: look at any robot_msgs type from the shell.

    robot-msg-viz list                    # every type, supported or not
    robot-msg-viz demo Pose               # the stock view of the sample, in the desktop viewer
    robot-msg-viz demo Pose --web         # the same in the web viewer (opens your browser)
    robot-msg-viz save Pose pose.rrd      # the sample recording, blueprint inside
    robot-msg-viz gallery OUT             # every gallery .rrd (types and installed use cases) + gallery.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_msg_viz import launch
from robot_msg_viz.gallery import GalleryEntry, entries


def find(name: str) -> GalleryEntry:
    """A gallery entry by key (``geometry/Pose``, ``object_detection``) or bare type name (``Pose``)."""
    every = entries()
    for entry in every:
        if name in (entry.key, entry.slug, entry.title) or entry.key.split("/")[-1] == name:
            return entry
    raise SystemExit(f"no gallery entry {name!r}; try: robot-msg-viz list")


def manifest(entry: GalleryEntry, rrd: str | None) -> dict:
    return {
        "kind": entry.kind, "key": entry.key, "slug": entry.slug, "title": entry.title, "group": entry.group,
        "summary": entry.summary, "supported": entry.supported, "code": entry.code, "consumes": list(entry.consumes),
        "source": entry.source, "timed": entry.timed, "tags": list(entry.tags), "rrd": rrd,
        "viewers": [list(v) for v in entry.viewers], "adapters": list(entry.adapters),
    }  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robot-msg-viz", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="every gallery entry, supported or not")
    demo = sub.add_parser("demo", help="open an entry's sample in the viewer")
    demo.add_argument("name")
    demo.add_argument("--web", action="store_true", help="the web viewer instead of the desktop one")
    save = sub.add_parser("save", help="write an entry's sample recording")
    save.add_argument("name")
    save.add_argument("path", type=Path)
    gallery = sub.add_parser("gallery", help="write every supported entry's .rrd and gallery.json")
    gallery.add_argument("out", type=Path)
    args = parser.parse_args(argv)

    if args.command == "list":
        for entry in entries():
            mark = "✓" if entry.supported else "·"
            print(f"{mark} {entry.kind:8} {entry.key:40} {entry.summary}")
        return 0
    if args.command in ("demo", "save"):
        entry = find(args.name)
        if not entry.supported or entry.write is None:
            raise SystemExit(f"{entry.key} has no stock view yet")
        if args.command == "save":
            print(entry.write(args.path))
            return 0
        import tempfile

        path = entry.write(Path(tempfile.mkdtemp(prefix="robot_msg_viz-")) / f"{entry.slug}.rrd")
        launch.open(path, web=args.web)
        print(path)
        return 0
    if args.command == "gallery":
        args.out.mkdir(parents=True, exist_ok=True)
        rows = []
        for entry in entries():
            rrd = None
            if entry.supported and entry.write is not None:
                rrd = f"{entry.slug}.rrd"
                entry.write(args.out / rrd)
            rows.append(manifest(entry, rrd))
        (args.out / "gallery.json").write_text(json.dumps(rows, indent=1))
        print(f"{sum(r['rrd'] is not None for r in rows)} recordings, {len(rows)} entries -> {args.out}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
