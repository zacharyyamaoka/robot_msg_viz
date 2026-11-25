"""python -m robot_msg_viz.site build OUT [--shots DIR] | serve OUT [--port 4493]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m robot_msg_viz.site", description="Build or serve the gallery site.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="write the site into OUT")
    build.add_argument("out", type=Path)
    build.add_argument(
        "--shots", type=Path, default=None,
        help="a folder of <slug>.png stills (tools/shoot.py writes OUT/shots/); without it every entry shows a placeholder",
    )  # fmt: skip
    serve = sub.add_parser("serve", help="serve a built site with CORS and a local Rerun web viewer")
    serve.add_argument("out", type=Path)
    serve.add_argument("--port", type=int, default=4493, help="the site's port; the web viewer takes the next one")
    args = parser.parse_args(argv)

    if args.command == "build":
        from robot_msg_viz.site.build import build as run_build

        stats = run_build(args.out, shots=args.shots)
        print(
            f"{stats['types']} types ({stats['supported']} with a view), {stats['use_cases']} use cases, "
            f"{stats['screenshots']} screenshots -> {stats['out']}/index.html"
        )
        return 0
    from robot_msg_viz.site.serve import serve as run_serve

    return run_serve(args.out, port=args.port)


if __name__ == "__main__":
    sys.exit(main())
