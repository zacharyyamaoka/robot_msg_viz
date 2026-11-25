"""python -m robot_msg_viz.site build OUT [options] | serve OUT [--port 4493]

    build OUT --shots DIR --hero media/hero.mp4 --no-plugins               robot_msg_viz's public page (types only)
    build OUT --use-cases-only --title robot_dashboards --intro intro.html --viewer --private ...   a plugin's page
"""

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
        help="a folder of <slug>.png stills (tools/shoot.py writes OUT/shots/) or of <slug>.webp + <slug>-480.webp; "
        "without it every entry shows a placeholder",
    )  # fmt: skip
    build.add_argument("--hero", type=Path, default=None,
                       help="the index's muted, looping .mp4 (its poster: the same name with .webp, beside it)")
    which = build.add_mutually_exclusive_group()
    which.add_argument("--no-plugins", action="store_true",
                       help="robot_msg_viz's types only: nothing an installed plugin registers reaches this build")
    which.add_argument("--use-cases-only", action="store_true",
                       help="a plugin's own page: its use cases, no Types tab (type chips link to --types-url)")
    build.add_argument("--title", default=None, help="the brand and page titles (default robot_msg_viz)")
    build.add_argument("--tagline", default=None, help="the index's headline after the title")
    build.add_argument("--intro", type=Path, default=None, help="an HTML fragment: the index's opening paragraph(s)")
    build.add_argument("--github", default=None, help="the repo the header links")
    build.add_argument("--types-url", default=None,
                       help="a site with the type pages, for the type chips (default with --use-cases-only: "
                       "robot_msg_viz's public page)")  # fmt: skip
    build.add_argument("--viewer", action="store_true",
                       help="serve Rerun's web viewer from this site (OUT/viewer/, from the installed rerun-sdk): for "
                       "a host the hosted viewer cannot fetch from, such as a page behind a sign-in")  # fmt: skip
    build.add_argument("--private", action="store_true",
                       help="the page is behind a sign-in: Open on desktop opens the downloaded .rrd, not its URL")
    serve = sub.add_parser("serve", help="serve a built site with CORS and a local Rerun web viewer")
    serve.add_argument("out", type=Path)
    serve.add_argument("--port", type=int, default=4493, help="the site's port; the web viewer takes the next one")
    args = parser.parse_args(argv)

    if args.command == "build":
        import dataclasses

        from robot_msg_viz.gallery import entries, type_entries, use_case_entries
        from robot_msg_viz.site.build import PUBLIC_SITE, Site
        from robot_msg_viz.site.build import build as run_build

        gallery = type_entries() if args.no_plugins else use_case_entries() if args.use_cases_only else entries()
        site = Site(use_cases_only=args.use_cases_only, private=args.private,
                    types_url=args.types_url or (PUBLIC_SITE if args.use_cases_only else None))  # fmt: skip
        changes = {"title": args.title, "tagline": args.tagline, "github": args.github,
                   "lead_html": args.intro.read_text(encoding="utf-8").strip() if args.intro else None}  # fmt: skip
        site = dataclasses.replace(site, **{k: v for k, v in changes.items() if v is not None})
        if args.use_cases_only:
            site = dataclasses.replace(site, code="")  # the plugin's code is on each use case's page
        stats = run_build(args.out, shots=args.shots, gallery=gallery, site=site, hero=args.hero, viewer=args.viewer)
        viewer = stats["viewer"]
        print(
            f"{stats['types']} types ({stats['supported']} with a view), {stats['use_cases']} use cases, "
            f"{stats['screenshots']} screenshots, hero {stats['hero'] or 'none'}, "
            f"viewer {'self-hosted, largest file %.1f MiB' % (viewer['largest'] / 2**20) if viewer else 'hosted'}"
            f" -> {stats['out']}/index.html"
        )
        return 0
    from robot_msg_viz.site.serve import serve as run_serve

    return run_serve(args.out, port=args.port)


if __name__ == "__main__":
    sys.exit(main())
