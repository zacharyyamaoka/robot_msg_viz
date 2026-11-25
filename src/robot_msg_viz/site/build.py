"""Build the gallery site into a folder: pure functions from gallery entries to HTML strings.

    python -m robot_msg_viz.site build OUT [--shots DIR] [--hero VIDEO] [--no-plugins | --use-cases-only] [--viewer]

Writes ``OUT/index.html`` (Types and Use cases tabs), ``OUT/types/<slug>.html`` and ``OUT/use-cases/<slug>.html``
(one per entry that has a view), ``OUT/rrd/<slug>.rrd`` (the entry's sample recording, blueprint inside),
``OUT/shots/<slug>.webp`` and ``<slug>-480.webp`` (from ``DIR/<slug>.png`` when the shooter made one, or copied from
``DIR/<slug>.webp`` + ``<slug>-480.webp``), ``OUT/media/`` (the index's hero video and its poster) and
``OUT/gallery.json``. Every link is relative, so OUT works at any URL prefix (a GitHub Pages project path too).

One builder for every gallery site (`Site`): robot_msg_viz's public page (``--no-plugins``: types only, so nothing a
plugin ships reaches it) and a plugin's own page (``--use-cases-only --title robot_dashboards ...``), which can also
carry Rerun's web viewer itself (``--viewer``, `robot_msg_viz.site.viewer`) for a host the hosted viewer cannot read.

WHY stdlib string building, no template engine or npm: robotmsgs.com's builder works this way and this site
shares its look (robot_msgs/website/build/pages.py, site/site.css); one less toolchain to install.
"""

from __future__ import annotations

import dataclasses
import hashlib
import html
import json
import keyword
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from robot_msg_viz import launch
from robot_msg_viz._engine import RERUN_SDK_VERSION
from robot_msg_viz.gallery import GalleryEntry
from robot_msg_viz.gallery import entries as gallery_entries

GITHUB = "https://github.com/zacharyyamaoka/robot_msg_viz"
PUBLIC_SITE = "https://zacharyyamaoka.github.io/robot_msg_viz/"
ROBOT_MSGS_DOCS = "https://robotmsgs.com/types/{domain}/{name}/"
STATIC = Path(__file__).resolve().parent / "static"
THUMB_WIDTH = 480
RERUN_TOP_BAR = 36 / 800  # the web viewer's menu bar: 36 px of the shooter's 1280x800 still (measured 2026-11-24)
# WHY a fixed list: build() clears only what it writes, never the whole OUT (OUT may be a folder you care about).
GENERATED = ("index.html", "types", "use-cases", "rrd", "shots", "static", "gallery.json", "viewer.json", "media",
             "viewer")  # fmt: skip


def esc(text: object) -> str:
    return html.escape(str(text))


# ------------------------------------------------------------------------------------------- model

HERO_CODE = """import robot_msg_viz as viz

viz.show(pose)               # open its stock view
viz.save("pose.rrd", pose)   # a file, layout inside
with viz.stream("arm") as live:
    live.log(joint_state)    # live, as they arrive"""

LEAD = (
    'Hand it a <a href="https://robotmsgs.com">robot_msgs</a> message and Rerun opens already laid out for it: the right '
    "views, colours and camera, saved as a blueprint. Every card below is a real recording: open it in your browser, or "
    "on your desktop with one command."
)


@dataclass(frozen=True)
class Site:
    """Who the site is for: robot_msg_viz's own page by default, or a plugin's (``use_cases_only``).

    - ``title``: the brand and the page titles; ``tagline``: the index's headline after it.
    - ``lead_html``: the index's opening paragraph(s), trusted HTML (a plugin's page passes its own: the layers diagram).
    - ``github``: the repo the header links; ``code``: the index's code block (empty: none).
    - ``use_cases_only``: one Use cases list, no Types tab (a plugin's page).
    - ``types_url``: where type pages live when this site has none (chips link there): robot_msg_viz's public page.
    - ``private``: the page is behind a sign-in, so a desktop viewer cannot fetch a recording by its URL; Open on
      desktop opens the downloaded file instead.
    - ``hero`` / ``hero_poster``: paths in OUT of the index's muted, looping video and its still.
    """

    title: str = "robot_msg_viz"
    tagline: str = "a ready-made Rerun view for every robot message"
    lead_html: str = LEAD
    github: str = GITHUB
    code: str = HERO_CODE
    use_cases_only: bool = False
    types_url: str | None = None
    private: bool = False
    hero: str | None = None
    hero_poster: str | None = None


@dataclass(frozen=True)
class Item:
    """One gallery entry and the files the build made for it (paths relative to OUT)."""

    entry: GalleryEntry
    page: str | None = None  # "types/geometry-Pose.html": only entries with a view get a page
    rrd: str | None = None
    rrd_bytes: int = 0
    shot: str | None = None
    thumb: str | None = None
    shot_size: tuple[int, int] = (1280, 800)

    @property
    def live(self) -> bool:
        return self.page is not None


def section_of(entry: GalleryEntry) -> str:
    return "types" if entry.kind == "type" else "use-cases"


def find_png(shots: Path | None, slug: str) -> Path | None:
    """The entry's still: ``DIR/<slug>.png`` (or ``DIR/shots/<slug>.png``, the shooter's own output folder), else a
    ready ``DIR/<slug>.webp`` (committed site media)."""
    if shots is None:
        return None
    for name in (f"{slug}.png", f"{slug}.webp"):
        for candidate in (shots / name, shots / "shots" / name):
            if candidate.is_file():
                return candidate
    return None


def convert(png: Path, out: Path, slug: str) -> tuple[str, str, tuple[int, int]]:
    """The still as WebP at full size and at THUMB_WIDTH; returns (shot, thumb, size). A WebP with its
    ``<slug>-480.webp`` beside it is copied as it is (no second lossy pass)."""
    from PIL import Image  # robot-msgs[rerun] brings matplotlib, which brings Pillow

    (out / "shots").mkdir(parents=True, exist_ok=True)
    shot, thumb = f"shots/{slug}.webp", f"shots/{slug}-{THUMB_WIDTH}.webp"
    ready = png.with_name(f"{slug}-{THUMB_WIDTH}.webp")
    if png.suffix == ".webp" and ready.is_file():
        shutil.copyfile(png, out / shot)
        shutil.copyfile(ready, out / thumb)
        with Image.open(png) as image:
            return shot, thumb, image.size
    with Image.open(png) as image:
        still = image.convert("RGB")
    still.save(out / shot, "WEBP", quality=84, method=5)
    # WHY crop the top bar off the thumbnail only: at 480 px Rerun's own menu bar (logo, command search) is noise
    # that pushes the view down; the full still on the entry page keeps it, as the viewer really looks.
    bar = round(still.height * RERUN_TOP_BAR)
    card = still.crop((0, bar, still.width, still.height))
    height = round(card.height * THUMB_WIDTH / card.width)
    card.resize((THUMB_WIDTH, height), Image.LANCZOS).save(out / thumb, "WEBP", quality=80, method=5)
    return shot, thumb, still.size


def make_items(gallery: list[GalleryEntry], out: Path, shots: Path | None) -> list[Item]:
    items = []
    for entry in gallery:
        if not (entry.supported and entry.write is not None):
            items.append(Item(entry))
            continue
        rrd = f"rrd/{entry.slug}.rrd"
        (out / "rrd").mkdir(parents=True, exist_ok=True)
        try:
            entry.write(out / rrd)
        except Exception as error:  # say which entry broke; never ship a page whose recording is missing
            raise RuntimeError(f"{entry.key} ({entry.source}): writing its sample recording failed: {error}") from error
        shot = thumb = None
        size = (1280, 800)
        png = find_png(shots, entry.slug)
        if png is not None:
            shot, thumb, size = convert(png, out, entry.slug)
        items.append(Item(
            entry, page=f"{section_of(entry)}/{entry.slug}.html", rrd=rrd, rrd_bytes=(out / rrd).stat().st_size,
            shot=shot, thumb=thumb, shot_size=size,
        ))  # fmt: skip
    return items


def grouped(items: list[Item]) -> list[tuple[str, list[Item]]]:
    """Items by group, groups in order of first appearance (the gallery's order: robotmsgs.com's domain order)."""
    groups: dict[str, list[Item]] = {}
    for item in items:
        groups.setdefault(item.entry.group, []).append(item)
    return list(groups.items())


def human_bytes(count: int) -> str:
    if count < 1024:
        return f"{count} B"
    if count < 1024 * 1024:
        return f"{count / 1024:.0f} KB"
    return f"{count / 1024 / 1024:.1f} MB"


# --------------------------------------------------------------------------------------- highlight

PY = re.compile(
    r"""(?P<comment>\#[^\n]*)
      |(?P<string>[rbfRBF]{0,2}(?:'[^'\n]*'|"[^"\n]*"))
      |(?P<number>(?<![\w.])-?\d+(?:\.\d+)?(?:e[+-]?\d+)?(?![\w]))
      |(?P<word>[A-Za-z_]\w*)""",
    re.X,
)
BUILTINS = {"print", "len", "range", "float", "int", "str", "tuple", "list", "dict", "isinstance", "open"}


def highlight(code: str) -> str:
    """Python with token spans (robotmsgs.com's palette classes); anything else just escapes."""
    out: list[str] = []
    last = 0
    for match in PY.finditer(code):
        out.append(esc(code[last : match.start()]))
        kind, text = match.lastgroup, match.group()
        last = match.end()
        if kind == "word":
            if text in keyword.kwlist or text in {"None", "True", "False"}:
                kind = "keyword"
            elif text in BUILTINS:
                kind = "builtin"
            elif text[0].isupper():
                kind = "type"
            else:
                out.append(esc(text))
                continue
        out.append(f'<span class="tok-{kind}">{esc(text)}</span>')
    out.append(esc(code[last:]))
    return "".join(out)


def code_block(code: str, *, label: str = "code", plain: bool = False, extra: str = "") -> str:
    body = esc(code) if plain else highlight(code)
    return (
        f'<div class="code"{extra}><button class="copy" type="button" aria-label="Copy {esc(label)}">Copy</button>'
        f"<pre><code>{body}</code></pre></div>"
    )


# --------------------------------------------------------------------------------------- components

ICONS = {
    "play": '<path d="M7 5v14l12-7z" fill="currentColor" stroke="none"/>',
    "desktop": '<rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/>',
    "download": '<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>',
    "back": '<path d="M15 6l-6 6 6 6"/>',
    "external": '<path d="M14 5h5v5M19 5l-8 8M18 14v5H5V6h5"/>',
}


def icon(name: str, size: int = 16) -> str:
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>'
    )


# WHY a red/green/blue triad for the mark: x/y/z colours are the one colour rule every view here keeps.
BRAND = (
    '<svg width="22" height="22" viewBox="0 0 32 32" aria-hidden="true">'
    '<rect x="3" y="3" width="26" height="26" rx="6" fill="none" stroke="currentColor" stroke-width="2.5"/>'
    '<path d="M11 21h11" stroke="#ef4444" stroke-width="2.5" stroke-linecap="round"/>'
    '<path d="M11 21V10" stroke="#22c55e" stroke-width="2.5" stroke-linecap="round"/>'
    '<path d="M11 21l-3.2 3.2" stroke="#3b82f6" stroke-width="2.5" stroke-linecap="round"/></svg>'
)
GITHUB_ICON = (
    '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 '
    "3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-"
    ".48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78"
    "-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 "
    "0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95"
    ".29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z\"/></svg>"
)
THEME_ICONS = (
    '<svg class="i-sun" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 '
    '1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>'
    '<svg class="i-moon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 '
    '21 12.8z"/></svg>'
)


def filter_control(n_all: int, n_supported: int, *, compact: bool = False) -> str:
    cls = "seg compact" if compact else "seg"
    return (
        f'<div class="{cls}" role="group" aria-label="Which types to show">'
        f'<button type="button" data-filter="all" aria-pressed="true">All ({n_all})</button>'
        f'<button type="button" data-filter="supported" aria-pressed="false">Supported ({n_supported})</button></div>'
    )


def search_box(placeholder: str, target: str) -> str:
    return (
        f'<div class="search"><input type="search" data-search="{target}" placeholder="{esc(placeholder)}" '
        f'aria-label="{esc(placeholder.rstrip("…"))}" autocomplete="off" spellcheck="false"><kbd>/</kbd></div>'
    )


def thumb_html(item: Item, root: str, *, big: bool = False) -> str:
    """The still, or a placeholder that says why there is none (never a blank box)."""
    entry = item.entry
    if item.shot:
        src = item.shot if big else item.thumb
        thumb_h = round(item.shot_size[1] * (1 - RERUN_TOP_BAR) * THUMB_WIDTH / item.shot_size[0])
        width, height = item.shot_size if big else (THUMB_WIDTH, thumb_h)
        alt = f"The Rerun viewer showing the sample {entry.title}"
        loading = "" if big else ' loading="lazy"'
        return f'<img src="{root}{src}" alt="{esc(alt)}" width="{width}" height="{height}"{loading} decoding="async">'
    if not item.live:  # the card's own "no view yet" badge says it; one label is enough
        return '<span class="ph" aria-hidden="true"></span>'
    return '<span class="ph"><span>no screenshot yet</span></span>'


def type_chip(key: str, by_key: dict[str, Item], root: str, site: Site | None = None) -> str:
    item = by_key.get(key)
    name = key.split("/")[-1]
    if item is not None and item.live:
        return f'<a class="chip" href="{root}{item.page}" title="{esc(key)}">{esc(name)}</a>'
    if item is None and site is not None and site.types_url and _has_view(key):
        url = f"{site.types_url}types/{key.replace('/', '-')}.html"
        return f'<a class="chip" href="{esc(url)}" rel="noopener" title="{esc(key)}: its stock view">{esc(name)}</a>'
    return f'<span class="chip off" title="{esc(key)}: no view yet">{esc(name)}</span>'


def _has_view(key: str) -> bool:
    from robot_msg_viz.registry import catalog

    return any(entry.key == key and entry.supported for entry in catalog())


def card(item: Item, root: str, by_key: dict[str, Item], site: Site | None = None) -> str:
    """A gallery card. WHY a stretched title link and not an <a> card: a use-case card holds its own type chips,
    and links cannot nest."""
    entry = item.entry
    words = " ".join((entry.title, entry.key, entry.group, entry.summary)).lower()
    name_cls = "card-name mono" if entry.kind == "type" else "card-name"
    if item.live:
        name = f'<a class="card-link" href="{root}{item.page}">{esc(entry.title)}</a>'
        flag, cls = "", "card"
    else:
        name = esc(entry.title)
        flag, cls = '<span class="badge">no view yet</span>', "card off"
    chips = ""
    if entry.kind == "use_case" and entry.consumes:
        chips = '<p class="chips">' + "".join(type_chip(key, by_key, root, site) for key in entry.consumes) + "</p>"
    supported = " data-supported" if item.live else ""
    return (
        f'<article class="{cls}" data-key="{esc(entry.key)}" data-words="{esc(words)}"{supported}>'
        f'<div class="thumb">{thumb_html(item, root)}</div>'
        f'<div class="card-body"><h3 class="{name_cls}">{name}{flag}</h3>'
        f'<p class="card-sum">{esc(entry.summary)}</p>{chips}</div></article>'
    )


def sidebar(items: list[Item], root: str, current: str, site: Site) -> str:
    """The left rail on entry pages (robotmsgs.com's): search, the filter, use cases, then types by domain."""
    types = [i for i in items if i.entry.kind == "type"]
    cases = [i for i in items if i.entry.kind != "type"]
    groups = [sidebar_group("use cases", cases, root, current)] if cases else []
    for domain, members in grouped(types):
        groups.append(sidebar_group(domain, members, root, current))
    filters = filter_control(len(types), sum(i.live for i in types), compact=True) if types else ""
    return (
        f'<aside id="sidebar" class="sidebar">{drawer_nav(root, site)}'
        f'<div class="sb-top">{search_box("Search…", "sidebar")}'
        f"{filters}</div>"
        f'<nav class="sb-nav" aria-label="Gallery">{"".join(groups)}<p class="sb-empty" hidden>Nothing matches.</p></nav>'
        "</aside>"
    )


def sidebar_group(label: str, members: list[Item], root: str, current: str) -> str:
    rows = []
    for item in members:
        entry = item.entry
        words = esc(" ".join((entry.title, entry.key, label)).lower())
        here = ' aria-current="page"' if entry.key == current else ""
        if item.live:
            rows.append(f'<li data-words="{words}" data-supported><a href="{root}{item.page}"{here}>{esc(entry.title)}</a></li>')
        else:
            rows.append(f'<li class="off" data-words="{words}"><span title="no view yet">{esc(entry.title)}</span></li>')
    return f'<div class="sb-group" data-group><h2 class="sb-h">{esc(label)}</h2><ul>{"".join(rows)}</ul></div>'


def nav_links(root: str, section: str, site: Site) -> str:
    def here(key: str) -> str:
        return ' aria-current="page"' if section == key else ""

    github = f'<a href="{esc(site.github)}" rel="noopener">GitHub<span class="ext" aria-hidden="true">↗</span></a>'
    if site.use_cases_only:
        return f'<a href="{root}index.html" data-nav="use-cases"{here("use-cases")}>Use cases</a>{github}'
    return (
        f'<a href="{root}index.html#types" data-nav="types"{here("types")}>Types</a>'
        f'<a href="{root}index.html#use-cases" data-nav="use-cases"{here("use-cases")}>Use cases</a>{github}'
    )


def drawer_nav(root: str, site: Site) -> str:
    return f'<nav class="drawer-nav" aria-label="Site">{nav_links(root, "", site)}</nav>'


def shell(*, title: str, description: str, root: str, body: str, version: str, site: Site, section: str = "",
          side: str = "") -> str:  # fmt: skip
    layout = "docs" if side else "wide"
    menu = (
        '<button class="icon-btn menu-btn" type="button" aria-label="Open menu" aria-expanded="false" '
        'aria-controls="sidebar"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16"/></svg></button>'
        if side
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(description)}">
<meta name="color-scheme" content="dark light">
<meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{esc(description)}">
<link rel="icon" href="{root}static/favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap">
<script>try{{var t=localStorage.getItem("theme");if(t!=="light"&&t!=="dark")t=matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";document.documentElement.dataset.theme=t;if(localStorage.getItem("rmv-filter")==="supported")document.documentElement.dataset.only="supported"}}catch(e){{}}</script>
<link rel="stylesheet" href="{root}static/site.css?v={version}">
</head>
<body class="{layout}" data-root="{root}">
<a class="skip" href="#main">Skip to content</a>
<header class="topbar">
  <div class="topbar-in">
    {menu}
    <a class="brand" href="{root}index.html">{BRAND}<span>{esc(site.title)}</span></a>
    <nav class="topnav" aria-label="Main">{nav_links(root, section, site)}</nav>
    <div class="spacer"></div>
    <a class="icon-btn gh-btn" href="{esc(site.github)}" rel="noopener" aria-label="{esc(site.title)} on GitHub" title="GitHub">{GITHUB_ICON}</a>
    <button class="icon-btn theme-btn" type="button" aria-label="Switch between light and dark">{THEME_ICONS}</button>
  </div>
</header>
<div class="shell">
{side}
<main id="main" class="main" tabindex="-1">
{body}
<footer class="foot"><span>Rerun {RERUN_SDK_VERSION} · generated from <code>robot_msg_viz.gallery.entries()</code>, not written per page</span><span><a href="{esc(site.github)}" rel="noopener">GitHub</a> · MIT license</span></footer>
</main>
</div>
<div class="scrim" hidden></div>
<script src="{root}static/site.js?v={version}" defer></script>
</body>
</html>
"""


# ------------------------------------------------------------------------------------------- pages

def hero_media(site: Site, root: str) -> str:
    """The index's hero: a short muted, looping video of the real viewer, its poster the fallback (no video, reduced
    motion, or before it loads)."""
    if not site.hero:
        return ""
    poster = f' poster="{root}{esc(site.hero_poster)}"' if site.hero_poster else ""
    still = (f'<img src="{root}{esc(site.hero_poster)}" alt="The Rerun viewer playing the gallery\'s recordings" '
             'width="1280" height="800">' if site.hero_poster else "")  # fmt: skip
    return (
        '<figure class="hero-media">'
        f'<video autoplay muted loop playsinline preload="auto" width="1280" height="800"{poster} '
        'aria-label="The real Rerun web viewer playing the gallery\'s recordings">'
        f'<source src="{root}{esc(site.hero)}" type="video/mp4">{still}</video>'
        '<figcaption class="muted small">Recorded in the real Rerun web viewer, from the recordings below.</figcaption>'
        "</figure>"
    )


EMPTY_USE_CASES = (
    '<p class="none">No use cases yet: dashboards for a whole job come from plugins (entry-point group '
    "<code>robot_msg_viz.gallery</code>), and none is installed in this build.</p>"
)


def index_page(items: list[Item], version: str, site: Site | None = None) -> str:
    site = site or Site()
    root = ""
    by_key = {i.entry.key: i for i in items}
    types = [i for i in items if i.entry.kind == "type"]
    cases = [i for i in items if i.entry.kind != "type"]
    n_live = sum(i.live for i in types)

    sections = []
    for domain, members in grouped(types):
        live = sum(i.live for i in members)
        cards = "".join(card(i, root, by_key, site) for i in members)
        sections.append(
            f'<section class="group" data-group aria-labelledby="d-{esc(domain)}">'
            f'<div class="group-h"><h2 id="d-{esc(domain)}">{esc(domain)}</h2>'
            f'<p class="muted"><code>robot_msgs.{esc(domain)}</code> · {live} of {len(members)} with a view</p></div>'
            f'<div class="grid">{cards}</div></section>'
        )
    types_panel = (
        f'<div class="toolbar">{filter_control(len(types), n_live)}{search_box("Search types…", "types")}</div>'
        + "".join(sections)
        + '<p class="empty" hidden>No type matches. <button type="button" class="linkish" data-reset>Show all</button></p>'
    )

    if cases:
        case_sections = []
        for category, members in grouped(cases):
            cards = "".join(card(i, root, by_key, site) for i in members)
            case_sections.append(
                f'<section class="group" data-group aria-labelledby="c-{esc(category)}">'
                f'<div class="group-h"><h2 id="c-{esc(category)}">{esc(category)}</h2>'
                f'<p class="muted">{len(members)} dashboard{"" if len(members) == 1 else "s"}</p></div>'
                f'<div class="grid">{cards}</div></section>'
            )
        cases_panel = (
            '<p class="muted panel-note">Preconfigured dashboards for a whole job: several robot_msgs types in one '
            "Rerun layout, each with sample data.</p>" + "".join(case_sections)
        )
    else:
        # WHY one plain line, naming no plugin: the public page is built with none installed (a plugin may be private)
        cases_panel = EMPTY_USE_CASES

    code = f'<div class="hero-code">{code_block(site.code, label="example")}</div>' if site.code else ""
    hero = f"""<section class="hero{' no-code' if not site.code else ''}" aria-labelledby="hero-h">
  <div class="hero-text">
    <p class="eyebrow">Rerun {RERUN_SDK_VERSION} · robot_msgs</p>
    <h1 id="hero-h">{esc(site.title)} <span class="dash">—</span> {esc(site.tagline)}</h1>
    <div class="lead">{site.lead_html}</div>
  </div>
  {code}
</section>
{hero_media(site, root)}"""
    title = f"{site.title} — {site.tagline}"
    if site.use_cases_only:
        body = f"""<div class="home">
{hero}
<section class="panel" id="panel-use-cases" aria-label="Use cases" data-panel="use-cases">
{cases_panel}
</section>
</div>"""
        return shell(title=title, description=f"{site.tagline}: dashboards on Rerun {RERUN_SDK_VERSION}, each with a "
                     "screenshot and sample data.", root=root, body=body, version=version, site=site,
                     section="use-cases")  # fmt: skip

    body = f"""<div class="home">
{hero}

<div class="tabs" role="tablist" aria-label="Gallery">
  <button class="tab" type="button" role="tab" id="tab-types" data-tab="types" aria-controls="panel-types" aria-selected="true">Types <span class="n">{len(types)}</span></button>
  <button class="tab" type="button" role="tab" id="tab-use-cases" data-tab="use-cases" aria-controls="panel-use-cases" aria-selected="false" tabindex="-1">Use cases <span class="n">{len(cases)}</span></button>
</div>
<section class="panel" id="panel-types" role="tabpanel" aria-labelledby="tab-types" data-panel="types">
{types_panel}
</section>
<section class="panel" id="panel-use-cases" role="tabpanel" aria-labelledby="tab-use-cases" data-panel="use-cases" hidden>
{cases_panel}
</section>
</div>"""
    return shell(
        title=title,
        description=f"A stock Rerun {RERUN_SDK_VERSION} view for every robot_msgs type, and dashboards for common "
        "robotics use cases, each with a screenshot and sample data.",
        root=root, body=body, version=version, site=site, section="types",
    )  # fmt: skip


def entry_page(item: Item, items: list[Item], version: str, site: Site | None = None) -> str:
    site = site or Site()
    root = "../"
    entry = item.entry
    by_key = {i.entry.key: i for i in items}
    is_type = entry.kind == "type"
    section = section_of(entry)
    crumb_root = "Types" if is_type else "Use cases"

    if is_type:
        domain, name = entry.key.split("/", 1)
        meta = (
            f'<code>robot_msgs.{esc(domain)}.{esc(name)}</code> · '
            f'<a href="{ROBOT_MSGS_DOCS.format(domain=domain, name=name)}" rel="noopener">its fields and functions '
            f"on robotmsgs.com</a>"
        )
        users = [i for i in items if i.entry.kind != "type" and entry.key in i.entry.consumes]
        related_h = "Used in"
        related = (
            '<p class="chips">' + "".join(f'<a class="chip" href="{root}{u.page}">{esc(u.entry.title)}</a>'
                                          if u.live else f'<span class="chip off">{esc(u.entry.title)}</span>'
                                          for u in users) + "</p>"
            if users else '<p class="muted">No use case draws this type yet.</p>'
        )  # fmt: skip
    else:
        meta = f"{esc(entry.group)} · a dashboard from <code>{esc(entry.source)}</code>"
        # WHY both lines (Zach, 2026-11-24): adapters bring any data in as robot_msgs, viewers draw it; Rerun today,
        # the rest on the roadmap, so the page says which is which instead of implying more.
        viewer_chips = "".join(
            f'<span class="chip">{esc(name)}</span>' if status == "supported"
            else f'<span class="chip off" title="on the roadmap">{esc(name)} · roadmap</span>'
            for name, status in entry.viewers
        )
        adapter_chips = "".join(f'<span class="chip">{esc(name)}</span>' for name in entry.adapters)
        meta += (f'</p><p class="meta-line">Viewers <span class="chips">{viewer_chips}</span></p>'
                 f'<p class="meta-line">Takes (adapters) <span class="chips">{adapter_chips}</span>')
        related_h = "Types it draws"
        related = (
            '<p class="chips">' + "".join(type_chip(k, by_key, root, site) for k in entry.consumes) + "</p>"
            if entry.consumes else '<p class="muted">It lists no robot_msgs types.</p>'
        )  # fmt: skip

    sample = (
        "a recording over time: drag the timeline at the bottom to scrub it"
        if entry.timed
        else "one message, built in code from a seeded sample"
    )
    desktop = launch.desktop_command("{url}")
    rrd_name = item.rrd.split("/")[-1] if item.rrd else ""
    # WHY from the download on a private site: behind a sign-in the desktop viewer gets the sign-in page, not the file
    private = f' data-desktop-file="~/Downloads/{esc(rrd_name)}"' if site.private else ""
    question = "" if is_type else '<p class="eyebrow">The question it answers</p>'
    links = ""
    if entry.links:
        links = '<p class="meta-line">' + " · ".join(
            f'<a href="{esc(url)}" rel="noopener">{esc(label)}<span class="ext" aria-hidden="true">↗</span></a>'
            for label, url in entry.links) + "</p>"  # fmt: skip
    body = f"""<article class="entry">
<nav class="crumbs" aria-label="Breadcrumb"><a href="{root}index.html#{section}">{crumb_root}</a> <span aria-hidden="true">/</span> <span>{esc(entry.group)}</span> <span aria-hidden="true">/</span> <span aria-current="page">{esc(entry.title)}</span></nav>
<h1>{esc(entry.title)}</h1>
{question}<p class="lead">{esc(entry.summary)}</p>
<p class="meta-line">{meta}</p>
{links}
<div class="player" data-rrd="{root}{item.rrd}" data-web-viewer="{esc(launch.WEB_VIEWER)}" data-desktop="{esc(desktop)}" data-title="{esc(entry.title)}"{private}>
  <div class="stage">
    <button class="stage-shot" type="button" data-act="web" aria-label="Open {esc(entry.title)} in the browser viewer">
      {thumb_html(item, root, big=True)}
      <span class="play">{icon("play", 18)}<span>Open in browser</span></span>
    </button>
  </div>
  <div class="actions">
    <button class="btn primary" type="button" data-act="web">{icon("play").replace("<svg", '<svg class="i-play"', 1)}{icon("back").replace("<svg", '<svg class="i-back"', 1)}<span>Open in browser</span></button>
    <button class="btn" type="button" data-act="desktop">{icon("desktop")}<span>Open on desktop</span></button>
    <a class="btn" href="{root}{item.rrd}" download="{esc(rrd_name)}">{icon("download")}<span>Download .rrd</span><span class="size">{human_bytes(item.rrd_bytes)}</span></a>
    <a class="newtab" href="#" target="_blank" rel="noopener" hidden>Open the viewer in a new tab<span class="ext" aria-hidden="true">↗</span></a>
  </div>
  <p class="notice" hidden></p>
  <div class="desktop" hidden>
    <p class="desktop-h"></p>
    {code_block("", label="desktop command", plain=True, extra=' data-command')}
    <p class="muted small">It runs Rerun's desktop viewer on this recording; <code>uvx</code> fetches it the first time and installs nothing for good. Already have it? <code>pip install rerun-sdk=={RERUN_SDK_VERSION}</code> once, then <code>rerun &lt;url&gt;</code>.</p>
  </div>
  <p class="muted small sample">Sample data: <a href="{root}{item.rrd}" download="{esc(rrd_name)}"><code>{esc(rrd_name)}</code></a> ({human_bytes(item.rrd_bytes)}), {sample}. The layout (a Rerun blueprint) is inside the file.</p>
</div>

<section aria-labelledby="use-it"><h2 id="use-it">Use it in your code</h2>
{code_block(entry.code, label="code")}
</section>
<section aria-labelledby="related"><h2 id="related">{related_h}</h2>
{related}
</section>
</article>"""
    return shell(
        title=f"{entry.title} · {site.title}",
        description=entry.summary,
        root=root, body=body, version=version, site=site, section=section, side=sidebar(items, root, entry.key, site),
    )  # fmt: skip


# ------------------------------------------------------------------------------------------- build


def clean(out: Path) -> None:
    for name in GENERATED:
        target = out / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def manifest_row(item: Item) -> dict:
    entry = item.entry
    return {
        "kind": entry.kind, "key": entry.key, "slug": entry.slug, "title": entry.title, "group": entry.group,
        "summary": entry.summary, "supported": entry.supported, "code": entry.code, "consumes": list(entry.consumes),
        "source": entry.source, "timed": entry.timed, "tags": list(entry.tags),
        "page": item.page, "rrd": item.rrd, "rrd_bytes": item.rrd_bytes, "shot": item.shot, "thumb": item.thumb,
    }  # fmt: skip


def build(
    out: str | Path, shots: str | Path | None = None, gallery: list[GalleryEntry] | None = None, *,
    site: Site | None = None, hero: str | Path | None = None, viewer: bool | str | Path = False,
) -> dict:  # fmt: skip
    """Write the site into ``out``; returns counts. ``gallery`` defaults to `robot_msg_viz.gallery.entries()`.

    ``hero``: an .mp4 for the index (its poster: the same name with .webp, beside it). ``viewer``: True vendors Rerun's
    web viewer from the installed rerun-sdk into OUT/viewer/ (a folder: from there), and OUT/viewer.json points at it."""
    out = Path(out).resolve()
    shots = Path(shots).resolve() if shots is not None else None
    gallery = gallery_entries() if gallery is None else list(gallery)
    site = site or Site()
    out.mkdir(parents=True, exist_ok=True)
    clean(out)
    shutil.copytree(STATIC, out / "static")
    if hero is not None:
        hero = Path(hero).resolve()
        (out / "media").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(hero, out / "media" / hero.name)
        poster = hero.with_suffix(".webp")
        if poster.is_file():
            shutil.copyfile(poster, out / "media" / poster.name)
        site = dataclasses.replace(site, hero=f"media/{hero.name}",
                                   hero_poster=f"media/{poster.name}" if poster.is_file() else None)  # fmt: skip
    vendored = None
    if viewer:
        from robot_msg_viz.site.viewer import vendor_viewer

        vendored = vendor_viewer(out, source=None if viewer is True else viewer)
    digest = hashlib.sha256()
    for name in sorted(p.name for p in STATIC.iterdir() if p.is_file()):
        digest.update((STATIC / name).read_bytes())
    version = digest.hexdigest()[:8]

    items = make_items(gallery, out, shots)
    (out / "index.html").write_text(index_page(items, version, site), encoding="utf-8")
    for item in items:
        if item.live:
            page = out / item.page
            page.parent.mkdir(parents=True, exist_ok=True)
            page.write_text(entry_page(item, items, version, site), encoding="utf-8")
    (out / "gallery.json").write_text(json.dumps([manifest_row(i) for i in items], indent=1), encoding="utf-8")
    types = [i for i in items if i.entry.kind == "type"]
    return {
        "out": str(out),
        "types": len(types),
        "supported": sum(i.live for i in types),
        "use_cases": sum(i.entry.kind != "type" for i in items),
        "screenshots": sum(i.shot is not None for i in items),
        "hero": site.hero,
        "viewer": vendored,
    }
