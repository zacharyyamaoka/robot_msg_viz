"""The gallery site builds from gallery.entries(): every entry has a card, supported ones a page and a recording.

Builds once per module into a temp folder without screenshots (placeholders), so it stays a few seconds.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from robot_msg_viz.gallery import GalleryEntry, entries
from robot_msg_viz.site.build import build


def cards(index: str) -> dict[str, str]:
    """data-key -> the card's HTML."""
    found = {}
    for match in re.finditer(r'<article class="card[^"]*" data-key="([^"]+)".*?</article>', index, re.S):
        assert match.group(1) not in found, f"two cards for {match.group(1)}"
        found[match.group(1)] = match.group(0)
    return found


@pytest.fixture(scope="module")
def site(tmp_path_factory) -> tuple[Path, list[GalleryEntry]]:
    out = tmp_path_factory.mktemp("site")
    gallery = entries()
    build(out, gallery=gallery)
    return out, gallery


def test_index_has_both_tabs_and_a_card_per_entry(site):
    out, gallery = site
    index = (out / "index.html").read_text()
    assert 'id="tab-types"' in index and 'id="tab-use-cases"' in index
    assert 'id="panel-types"' in index and 'id="panel-use-cases"' in index
    assert "https://github.com/zacharyyamaoka/robot_msg_viz" in index
    assert set(cards(index)) == {e.key for e in gallery}


def test_supported_cards_link_to_pages_unsupported_do_not(site):
    out, gallery = site
    found = cards((out / "index.html").read_text())
    for entry in gallery:
        card = found[entry.key]
        links = re.findall(r'href="([^"]+)"', card)
        if entry.supported and entry.write is not None:
            folder = "types" if entry.kind == "type" else "use-cases"
            assert f"{folder}/{entry.slug}.html" in links, entry.key
            assert (out / folder / f"{entry.slug}.html").is_file(), entry.key
        else:
            assert links == [], f"{entry.key} has no view but its card links {links}"
            assert "no view yet" in card
            assert not (out / "types" / f"{entry.slug}.html").exists()


def test_filter_counts_match_the_gallery(site):
    out, gallery = site
    index = (out / "index.html").read_text()
    types = [e for e in gallery if e.kind == "type"]
    supported = [e for e in types if e.supported]
    assert f">All ({len(types)})<" in index
    assert f">Supported ({len(supported)})<" in index


def test_every_entry_page_links_its_recording(site):
    out, gallery = site
    for entry in gallery:
        if not (entry.supported and entry.write is not None):
            continue
        folder = "types" if entry.kind == "type" else "use-cases"
        page = (out / folder / f"{entry.slug}.html").read_text()
        assert f'"../rrd/{entry.slug}.rrd"' in page, entry.key
        rrd = out / "rrd" / f"{entry.slug}.rrd"
        assert rrd.is_file() and rrd.stat().st_size > 0, entry.key
        assert "Open in browser" in page and "Open on desktop" in page and "Download .rrd" in page


def test_no_absolute_paths_leak(site):
    out, _ = site
    for path in [*out.rglob("*.html"), out / "gallery.json"]:
        text = path.read_text()
        assert "/home/" not in text and str(out) not in text, path.name
        assert 'href="/' not in text and 'src="/' not in text, f"{path.name}: links must be relative"


def test_manifest_lists_every_entry(site):
    out, gallery = site
    rows = json.loads((out / "gallery.json").read_text())
    assert [r["key"] for r in rows] == [e.key for e in gallery]


def test_use_cases_tab_cards_and_type_chips(tmp_path):
    """A plugin's use case gets a card, a page and chips that link the types it draws (zero use cases is fine too)."""
    every = entries()
    pose = next(e for e in every if e.key == "geometry/Pose")
    missing = next(e for e in every if not e.supported)
    case = GalleryEntry(
        kind="use_case", key="pick_demo", title="Pick demo", group="Manipulation",
        summary="A pose to reach for.", supported=True, code="print('demo')", write=pose.write,
        consumes=(pose.key, missing.key), source="tests",
    )  # fmt: skip
    build(tmp_path, gallery=[pose, missing, case])
    index = (tmp_path / "index.html").read_text()
    card = cards(index)["pick_demo"]
    assert 'href="use-cases/pick_demo.html"' in card
    assert 'href="types/geometry-Pose.html"' in card  # a chip to the type page
    assert f'title="{missing.key}: no view yet"' in card  # greyed, no link
    assert ">Use cases <span class=\"n\">1</span>" in index
    page = (tmp_path / "use-cases" / "pick_demo.html").read_text()
    assert 'href="../types/geometry-Pose.html"' in page
    pose_page = (tmp_path / "types" / "geometry-Pose.html").read_text()
    assert 'href="../use-cases/pick_demo.html"' in pose_page  # "Used in"


def test_serve_handler_sends_cors_and_the_local_viewer(site):
    """The serve command's file server: CORS on every file, /viewer.json from memory (never a file in OUT)."""
    import functools
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from robot_msg_viz.site.serve import Handler

    out, gallery = site
    handler = type("TestHandler", (Handler,), {"viewer_json": b'{"web_viewer": "http://127.0.0.1:1/"}'})
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(handler, directory=str(out)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}/"
        rrd = next(f"rrd/{e.slug}.rrd" for e in gallery if e.supported and e.write is not None)
        with urllib.request.urlopen(base + rrd) as response:
            assert response.headers["Access-Control-Allow-Origin"] == "*"
        with urllib.request.urlopen(base + "viewer.json") as response:
            assert json.loads(response.read())["web_viewer"] == "http://127.0.0.1:1/"
        assert not (out / "viewer.json").exists()
    finally:
        server.shutdown()
        server.server_close()
