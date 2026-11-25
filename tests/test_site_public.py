"""The public landing page (GitHub Pages) and the builder's modes: what CI builds and deploys.

- Every supported type has a committed still in website/media/shots, and nothing else does (a plugin's stills, which
  may be private, never sit in this public repo); the hero video is H.264, 6-20 s, and all media fit in 6 MB.
- The public build (``--no-plugins``) has the hero, a still on every supported card, no use case at all and a plain
  empty-state line, and leaks no local path, vault, agent folder or e-mail address.
- A plugin's page (``--use-cases-only``) and the self-hosted viewer (``--viewer``): the wasm split under the per-file
  limit, the loader that joins it, a relative viewer.json.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import pytest

from robot_msg_viz.gallery import GalleryEntry, type_entries
from robot_msg_viz.site import viewer as site_viewer
from robot_msg_viz.site.build import EMPTY_USE_CASES, Site, build

REPO = Path(__file__).resolve().parents[1]
MEDIA = REPO / "website" / "media"
MEDIA_BUDGET = 6 * 1024 * 1024
# what must never reach a public page: a home directory, the vault, an agent's folder, an e-mail address
LEAKS = re.compile(r"/home/|zach_brain|\.claude\b|[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]*[a-z]{2,}")


def supported() -> list[GalleryEntry]:
    return [e for e in type_entries() if e.supported and e.write is not None]


def mp4_duration(path: Path) -> float:
    data = path.read_bytes()
    at = data.index(b"mvhd") + 4
    version = data[at]
    if version == 1:
        timescale, duration = struct.unpack(">IQ", data[at + 20 : at + 32])
    else:
        timescale, duration = struct.unpack(">II", data[at + 12 : at + 20])
    return duration / timescale


def test_every_supported_type_has_a_committed_still():
    missing = [e.slug for e in supported()
               if not (MEDIA / "shots" / f"{e.slug}.webp").is_file()
               or not (MEDIA / "shots" / f"{e.slug}-480.webp").is_file()]  # fmt: skip
    assert not missing, f"no still in website/media/shots for {missing}: shoot them (website/README.md, Media)"


def test_only_type_stills_are_committed():
    """A use case's still comes from a plugin, which may be private: it never sits in this public repo."""
    slugs = {e.slug for e in type_entries()}
    stray = [p.name for p in (MEDIA / "shots").iterdir() if p.name.removesuffix(".webp").removesuffix("-480") not in slugs]
    assert not stray, f"stills that are not robot_msgs types: {stray}"


def test_the_hero_is_a_short_h264_loop_and_all_media_fit_the_budget():
    hero = MEDIA / "hero.mp4"
    assert hero.is_file() and (MEDIA / "hero.webp").is_file(), "the hero video and its poster"
    head = hero.read_bytes()[:400_000]
    assert head[4:8] == b"ftyp" and b"avc1" in head, "H.264 in an mp4, moov first (plays inline everywhere)"
    assert 6.0 <= mp4_duration(hero) <= 20.0
    total = sum(p.stat().st_size for p in MEDIA.rglob("*") if p.is_file())
    assert total <= MEDIA_BUDGET, f"website/media is {total / 2**20:.1f} MiB"


@pytest.fixture(scope="module")
def public(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("public")
    build(out, shots=MEDIA / "shots", gallery=type_entries(), hero=MEDIA / "hero.mp4")
    return out


def test_the_public_page_has_the_hero_stills_and_no_use_case(public):
    index = (public / "index.html").read_text()
    assert '<video autoplay muted loop playsinline' in index
    assert 'src="media/hero.mp4"' in index and 'poster="media/hero.webp"' in index
    assert (public / "media" / "hero.mp4").is_file() and (public / "media" / "hero.webp").is_file()
    assert "no screenshot yet" not in index, "every supported card has its still"
    assert EMPTY_USE_CASES in index and ">Use cases <span class=\"n\">0</span>" in index
    assert not (public / "use-cases").exists()
    assert all(row["kind"] == "type" for row in json.loads((public / "gallery.json").read_text()))
    assert not (public / "viewer.json").exists(), "the public page uses the hosted viewer"


def test_the_public_page_leaks_nothing_private(public):
    for path in public.rglob("*"):
        if path.suffix in (".html", ".json", ".js", ".css", ".svg"):
            found = LEAKS.findall(path.read_text(encoding="utf-8"))
            assert not found, f"{path.relative_to(public)}: {found[:3]}"


FAKE_INDEX = """<html><script src="re_viewer.js"></script><script>
async function wasm_with_progress() {
          const response = await fetch("./re_viewer_bg.wasm");
}</script></html>"""


def fake_viewer(folder: Path, wasm: bytes) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "index.html").write_text(FAKE_INDEX)
    (folder / "re_viewer_bg.wasm").write_bytes(wasm)
    for name in ("re_viewer.js", "favicon.ico", "apple-touch-icon.png"):
        (folder / name).write_bytes(b"x")
    return folder


def test_the_self_hosted_viewer_splits_the_wasm_and_joins_it_in_order(tmp_path):
    wasm = bytes(range(256)) * 41  # 10 496 bytes
    stats = site_viewer.vendor_viewer(tmp_path / "out", source=fake_viewer(tmp_path / "raw", wasm), part_bytes=4096)
    folder = tmp_path / "out" / "viewer"
    manifest = json.loads((folder / "re_viewer_bg.parts.json").read_text())
    assert manifest["bytes"] == len(wasm) and len(manifest["parts"]) == stats["parts"] == 3
    assert b"".join((folder / part).read_bytes() for part in manifest["parts"]) == wasm
    assert max((folder / part).stat().st_size for part in manifest["parts"]) <= 4096
    index = (folder / "index.html").read_text()
    assert site_viewer.LOADER not in index and "re_viewer_bg.parts.json" in index
    assert not (folder / "re_viewer_bg.wasm").exists()
    assert json.loads((tmp_path / "out" / "viewer.json").read_text())["web_viewer"] == "viewer/"


def test_the_real_viewers_parts_fit_cloudflares_per_file_limit():
    assert site_viewer.PART_BYTES <= 25 * 1024 * 1024


def test_a_changed_loader_fails_loudly(tmp_path):
    raw = fake_viewer(tmp_path / "raw", b"wasm")
    (raw / "index.html").write_text("<html>no loader here</html>")
    with pytest.raises(RuntimeError, match="no longer has exactly one"):
        site_viewer.vendor_viewer(tmp_path / "out", source=raw)


def test_a_plugins_own_page(tmp_path):
    """--use-cases-only: one list of use cases, its own title and intro, chips to the public type pages, the
    self-hosted viewer, and Open on desktop from the downloaded file (the page is behind a sign-in)."""
    pose = next(e for e in type_entries() if e.key == "geometry/Pose")
    case = GalleryEntry(kind="use_case", key="pick_demo", title="Pick demo", group="Manipulation",
                        summary="Where should the gripper go?", supported=True, write=pose.write,
                        consumes=("geometry/Pose",), source="tests",
                        links=(("Source on GitHub", "https://github.com/example/pick_demo"),))  # fmt: skip
    site = Site(title="pick_dashboards", tagline="dashboards for picking", lead_html="<p>Our <b>intro</b>.</p>",
                github="https://github.com/example/pick_dashboards", code="", use_cases_only=True,
                types_url="https://zacharyyamaoka.github.io/robot_msg_viz/", private=True)  # fmt: skip
    build(tmp_path, gallery=[case], site=site, viewer=fake_viewer(tmp_path / "raw", b"w" * 100))
    index = (tmp_path / "index.html").read_text()
    assert 'role="tablist"' not in index and 'id="panel-types"' not in index
    assert "pick_dashboards <span class=\"dash\">—</span> dashboards for picking" in index
    assert "<p>Our <b>intro</b>.</p>" in index and "https://github.com/example/pick_dashboards" in index
    assert 'href="https://zacharyyamaoka.github.io/robot_msg_viz/types/geometry-Pose.html"' in index
    page = (tmp_path / "use-cases" / "pick_demo.html").read_text()
    assert "The question it answers" in page and "https://github.com/example/pick_demo" in page
    assert 'data-desktop-file="~/Downloads/pick_demo.rrd"' in page
    assert (tmp_path / "viewer.json").is_file() and (tmp_path / "viewer" / "index.html").is_file()
