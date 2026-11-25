"""Self-host Rerun's web viewer beside the site: ``build OUT --viewer`` writes ``OUT/viewer/`` and ``OUT/viewer.json``.

WHY (2026-11-25): a site behind a sign-in (robot_dashboards' page on Cloudflare, behind Access) cannot use the hosted
app.rerun.io: that viewer fetches the .rrd cross-origin and the request carries no Access cookie, so it gets the
sign-in page instead of the recording. A viewer served from the site's own origin fetches it with the cookie.

Where the files come from: the rerun-sdk wheel this package already pins (``rerun --serve-web`` serves the viewer from
inside its binary), so the copy is exactly the SDK version that wrote the .rrd, and nothing is downloaded.

WHY the wasm in parts: Rerun 0.40's ``re_viewer_bg.wasm`` is ~43 MiB and Cloudflare Workers serve at most 25 MiB per
static asset, so it is split into ``re_viewer_bg.part<k>.wasm`` files of at most PART_BYTES and the viewer's
``index.html`` fetches and joins them (one line of its loader replaced; the rest of the page is Rerun's own). The
joined stream carries ``rerun-final-length``, the header Rerun's own server sends, so the loading bar still counts.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from robot_msg_viz._engine import RERUN_SDK_VERSION

PART_BYTES = 20 * 1024 * 1024  # under Cloudflare Workers' 25 MiB per-asset limit, with room for a bigger viewer
FILES = ("index.html", "re_viewer.js", "re_viewer_bg.wasm", "favicon.ico", "apple-touch-icon.png")
LOADER = 'const response = await fetch("./re_viewer_bg.wasm");'
JOINED = (
    "const response = await (async () => {\n"
    "            // robot_msg_viz: the wasm ships in parts (a static host's per-file limit); fetch and join them\n"
    "            const manifest = await (await fetch(\"./re_viewer_bg.parts.json\")).json();\n"
    "            const parts = await Promise.all(manifest.parts.map((part) => fetch(\"./\" + part)));\n"
    "            for (const part of parts) if (!part.ok) throw new Error(\"wasm part \" + part.url + \": \" + part.status);\n"
    "            const body = new ReadableStream({ async start(controller) {\n"
    "              for (const part of parts) {\n"
    "                const reader = part.body.getReader();\n"
    "                for (;;) { const { done, value } = await reader.read(); if (done) break; controller.enqueue(value); }\n"
    "              }\n"
    "              controller.close();\n"
    "            } });\n"
    "            return new Response(body, { status: 200, headers: { \"content-type\": \"application/wasm\",\n"
    "              \"rerun-final-length\": String(manifest.bytes) } });\n"
    "          })();"
)


def _fetch_from_sdk(folder: Path) -> None:
    """Start the SDK's own `rerun --serve-web` on free local ports, copy FILES from it, stop it."""
    from robot_msg_viz.site.serve import HOST, free_port, rerun_command, wait_for

    web, grpc = free_port(), free_port()
    viewer = subprocess.Popen(
        [*rerun_command(), "--serve-web", "--bind", HOST, "--web-viewer-port", str(web), "--port", str(grpc)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )  # fmt: skip
    try:
        base = f"http://{HOST}:{web}/"
        if not wait_for(base, timeout=90):
            raise RuntimeError("rerun --serve-web did not answer within 90 s")
        for name in FILES:
            with urllib.request.urlopen(base + name, timeout=120) as response:
                (folder / name).write_bytes(response.read())
    finally:
        try:
            os.killpg(viewer.pid, signal.SIGTERM)
            viewer.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(viewer.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def vendor_viewer(out: str | Path, *, source: str | Path | None = None, part_bytes: int = PART_BYTES) -> dict:
    """Write Rerun's web viewer to ``out/viewer/`` (wasm in parts) and ``out/viewer.json`` pointing at it.

    ``source``: a folder already holding FILES (a test, or an offline copy); default: the installed rerun-sdk.
    Returns {"files": {name: bytes}, "parts": n, "largest": bytes}."""
    out = Path(out)
    folder = out / "viewer"
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        raw = Path(source) if source is not None else Path(scratch)
        if source is None:
            _fetch_from_sdk(raw)
        index = (raw / "index.html").read_text(encoding="utf-8")
        if index.count(LOADER) != 1:
            # a newer viewer changed its loader: fail loudly rather than ship a page that cannot start
            raise RuntimeError(f"Rerun {RERUN_SDK_VERSION}'s index.html no longer has exactly one {LOADER!r}")
        (folder / "index.html").write_text(index.replace(LOADER, JOINED), encoding="utf-8")
        for name in FILES:
            if name not in ("index.html", "re_viewer_bg.wasm"):
                (folder / name).write_bytes((raw / name).read_bytes())
        wasm = (raw / "re_viewer_bg.wasm").read_bytes()
    parts = []
    for k, start in enumerate(range(0, len(wasm), part_bytes)):
        name = f"re_viewer_bg.part{k}.wasm"
        (folder / name).write_bytes(wasm[start : start + part_bytes])
        parts.append(name)
    (folder / "re_viewer_bg.parts.json").write_text(json.dumps({"parts": parts, "bytes": len(wasm)}))
    # WHY a relative folder URL: the site works under any prefix, and a host that drops index.html still serves it
    (out / "viewer.json").write_text(json.dumps({"web_viewer": "viewer/", "rerun": RERUN_SDK_VERSION}))
    files = {p.name: p.stat().st_size for p in sorted(folder.iterdir())}
    return {"files": files, "parts": len(parts), "largest": max(files.values())}
