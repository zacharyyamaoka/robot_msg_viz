"""Serve a built site on 127.0.0.1 with CORS, plus Rerun's own web viewer next to it.

    python -m robot_msg_viz.site serve OUT [--port 4493]

One command, two servers: OUT on ``--port`` (every response carries ``Access-Control-Allow-Origin: *``, so a
viewer on another origin can fetch the .rrd files), and the web viewer bundled with rerun-sdk
(``rerun --serve-web``) on ``--port + 1``. It answers ``/viewer.json`` naming that local viewer; the pages
prefer it over the hosted app.rerun.io, because an https page fetching http://127.0.0.1 is unreliable
(mixed content, Private Network Access). Ctrl-C or SIGTERM stops both.

WHY viewer.json is answered from memory, not written into OUT: a file would be wiped by a rebuild while serving,
and one left behind by a killed server would ship in a static deploy pointing at a dead localhost viewer.
"""

from __future__ import annotations

import functools
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from robot_msg_viz import launch

HOST = "127.0.0.1"


class Handler(SimpleHTTPRequestHandler):
    viewer_json: bytes | None = None  # set by serve() once the local web viewer answers
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".rrd": "application/octet-stream", ".webp": "image/webp"}

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        if self.path.split("?")[0].endswith("viewer.json"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?")[0] == "/viewer.json" and self.viewer_json is not None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(self.viewer_json)))
            self.end_headers()
            self.wfile.write(self.viewer_json)
            return
        super().do_GET()

    def do_OPTIONS(self) -> None:  # noqa: N802  (http.server's naming)
        # WHY: Chrome's Private Network Access preflight, when a public page (app.rerun.io) fetches from 127.0.0.1.
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        if len(args) > 1 and str(args[1])[:1] in ("4", "5"):  # log_request(code): only failed requests
            super().log_message(format, *args)


def free_port(preferred: int | None = None) -> int:
    """``preferred`` when nothing listens on it, else any free port."""
    for port in ([preferred] if preferred else []) + [0]:
        with socket.socket() as probe:
            try:
                probe.bind((HOST, port))
            except OSError:
                continue
            return probe.getsockname()[1]
    raise RuntimeError("no free port")


def rerun_command() -> list[str]:
    """The ``rerun`` next to this Python (the venv's rerun-sdk), else the one launch.py finds."""
    beside = Path(sys.executable).with_name("rerun")
    return [str(beside)] if beside.exists() else launch.rerun_binary()


def wait_for(url: str, timeout: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(0.3)
    return False


def serve(out: str | Path, port: int = 4493) -> int:
    out = Path(out).resolve()
    if not (out / "index.html").is_file():
        raise SystemExit(f"{out} has no index.html: run `python -m robot_msg_viz.site build {out}` first")
    # Bind the site first: a busy --port fails here, before a viewer is started.
    handler = type("SiteHandler", (Handler,), {})
    server = ThreadingHTTPServer((HOST, port), functools.partial(handler, directory=str(out)))
    web_port = free_port(port + 1)
    grpc_port = free_port()
    viewer = subprocess.Popen(
        # WHY --bind: rerun listens on every interface by default; this is a local preview, not a LAN service.
        [*rerun_command(), "--serve-web", "--bind", HOST, "--web-viewer-port", str(web_port), "--port", str(grpc_port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )  # fmt: skip
    def stop(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        web = f"http://{HOST}:{web_port}/"
        if wait_for(web):
            handler.viewer_json = json.dumps({"web_viewer": web}).encode()
            note = f"web viewer {web}"
        else:
            note = "the local web viewer did not start; pages use the hosted one"
        print(f"http://{HOST}:{port}/  ({note}; Ctrl-C stops both)", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        try:
            os.killpg(viewer.pid, signal.SIGTERM)
            viewer.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(viewer.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return 0
