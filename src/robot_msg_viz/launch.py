"""Open a recording: in the desktop viewer if you have it, else in the browser.

A web page cannot tell whether the desktop viewer is installed, so the gallery offers both:
``web_url`` (Rerun's hosted web viewer playing the .rrd, nothing to install) and
``desktop_command`` (one line that runs the native viewer on the same URL; ``uvx`` fetches it if
it is not installed). From Python, `open` prefers the desktop viewer and falls back to the web one.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

from robot_msg_viz._engine import RERUN_SDK_VERSION

# WHY the versioned URL: the hosted viewer must be at least as new as the SDK that wrote the .rrd,
# and the newest one may not read old files; robotmsgs.com links the same version.
WEB_VIEWER = f"https://app.rerun.io/version/{RERUN_SDK_VERSION}/index.html"


def web_url(rrd_url: str) -> str:
    """Rerun's hosted web viewer playing ``rrd_url`` (an http(s) URL the browser can fetch)."""
    # WHY persist=0: the hosted viewer otherwise remembers each app id's layout across visits, so a gallery
    # entry whose blueprint changed would open in the stale one (prior-art sweep 2026-11-24, claim 46).
    return f"{WEB_VIEWER}?url={quote(rrd_url, safe='')}&persist=0"


def desktop_command(rrd_url: str) -> str:
    """One shell line that opens ``rrd_url`` in the native viewer (installs nothing permanently)."""
    return f"uvx --from rerun-sdk=={RERUN_SDK_VERSION} rerun {rrd_url}"


def rerun_binary() -> list[str]:
    """The native viewer: the ``rerun`` on PATH, else the one inside this Python's rerun-sdk."""
    found = shutil.which("rerun")
    return [found] if found else [sys.executable, "-m", "rerun"]


def open(path_or_url: str | Path, *, web: bool = False) -> subprocess.Popen:  # noqa: A001
    """Open an .rrd (path or URL) in the desktop viewer, or with ``web=True`` the local web viewer."""
    target = str(path_or_url)
    args = [*rerun_binary(), "--web-viewer", target] if web else [*rerun_binary(), target]
    return subprocess.Popen(args, start_new_session=True)
