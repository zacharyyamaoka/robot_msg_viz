"""Importing robot_msg_viz registers every view's logger: the first show() in a fresh process uses it.

Ruler addition (2026-11-24, before any agent evaluation): the contract tests ran after catalog() had loaded the
views, so a logger registered lazily looked fine there and still missed the first message in a user's script.
"""

import subprocess
import sys

CHECK = """
import robot_msg_viz as viz
from robot_msg_viz._engine import log
from robot_msgs.geometry import Inertia
assert log.dispatch(Inertia) is not log.dispatch(object), "Inertia's logger is not registered at import"
print("ok")
"""


def test_views_register_at_import():
    out = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
