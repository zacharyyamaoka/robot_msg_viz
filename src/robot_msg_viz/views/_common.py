"""Helpers every view module may import (frozen with the contract: add new helpers in your own module).

- `world(name)`: the shared 3D scene (robot_msgs logs everything spatial under /world).
- `rng(seed)`: a seeded generator, so samples are deterministic.
- `DEG`: radians -> degrees for anything shown as an angle (Zach's rule: angles on screen in degrees).
- `origin_triad(recording)`: a small grey world-origin frame, so a lone pose has a reference.
"""

from __future__ import annotations

import math

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

DEG = 180.0 / math.pi


def world(name: str = "") -> list[rrb.View]:
    return [rrb.Spatial3DView(origin="/world", name="world")]


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def origin_triad(recording: rr.RecordingStream, length: float = 0.05) -> None:
    """Log a short world-origin frame at /world/origin (static)."""
    recording.log("world/origin", rr.TransformAxes3D(length), static=True)
