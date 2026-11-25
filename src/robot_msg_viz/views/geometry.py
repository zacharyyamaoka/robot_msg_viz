"""Stock views of robot_msgs.geometry types (docs/CONTRACT.md: one TypeView per type, registered on import).

Pose is the worked example the other views follow; a lane fills in the rest of the domain.
"""

from __future__ import annotations

import rerun.blueprint as rrb
from robot_msgs.geometry import Pose

from robot_msg_viz.registry import TypeView, register


def world(name: str) -> list[rrb.View]:
    """The shared 3D scene every spatial type draws into (robot_msgs logs spatial data under /world)."""
    return [rrb.Spatial3DView(origin="/world", name="world")]


register(TypeView(
    type=Pose,
    summary="A coordinate frame: red, green and blue axes (x, y, z) at the position, turned by the orientation.",
    sample=lambda: Pose.from_xyz_rpy(0.4, 0.1, 0.3, 0.0, 0.5, 0.8),
    views=world,
))  # fmt: skip
