"""Stock views of robot_msgs.manipulation types (docs/CONTRACT.md: one TypeView per type, registered on import).

- GraspPose: a two-finger gripper drawn in the grasp frame at the closing width, the open (pre-grasp)
  width as faint fingers, the approach as an arrow into the palm, coloured by score; one label.
- Waypoint: the tool frame, the jaws when it sets a width, the moving target's velocity; one label.
- WaypointAction: the tool path as arrows in order (blue at the start to orange at the end), a small
  frame and the jaws at each waypoint, one numbered label per place the tool stops.
- NamedPose: a card (name, group, what it is for, each joint in degrees with its tolerance, why); a
  Cartesian pose also draws its tool frame, labelled, in the 3D scene.
- AllowedCollisionMatrix: the matrix as a picture, each skipped pair coloured by its reason (class
  colours through rr.AnnotationContext, so hovering a cell names it), and a card listing the pairs.
- CollisionElement: every shape of the element, shaded, in the colour of its (motion, compliance) pair, its
  margin as a faint inflated outline, a plane as a translucent patch with an arrow out of its free side (a plane
  element is a half-space, solid opposite its normal); one label with its source, tags and margin.
- CollisionScene: every element so, labelled, beside a card: the elements by source and tag, the colour key, the
  robot state in degrees and the allowed-collision pairs.

Not supported, on purpose:

- CartesianLimits: six speed, acceleration and jerk caps, pure configuration with nothing to draw
  (docs/CONTRACT.md: never fake a view); it keeps the repr-as-text fallback.

WHY the gripper glyph opens along x and approaches along +z, 40 mm deep: robot_msgs' Viser viewer
draws the same glyph (robot_msgs.viz.viser._gripper), so a grasp reads the same in both viewers.
The fingers are a drawing of the widths, not the robot's real gripper geometry.

WHY the collision colours are robot_msgs.viz.viser's ELEMENT_RGB, copied (2026-11-26): a scene should read the
same in its Viser quick look and here, and importing that module pulls in viser, which this package does not need.

WHY overrides for GraspPose, Waypoint and WaypointAction (2026-11-24): the stock logger draws a 10 cm
frame and a text label for a grasp (no jaws, no approach), an unlabelled frame for a waypoint, and a
path whose 10 cm frames swamp a pick's few-centimetre moves. NamedPose and AllowedCollisionMatrix have
no stock logger.
"""

from __future__ import annotations

# WHY empty (2026-11-25): this module registers its TypeViews on import; its functions are sample data and helpers,
# not API. The public API is robot_msg_viz's own __all__ (show/save/stream/blueprint), and robot-sdk's scanner reads
# __all__ to decide what a package publishes.
__all__: list[str] = []

import math

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from robot_msgs.geometry import Pose, PoseStamped, Vector3
from robot_msgs.manipulation import (
    AllowedCollisionMatrix,
    CollisionElement,
    CollisionScene,
    GraspPose,
    NamedPose,
    Waypoint,
    WaypointAction,
)
from robot_msgs.sensor import JointState
from robot_msgs.shape import Mesh, Plane, SolidPrimitive
from robot_msgs.std import Header

from robot_msg_viz._engine import log, part
from robot_msg_viz.registry import TypeView, register
from robot_msg_viz.views._common import DEG, origin_triad
from robot_msg_viz.views.shape import shaded

FINGER_DEPTH = 0.04  # m: finger length in the glyph (robot_msgs.viz.viser._gripper's)
APPROACH = 0.06  # m: length of the approach arrow behind the palm
TOOL_AXES = 0.03  # m: a tool frame's triad; the stock 0.1 m dwarfs a gripper
INK = (37, 99, 160)  # robot_msgs' ink blue
MUTED = (148, 163, 184)  # robot_msgs.viz.viser's MUTED: the open jaws, guides
FAINT = (165, 170, 180)
START = (90, 140, 230)  # the stock logger's first-to-last ramp (its trajectory ghosts and labels)
END = (220, 120, 20)
VELOCITY = (20, 150, 140)
LABEL_DOT = 0.002  # m: a label's anchor point
# A viridis-like ramp for a score in [0, 1], trimmed at both ends so the lowest still reads on a dark
# canvas and the highest on a light one.
SCORE_RAMP = ((72, 35, 116), (49, 104, 142), (33, 145, 140), (53, 183, 121), (144, 200, 40))


def world(name: str) -> list[rrb.View]:
    """The shared 3D scene (robot_msgs logs spatial data under /world)."""
    return [rrb.Spatial3DView(origin="/world", name="world")]


def mm(metres: float) -> str:
    """A length for a label in millimetres, four significant figures, with a true minus sign."""
    return f"{metres * 1000.0:.4g} mm".replace("-", "−")


def ramp(stops: tuple[tuple[int, int, int], ...], fraction: float) -> tuple[int, int, int]:
    """The colour `fraction` (0..1) of the way along `stops`, linearly between neighbours."""
    position = min(max(fraction, 0.0), 1.0) * (len(stops) - 1)
    low = min(int(position), len(stops) - 2)
    t = position - low
    return tuple(round(a + (b - a) * t) for a, b in zip(stops[low], stops[low + 1], strict=True))  # type: ignore[return-value]


def score_colour(score: float | None) -> tuple[int, int, int]:
    """A score in [0, 1] on the ramp; a missing score, or one outside [0, 1] whose scale we do not know, in ink."""
    if score is None or not 0.0 <= score <= 1.0:
        return INK
    return ramp(SCORE_RAMP, score)


def xyzw(pose: Pose) -> list[float]:
    q = pose.orientation
    return [q.x, q.y, q.z, q.w]


def xyz(pose: Pose) -> np.ndarray:
    return pose.position.as_array()


def tool_frame(recording: rr.RecordingStream, path: str, pose: Pose, axes: float = TOOL_AXES) -> None:
    """A tool frame at `pose`: its transform and a small red/green/blue triad (children draw in it)."""
    recording.log(path, rr.Transform3D(translation=xyz(pose), quaternion=xyzw(pose)), rr.TransformAxes3D(axes),
                  static=True)  # fmt: skip


def jaws(recording: rr.RecordingStream, path: str, width: float, colour, radius: float) -> None:
    """Two fingers `width` apart along x, tips at the tool origin, joined by the palm 40 mm back along -z."""
    half = width / 2.0
    strips = [
        [[-half, 0.0, 0.0], [-half, 0.0, -FINGER_DEPTH], [half, 0.0, -FINGER_DEPTH], [half, 0.0, 0.0]],
    ]
    recording.log(path, rr.LineStrips3D(strips, radii=radius, colors=[colour]), static=True)


def label(recording: rr.RecordingStream, path: str, position, text: str, colour=FAINT) -> None:
    recording.log(path, rr.Points3D([position], radii=LABEL_DOT, colors=[colour], labels=[text]), static=True)


# ── GraspPose ────────────────────────────────────────────────────────────────


def grasp_text(grasp: GraspPose) -> str:
    text = f"{grasp.object_id or 'grasp'} · close to {mm(grasp.width)}"
    if grasp.pre_grasp_width is not None:
        text += f", open {mm(grasp.pre_grasp_width)}"
    if grasp.score is not None:
        text += f" · score {grasp.score:.3g}"
    return text


@log.register
def _(grasp: GraspPose, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    base = f"world/{part(name or 'GraspPose')}"
    colour = score_colour(grasp.score)
    tool_frame(recording, base, grasp.pose)
    jaws(recording, f"{base}/jaws", grasp.width, colour, 0.003)
    if grasp.pre_grasp_width is not None:
        jaws(recording, f"{base}/open_jaws", grasp.pre_grasp_width, MUTED, 0.0012)
    behind = -FINGER_DEPTH - APPROACH
    recording.log(f"{base}/approach", rr.Arrows3D(origins=[[0.0, 0.0, behind]], vectors=[[0.0, 0.0, APPROACH]],
                                                  radii=0.002, colors=[colour]), static=True)  # fmt: skip
    label(recording, f"{base}/label", [0.0, 0.0, behind - 0.012], grasp_text(grasp), colour)
    origin_triad(recording)  # the 5 cm default: a reference, smaller than the grasp it frames


def sample_grasp() -> GraspPose:
    """A top-down grasp across a bottle lying on the belt in front of the robot base."""
    return GraspPose(
        pose=Pose.from_xyz_rpy(0.25, 0.05, 0.045, math.pi, 0.0, math.radians(20.0)),
        width=0.058, pre_grasp_width=0.085, score=0.87, object_id="bottle",
        header=Header(frame_id="world"),
    )  # fmt: skip


register(TypeView(
    type=GraspPose,
    summary="A two-finger gripper at the grasp: closing width solid, open width faint, approach arrow into the palm, coloured by score.",
    sample=sample_grasp,
    views=world,
    tags=("3d",),
))  # fmt: skip


# ── Waypoint ─────────────────────────────────────────────────────────────────


def waypoint_text(w: Waypoint) -> str:
    """The waypoint's name, what it means and what the gripper and path do there, in plain words."""
    text = w.name + (f" ({w.label})" if w.label and w.label != w.name else "")
    if w.gripper_width is not None:
        text += f" · jaws {mm(w.gripper_width)}"
        if w.gripper_max_effort is not None:
            text += f" at ≤ {w.gripper_max_effort:.3g} N"
    if w.duration is not None:
        text += f" · {w.duration:.3g} s to get here"
    if w.dwell:
        text += f" · wait {w.dwell:.3g} s"
    if w.blend_radius:
        text += f" · corner rounded {mm(w.blend_radius)}"
    if w.target_velocity is not None:
        text += f" · {moving_text(w.target_velocity)}"
    return text


def moving_text(velocity: Vector3) -> str:
    return f"target moves {float(np.linalg.norm(velocity.as_array())):.3g} m/s"


def velocity_arrow(recording: rr.RecordingStream, path: str, at: np.ndarray, velocity: Vector3) -> None:
    """The moving target's velocity as a teal arrow of what it travels in one second (its words are in the
    waypoint's label, so two labels never print over each other)."""
    recording.log(path, rr.Arrows3D(origins=[at], vectors=[velocity.as_array()], radii=0.0015, colors=[VELOCITY]),
                  static=True)  # fmt: skip


@log.register
def _(waypoint: Waypoint, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    base = f"world/{part(name or waypoint.name)}"
    tool_frame(recording, f"{base}/tool", waypoint.pose)
    if waypoint.gripper_width is not None:
        jaws(recording, f"{base}/tool/jaws", waypoint.gripper_width, INK, 0.0025)
    label(recording, f"{base}/tool/label", [0.0, 0.0, -FINGER_DEPTH - 0.015], waypoint_text(waypoint), INK)
    if waypoint.target_velocity is not None:
        velocity_arrow(recording, f"{base}/target_velocity", xyz(waypoint.pose), waypoint.target_velocity)
    origin_triad(recording)


register(TypeView(
    type=Waypoint,
    summary="The tool frame at the waypoint, its jaws when it sets a width and the moving target's velocity, with one label.",
    sample=lambda: Waypoint(
        name="close", label="close", pose=Pose.from_xyz_rpy(0.25, 0.05, 0.045, math.pi, 0.0, math.radians(20.0)),
        gripper_width=0.058, gripper_max_effort=20.0, dwell=0.2, target_velocity=Vector3(0.1, 0.0, 0.0),
    ),
    views=world,
    tags=("3d",),
))  # fmt: skip


# ── WaypointAction ───────────────────────────────────────────────────────────


def stops(action: WaypointAction, tolerance: float = 1e-3) -> list[tuple[np.ndarray, list[int]]]:
    """The places the tool goes, in first-visit order, each with every waypoint index at it.

    WHY group: a pick's grasp and close (and a cycle's start and ready) share one pose, and their
    labels would print on top of each other.
    """
    places: list[tuple[np.ndarray, list[int]]] = []
    for i, w in enumerate(action.waypoints):
        here = xyz(w.pose)
        for at, indices in places:
            if np.linalg.norm(at - here) <= tolerance:
                indices.append(i)
                break
        else:
            places.append((here, [i]))
    return places


def short_text(i: int, w: Waypoint) -> str:
    text = f"{i} {w.name}" + (f" ({w.label})" if w.label and w.label != w.name else "")
    if w.gripper_width is not None:
        text += f" · jaws {mm(w.gripper_width)}"
    return text


@log.register
def _(action: WaypointAction, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    base = f"world/{part(name or 'WaypointAction')}"
    count = len(action.waypoints)
    colours = [ramp((START, END), i / max(count - 1, 1)) for i in range(count)]
    points = np.array([xyz(w.pose) for w in action.waypoints])
    moves = [i for i in range(count - 1) if np.linalg.norm(points[i + 1] - points[i]) > 1e-6]
    if moves:
        recording.log(f"{base}/path", rr.Arrows3D(
            origins=points[moves], vectors=points[[i + 1 for i in moves]] - points[moves],
            radii=0.0015, colors=[colours[i + 1] for i in moves],
        ), static=True)  # fmt: skip
    for i, w in enumerate(action.waypoints):
        path = f"{base}/{i}_{part(w.name)}"
        tool_frame(recording, path, w.pose, axes=0.025)
        if w.gripper_width is not None:
            jaws(recording, f"{path}/jaws", w.gripper_width, colours[i], 0.0018)
    # WHY one arrow at the catch, not one per moving waypoint: a belt pick rides the target for several
    # waypoints and their arrows would bury the pick; the catch is where the generator syncs.
    moving = [i for i, w in enumerate(action.waypoints) if w.target_velocity is not None]
    sync = action.sync_target_index()
    catch = sync if sync is not None else (moving[0] if moving else None)
    for k, (at, indices) in enumerate(stops(action)):
        text = "; ".join(short_text(i, action.waypoints[i]) for i in indices)
        if catch in indices and action.waypoints[catch].target_velocity is not None:
            text += f" · {moving_text(action.waypoints[catch].target_velocity)}"
            text += ", caught here" if sync is not None else ""
        recording.log(f"{base}/labels/{k}", rr.Points3D([at], radii=0.004, colors=[colours[indices[0]]],
                                                        labels=[text]), static=True)  # fmt: skip
    if catch is not None and action.waypoints[catch].target_velocity is not None:
        velocity_arrow(recording, f"{base}/target_velocity", points[catch], action.waypoints[catch].target_velocity)
    origin_triad(recording, length=0.1)


def sample_action() -> WaypointAction:
    """A pick off a moving belt and a place in a bin: 11 waypoints, the tool pointing down throughout."""
    belt = Vector3(0.1, 0.0, 0.0)
    pick, place = math.radians(20.0), math.radians(-35.0)

    def at(x: float, y: float, z: float, yaw: float) -> Pose:
        return Pose.from_xyz_rpy(x, y, z, math.pi, 0.0, yaw)

    return WaypointAction(
        tool_frame="tcp", header=Header(frame_id="world"), sync_target="close",
        waypoints=(
            Waypoint(name="start", label="start", pose=at(0.25, 0.25, 0.40, 0.0)),
            Waypoint(name="approach", label="approach", pose=at(0.45, -0.05, 0.25, pick), gripper_width=0.085,
                     blend_radius=0.03, target_velocity=belt),
            Waypoint(name="pre_grasp", label="pre_grasp", pose=at(0.45, -0.05, 0.10, pick), target_velocity=belt),
            Waypoint(name="grasp", label="grasp", pose=at(0.45, -0.05, 0.045, pick), target_velocity=belt),
            Waypoint(name="close", label="close", pose=at(0.45, -0.05, 0.045, pick), gripper_width=0.058,
                     gripper_max_effort=20.0, dwell=0.2, target_velocity=belt),
            Waypoint(name="lift", label="lift", pose=at(0.45, -0.05, 0.20, pick), target_velocity=belt,
                     sync_at_target_time=True),
            Waypoint(name="carry", label="carry", pose=at(0.10, -0.40, 0.30, place), blend_radius=0.05),
            Waypoint(name="place", label="place", pose=at(0.10, -0.40, 0.12, place)),
            Waypoint(name="release", label="release", pose=at(0.10, -0.40, 0.12, place), gripper_width=0.085),
            Waypoint(name="retreat", label="retreat", pose=at(0.10, -0.40, 0.30, place)),
            Waypoint(name="ready", label="ready", pose=at(0.25, 0.25, 0.40, 0.0)),
        ),
    )  # fmt: skip


register(TypeView(
    type=WaypointAction,
    summary="The tool path in order (blue start to orange end) with a small frame, the jaws and a numbered label at each waypoint.",
    sample=sample_action,
    views=world,
    tags=("3d",),
))  # fmt: skip


# ── NamedPose ────────────────────────────────────────────────────────────────


def named_pose_views(name: str) -> list[rrb.View]:
    return [
        rrb.Spatial3DView(origin="/world", name="world"),
        rrb.TextDocumentView(origin=f"/notes/{part(name)}", name=name),
    ]


def degrees(value: float) -> str:
    return f"{value * DEG:.4g}°".replace("-", "−")


def named_pose_card(pose: NamedPose) -> str:
    """Markdown: what the pose is, each joint in degrees (Zach's rule; stored in radians) and why."""
    lines = [f"### {pose.name} · {pose.group}", ""]
    if pose.description:
        lines += [pose.description, ""]
    if pose.tool_pose is not None:
        p = pose.tool_pose.pose
        roll, pitch, yaw = p.orientation.to_euler()
        frame = pose.tool_pose.header.frame_id or "its frame"
        lines += [
            f"**Cartesian:** `{pose.tip_link}` at x {mm(p.position.x)}, y {mm(p.position.y)}, z {mm(p.position.z)}; "
            f"roll {degrees(roll)}, pitch {degrees(pitch)}, yaw {degrees(yaw)}, in `{frame}`. "
            "The joint values below are the IK seed, not the target.",
            "",
        ]
    column = "IK seed" if pose.tool_pose is not None else "target"
    tolerance = pose.tolerance
    header = f"| joint | {column} |" + (" tolerance |" if tolerance is not None else "")
    lines += [header, "|---|---:|" + ("---:|" if tolerance is not None else "")]
    for k, joint in enumerate(pose.joint_names):
        row = f"| {joint} | {degrees(float(pose.positions[k]))} |"
        if tolerance is not None:
            row += f" ±{degrees(float(tolerance[k]))} |"
        lines.append(row)
    lines.append("")
    if pose.why:
        lines += [f"*Why these numbers:* {pose.why}", ""]
    lines.append(f"digest `{pose.digest()}`")
    return "\n".join(lines)


@log.register
def _(pose: NamedPose, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    key = part(name or pose.name)
    recording.log(f"notes/{key}", rr.TextDocument(named_pose_card(pose), media_type="text/markdown"), static=True)
    if pose.tool_pose is None:
        return  # WHY nothing in 3D: a joint-space pose has no place in space without the robot's model
    tool = pose.tool_pose.pose
    base = f"world/{key}"
    tool_frame(recording, base, tool, axes=0.08)
    frame = pose.tool_pose.header.frame_id or "its frame"
    label(recording, f"{base}/label", [0.0, 0.0, -0.02], f"{pose.name} · {pose.tip_link} in {frame}", INK)
    origin_triad(recording, length=0.1)


def sample_named_pose() -> NamedPose:
    """A UR-style arm's saved "over_bin" tool pose, with its IK seed in degrees, as a description publishes it."""
    seed = {"shoulder_pan": -61.0, "shoulder_lift": -98.5, "elbow": -101.2,
            "wrist_1": -70.3, "wrist_2": 90.0, "wrist_3": -26.0}  # fmt: skip
    return NamedPose(
        name="over_bin", group="arm",
        joint_names=tuple(seed), positions=[math.radians(v) for v in seed.values()],
        tolerance=[math.radians(2.0)] * len(seed),
        tool_pose=PoseStamped.from_xyz_rpy(0.10, -0.40, 0.30, math.pi, 0.0, math.radians(-35.0), frame_id="world"),
        tip_link="tcp",
        why="the claw 180 mm over the bin's drop, plumb, captured on the arm",
        description="Parked over the drop bin, clear of the camera's view; the arm releases here after each pick.",
    )  # fmt: skip


register(TypeView(
    type=NamedPose,
    summary="A card with the saved pose's joints in degrees, tolerance and reason; a Cartesian pose also draws its tool frame.",
    sample=sample_named_pose,
    views=named_pose_views,
    tags=("3d", "text"),
))  # fmt: skip


# ── AllowedCollisionMatrix ───────────────────────────────────────────────────

# Class ids of the matrix picture; colours by meaning: the pairs a checker tests stay neutral, each
# reason a pair is skipped gets its own hue, "Always" (a modelling artefact) is the one warm warning.
CHECKED, SELF, NO_REASON = 0, 1, 2
REASON_CLASSES = {"Adjacent": 3, "Default": 4, "Always": 5, "Never": 6, "User": 7}
CLASSES = [
    (CHECKED, "checked (contact is tested)", (226, 229, 234)),
    (SELF, "same body", (150, 155, 165)),
    (NO_REASON, "skipped, no reason given (User assumed)", (205, 170, 225)),
    (REASON_CLASSES["Adjacent"], "skipped: Adjacent (joined by a joint)", (66, 133, 244)),
    (REASON_CLASSES["Default"], "skipped: Default (touching at the default pose)", (0, 150, 136)),
    (REASON_CLASSES["Always"], "skipped: Always (touching in every sample)", (240, 150, 0)),
    (REASON_CLASSES["Never"], "skipped: Never (touching in no sample)", (76, 175, 80)),
    (REASON_CLASSES["User"], "skipped: User (allowed by hand)", (156, 39, 176)),
]
NO_REASON_TEXT = "no reason given (User assumed)"
COLOUR_WORDS = {"Adjacent": "blue", "Default": "teal", "Always": "orange", "Never": "green", "User": "purple",
                NO_REASON_TEXT: "lilac"}  # fmt: skip
CELL = 10  # pixels per cell in the picture
MARGIN = 0.6  # the name column's width, in matrix widths: Rerun centres a label on its point, so the names
# need room left of the matrix, and the view fits what is drawn (labels are not counted)


def acm_views(name: str) -> list[rrb.View]:
    return [
        rrb.Spatial2DView(origin=f"/image/{part(name)}", name=name),
        rrb.TextDocumentView(origin=f"/notes/{part(name)}", name=f"{name} pairs"),
    ]


def acm_classes(acm: AllowedCollisionMatrix) -> np.ndarray:
    """The (n, n) class id of every cell."""
    n = len(acm.entry_names)
    classes = np.where(acm.entry_values, NO_REASON, CHECKED).astype(np.uint8)
    classes[np.arange(n), np.arange(n)] = SELF
    index = {name: k for k, name in enumerate(acm.entry_names)}
    for a, b, reason in acm.reasons:
        classes[index[a], index[b]] = classes[index[b], index[a]] = REASON_CLASSES[reason]
    return classes


def acm_card(acm: AllowedCollisionMatrix) -> str:
    """Markdown: how many pairs are checked and skipped, every skipped pair under its reason, how it was made."""
    n = len(acm.entry_names)
    checked, allowed = acm.checked_pairs(), acm.allowed_pairs()
    pairs = n * (n - 1) // 2
    lines = [f"### {n} {'body' if n == 1 else 'bodies'} · {pairs} {'pair' if pairs == 1 else 'pairs'}", "",
             f"**{len(checked)} checked**, {len(allowed)} skipped.", ""]  # fmt: skip
    given = {(a, b): reason for a, b, reason in acm.reasons}
    groups: dict[str, list[str]] = {}
    for a, b in allowed:
        groups.setdefault(given.get((a, b), NO_REASON_TEXT), []).append(f"{a} – {b}")
    lines += ["In the picture: light grey = checked, grey = the same body"
              + ("; skipped pairs by reason:" if allowed else "."), ""]
    for reason in COLOUR_WORDS:
        if reason in groups:
            lines.append(f"- **{reason}**, {COLOUR_WORDS[reason]} ({len(groups[reason])}): " + ", ".join(groups[reason]))
    lines.append("")
    if acm.default_entry_names:
        defaults = ", ".join(f"{k} {'skipped' if v else 'checked'}"
                             for k, v in zip(acm.default_entry_names, acm.default_entry_values, strict=True))
        lines += [f"*Bodies not in the matrix:* {defaults}.", ""]
    if acm.why:
        lines += [f"*How it was made:* {acm.why}", ""]
    lines.append(f"digest `{acm.digest()}`")
    return "\n".join(lines)


@log.register
def _(acm: AllowedCollisionMatrix, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    key = part(name or "AllowedCollisionMatrix")
    base = f"image/{key}"
    names = acm.entry_names
    n = len(names)
    recording.log(f"notes/{key}", rr.TextDocument(acm_card(acm), media_type="text/markdown"), static=True)
    recording.log(base, rr.AnnotationContext(CLASSES), static=True)
    if n == 0:
        return
    picture = np.kron(acm_classes(acm), np.ones((CELL, CELL), dtype=np.uint8))
    recording.log(f"{base}/cells", rr.SegmentationImage(picture), static=True)
    edges = np.arange(n + 1) * CELL
    lines = [[[e, 0], [e, n * CELL]] for e in edges] + [[[0, e], [n * CELL, e]] for e in edges]
    recording.log(f"{base}/grid", rr.LineStrips2D(lines, radii=0.25, colors=[(255, 255, 255)], draw_order=20.0),
                  static=True)  # fmt: skip
    centres = (np.arange(n) + 0.5) * CELL
    # Rows carry the names in a column left of the matrix; columns carry the same numbers on top, so long
    # link names never collide over narrow columns. Faint rules run each row out to its name.
    margin = MARGIN * n * CELL
    rules = [[[-margin, e], [0, e]] for e in edges]
    recording.log(f"{base}/rules", rr.LineStrips2D(rules, radii=0.15, colors=[(215, 218, 224)]), static=True)
    # WHY lifted 0.3 cell: Rerun 0.40 draws a 2D point's label just below the point (measured in the still).
    lifted = centres - 0.3 * CELL
    recording.log(f"{base}/rows", rr.Points2D(np.column_stack([np.full(n, -0.5 * margin), lifted]), radii=0.01,
                                              labels=[f"{k + 1} {name}" for k, name in enumerate(names)],
                                              colors=[(90, 95, 105)], draw_order=30.0), static=True)  # fmt: skip
    recording.log(f"{base}/columns", rr.Points2D(np.column_stack([centres, np.full(n, -0.8 * CELL)]), radii=0.01,
                                                 labels=[str(k + 1) for k in range(n)],
                                                 colors=[(90, 95, 105)], draw_order=30.0), static=True)  # fmt: skip


def sample_acm() -> AllowedCollisionMatrix:
    """A UR5e with a claw over a belt: chain neighbours Adjacent, far links Never, the claw allowed to skim the belt."""
    arm = ("base_link", "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link",
           "wrist_3_link", "claw")  # fmt: skip
    adjacent = [(a, b, "Adjacent") for a, b in zip(arm, arm[1:], strict=False)]
    never = [("base_link", "upper_arm_link", "Never"), ("base_link", "wrist_1_link", "Never"),
             ("shoulder_link", "forearm_link", "Never"), ("shoulder_link", "wrist_1_link", "Never"),
             ("shoulder_link", "wrist_2_link", "Never"), ("wrist_1_link", "wrist_3_link", "Never"),
             ("shoulder_link", "conveyor", "Never")]  # fmt: skip
    scene = [("base_link", "table", "Default"), ("table", "conveyor", "Always"), ("claw", "conveyor", "User"),
             ("base_link", "conveyor")]  # fmt: skip
    return AllowedCollisionMatrix.from_pairs(
        (*arm, "table", "conveyor"), adjacent + never + scene,
        why="robot_collision, 10 000 sampled poses, seed 0",
    )


register(TypeView(
    type=AllowedCollisionMatrix,
    summary="The matrix as a picture, each skipped pair coloured by its reason (hover a cell for it), and a card listing the pairs.",
    sample=sample_acm,
    views=acm_views,
    tags=("2d", "text"),
))  # fmt: skip


# ── CollisionElement / CollisionScene ────────────────────────────────────────

# (motion, compliance) -> colour: robot_msgs.viz.viser.ELEMENT_RGB (see the module docstring).
ELEMENT_RGB = {
    ("fixed", "rigid"): (100, 116, 139),
    ("fixed", "deformable"): (20, 184, 166),
    ("moving", "rigid"): (99, 102, 241),
    ("moving", "deformable"): (236, 72, 153),
}
ELEMENT_WORDS = {("fixed", "rigid"): "slate", ("fixed", "deformable"): "teal", ("moving", "rigid"): "indigo",
                 ("moving", "deformable"): "pink"}  # fmt: skip
PLANE_ALPHA = 70  # a plane patch is translucent, so what stands on it stays visible
PLANE_HALF = 0.5  # m: half-width of a lone element's plane patch (a scene fits it to its other elements)


def element_text(element: CollisionElement) -> str:
    """Name, source and tags in plain words; the margin and a non-default contact policy only when set."""
    name = element.label or element.id
    text = f"{name} · {element.source}" + (f" ({element.link})" if element.link else "")
    text += f" · {element.motion}, {element.compliance}"
    if element.margin:
        text += f" · margin {mm(element.margin)}"
    if element.contact != "avoid":
        text += f" · contact {element.contact}"
    return text


def plane_patch(plane: Plane, place: np.ndarray, half: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A square of `plane` (in the element frame `place`, 4×4), `half` wide, around its point nearest the origin:
    corners in the scene frame, its centre and its unit normal there."""
    normal = np.array(plane.coef[:3], dtype=float)
    length = float(np.linalg.norm(normal))
    normal /= length
    centre = -float(plane.coef[3]) / length * normal
    u = np.cross(normal, [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    corners = np.array([centre + a * half * u + b * half * v for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    rotation, translation = place[:3, :3], place[:3, 3]
    return corners @ rotation.T + translation, rotation @ centre + translation, rotation @ normal


def solid_mesh(shape: SolidPrimitive | Mesh) -> Mesh:
    return shape.to_mesh(48) if isinstance(shape, SolidPrimitive) else shape


def inflated_outline(shape: SolidPrimitive | Mesh, margin: float) -> np.ndarray | None:
    """The 12 edges of the shape's bounding box grown by `margin`, in the shape frame (None for a bare margin)."""
    if margin <= 0.0:
        return None
    vertices = np.asarray(solid_mesh(shape).vertices)
    low, high = vertices.min(axis=0) - margin, vertices.max(axis=0) + margin
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    return np.array([[corners[a], corners[b]] for a, b in edges])


def scene_extent(elements) -> tuple[np.ndarray, np.ndarray] | None:
    """The scene-frame bounds of every solid shape (planes excluded: they are infinite)."""
    points = []
    for element in elements:
        for shape, place in element.shapes():
            if not isinstance(shape, Plane):
                points.append(np.asarray(solid_mesh(shape).vertices) @ place[:3, :3].T + place[:3, 3])
    if not points:
        return None
    stacked = np.vstack(points)
    return stacked.min(axis=0), stacked.max(axis=0)


def draw_element(recording: rr.RecordingStream, base: str, element: CollisionElement, plane_half: float) -> None:
    rgb = ELEMENT_RGB[(element.motion, element.compliance)]
    top = None
    for k, (shape, place) in enumerate(element.shapes()):
        path = f"{base}/shape_{k}"
        if isinstance(shape, Plane):
            corners, centre, normal = plane_patch(shape, place, plane_half)
            recording.log(path, rr.Mesh3D(vertex_positions=corners, triangle_indices=[[0, 1, 2], [0, 2, 3]],
                                          albedo_factor=(*rgb, PLANE_ALPHA),
                                          face_rendering=rr.components.MeshFaceRendering.DoubleSided),
                          static=True)  # fmt: skip
            recording.log(f"{path}/edge", rr.LineStrips3D([np.vstack([corners, corners[:1]])], radii=-1.0,
                                                          colors=[rgb]), static=True)  # fmt: skip
            # the free side: a half-space is solid opposite its normal
            recording.log(f"{path}/free_side", rr.Arrows3D(origins=[centre], vectors=[0.15 * plane_half * normal],
                                                           radii=0.003, colors=[rgb]), static=True)  # fmt: skip
            anchor = corners[2]
        else:
            mesh = solid_mesh(shape)
            vertices = np.asarray(mesh.vertices) @ place[:3, :3].T + place[:3, 3]
            recording.log(path, rr.Mesh3D(**shaded(vertices, np.asarray(mesh.triangles)), albedo_factor=(*rgb, 255)),
                          static=True)  # fmt: skip
            outline = inflated_outline(shape, element.margin)
            if outline is not None:
                world = outline @ place[:3, :3].T + place[:3, 3]
                recording.log(f"{path}/margin", rr.LineStrips3D(world, radii=-0.75, colors=[MUTED]), static=True)
            peak = vertices[np.argmax(vertices[:, 2])]
            anchor = peak + [0.0, 0.0, 0.02]
        top = anchor if top is None or anchor[2] > top[2] else top
    if top is None:  # an element with no shapes: its frame alone, so it is still there to see
        top = element.pose.position.as_array()
        recording.log(f"{base}/frame", rr.Transform3D(translation=top, quaternion=xyzw(element.pose)),
                      rr.TransformAxes3D(0.05), static=True)  # fmt: skip
    label(recording, f"{base}/label", top, element_text(element), rgb)


@log.register
def _(element: CollisionElement, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    extent = scene_extent([element])
    half = PLANE_HALF if extent is None else max(PLANE_HALF, float(np.max(extent[1] - extent[0])))
    draw_element(recording, f"world/{part(name or element.id)}", element, half)
    origin_triad(recording, 0.1)


def scene_card(scene: CollisionScene) -> str:
    lines = [f"### {scene.name or 'collision scene'}", ""]
    if scene.why:
        lines += [scene.why, ""]
    lines += [f"In frame `{scene.header.frame_id}` · {len(scene.elements)} elements", ""]
    if not scene.elements:
        lines += ["Empty: nothing to avoid.", ""]
    for source in ("scene", "object", "self"):
        members = [e for e in scene.elements if e.source == source]
        if members:
            lines.append(f"- **{source}**: " + ", ".join(f"{e.label or e.id} ({e.motion}, {e.compliance})"
                                                         for e in members))  # fmt: skip
    lines += ["", "Colour: " + ", ".join(f"{ELEMENT_WORDS[k]} = {k[0]} {k[1]}" for k in ELEMENT_RGB) + "."]
    state: JointState | None = scene.robot_state
    if state is not None and state.position is not None and len(state.name):
        joints = ", ".join(f"{n} {float(p) * DEG:.4g}°" for n, p in zip(state.name, state.position, strict=False))
        lines += ["", f"Robot posed at: {joints}"]
    if scene.acm is not None:
        given = {(a, b): reason for a, b, reason in scene.acm.reasons}
        skipped = [f"{a} – {b}" + (f" ({given[(a, b)]})" if (a, b) in given else "") for a, b in scene.acm.allowed_pairs()]
        lines += ["", f"Not checked ({len(skipped)}): " + (", ".join(skipped) if skipped else "none")]
    return "\n".join(lines)


@log.register
def _(scene: CollisionScene, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    key = part(name or scene.name or "CollisionScene")
    extent = scene_extent(scene.elements)
    half = PLANE_HALF if extent is None else 0.75 * float(np.max(extent[1] - extent[0]))
    for element in scene.elements:
        draw_element(recording, f"world/{key}/{part(element.id)}", element, half)
    if not scene.elements:
        label(recording, f"world/{key}/empty", [0.0, 0.0, 0.0], "empty collision scene")
    origin_triad(recording, 0.1)
    recording.log(f"notes/{key}", rr.TextDocument(scene_card(scene), media_type="text/markdown"), static=True)


def scene_views(name: str) -> list[rrb.View]:
    return [rrb.Spatial3DView(origin="/world", name="world"),
            rrb.TextDocumentView(origin=f"/notes/{part(name)}", name=f"{name} elements")]  # fmt: skip


def sample_element() -> CollisionElement:
    """A glove the detector boxed on the belt: moving, deformable, inflated 10 mm."""
    return CollisionElement(
        id="glove", source="object", pose=Pose.from_xyz_rpy(0.55, 0.05, 0.015, 0.0, 0.0, 0.4),
        primitives=(SolidPrimitive.box(0.20, 0.10, 0.03),), primitive_poses=(Pose.identity(),),
        motion="moving", compliance="deformable", margin=0.01, label="glove", why="detector box, 5 mm noise",
    )  # fmt: skip


def sample_scene() -> CollisionScene:
    """BAM's cell, v1: a floor, the conveyor, a bin, two items on the belt and the robot's forearm at q."""
    def box(element_id, x, y, z, size, **tags):
        return CollisionElement(id=element_id, source=tags.pop("source", "scene"), pose=Pose.from_xyz_rpy(x, y, z, 0, 0, 0),
                                primitives=(SolidPrimitive.box(*size),), primitive_poses=(Pose.identity(),), **tags)  # fmt: skip

    return CollisionScene(
        name="conveyor cell",
        elements=(
            CollisionElement(id="floor", source="scene", planes=(Plane.from_point_normal((0, 0, 0), (0, 0, 1)),),
                             plane_poses=(Pose.identity(),), label="concrete floor"),
            box("conveyor", 0.6, 0.0, 0.35, (0.5, 1.6, 0.7), margin=0.005),
            box("bin", 0.0, -0.6, 0.25, (0.4, 0.3, 0.5), label="sorting bin"),
            CollisionElement(id="bottle", source="object", pose=Pose.from_xyz_rpy(0.55, 0.2, 0.735, math.pi / 2, 0, 0.3),
                             primitives=(SolidPrimitive.cylinder(0.24, 0.035),), primitive_poses=(Pose.identity(),),
                             motion="moving", compliance="deformable", margin=0.01, label="bottle"),
            box("glove", 0.62, -0.15, 0.715, (0.2, 0.1, 0.03), source="object", motion="moving",
                compliance="deformable", margin=0.01, label="glove"),
            CollisionElement(id="forearm_link", source="self", link="forearm_link",
                             pose=Pose.from_xyz_rpy(0.15, 0.0, 1.05, 0, 1.2, 0),
                             primitives=(SolidPrimitive.cylinder(0.39, 0.045),), primitive_poses=(Pose.identity(),),
                             motion="moving", label="forearm"),
        ),
        robot_state=JointState(name=("shoulder", "elbow"), position=[0.1, -1.2]),
        acm=AllowedCollisionMatrix.from_pairs(("floor", "conveyor", "bin", "forearm_link"),
                                              [("floor", "conveyor", "Adjacent"), ("floor", "bin", "Adjacent")]),
        why="v1: the static cell, the detected items, the robot at q",
    )  # fmt: skip


register(TypeView(
    type=CollisionElement,
    summary="The element's shapes, shaded in its motion/compliance colour, its margin as a faint outline; source and tags in the label.",
    sample=sample_element,
    views=world,
    tags=("3d",),
))  # fmt: skip

register(TypeView(
    type=CollisionScene,
    summary="Every collision element coloured by motion and compliance, labelled, beside a card of the elements, the robot state in degrees and the ACM.",
    sample=sample_scene,
    views=scene_views,
    tags=("3d", "text"),
))  # fmt: skip
