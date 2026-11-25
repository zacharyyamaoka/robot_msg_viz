"""Stock views of robot_msgs.geometry types (docs/CONTRACT.md: one TypeView per type, registered on import).

One visual language for the whole domain:

- **A frame** (Pose, PoseStamped, Transform, TransformStamped): robot_msgs' red/green/blue triad (x, y, z), its
  name beside it, and a thin grey line back to its parent frame at the world origin. A stamped message also
  draws its parent frame there, named after ``header.frame_id``.
- **A rotation** (Quaternion): a faint triad before it, the bold triad after it, the arc each axis tip sweeps,
  and the axis it turns about, labelled with the angle in degrees.
- **A position or vector** (Point, Vector3, Pose2D): a marker or arrow with its value in the label, and the
  lines that make its place in 3D readable (a drop to the floor, the x/y/z steps, the heading arc).
- **A pair of rates or loads** (Twist, Accel, Wrench): two arrows at the frame they are expressed in (orange
  for the linear part or force, purple for the angular part or torque, with a spin ring), and one plot per
  quantity over time, x/y/z in red/green/blue, angular rates in degrees.
- **A mass** (Inertia): the equivalent inertia ellipsoid at the centre of mass, labelled with the mass.

Every spatial type also logs a short world-origin triad once per recording (`_common.origin_triad`), so a lone
message has a reference.

WHY every logger writes every one of its entities on every call, empty when it does not apply (no line home for a
frame at the origin, no axis for a zero rotation): in a stream the viewer shows each entity's latest value, so an
entity skipped this time would keep showing the previous message's drawing.
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Callable
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from robot_msgs.geometry import (
    Accel,
    Inertia,
    Point,
    Pose,
    Pose2D,
    PoseStamped,
    Quaternion,
    Transform,
    TransformStamped,
    Twist,
    Vector3,
    Wrench,
)
from robot_msgs.geometry import rotations as rot
from robot_msgs.std import Header

from robot_msg_viz._engine import TIMELINE, log, part
from robot_msg_viz.registry import TypeView, register
from robot_msg_viz.views._common import DEG, origin_triad, rng

AXIS_LENGTH = 0.1  # metres: robot_msgs' stock triad, so our frames match msg.show()
AXES = ((220, 50, 60), (40, 160, 70), (50, 105, 230))  # x, y, z: red, green, blue
FAINT_AXES = ((240, 170, 175), (165, 215, 175), (170, 190, 240))  # the same, washed out: "before"
GREY = (130, 135, 145)
INK = (70, 75, 90)  # label anchors, a plain vector, the rotation axis: dark and neutral
LINEAR = (235, 130, 30)  # orange: linear velocity / acceleration, force
ANGULAR = (150, 80, 210)  # purple: angular velocity / acceleration, torque (never blue: that is z)
MASS = (95, 140, 175, 70)  # the inertia ellipsoid: a calm translucent slate
THIN = rr.Radius.ui_points(1.0)  # guide lines: one screen point wide at any zoom
ARROW = rr.Radius.ui_points(2.5)  # arrows: a fixed screen width, so a short one is not all arrowhead

_base = log.dispatch(object)  # robot_msgs' stock logger, behind our overrides
_REFERENCED: weakref.WeakSet[Any] = weakref.WeakSet()
_STYLED: weakref.WeakKeyDictionary[Any, set[str]] = weakref.WeakKeyDictionary()


def world(name: str = "") -> list[rrb.View]:
    """The shared 3D scene every spatial type draws into (robot_msgs logs spatial data under /world)."""
    return [rrb.Spatial3DView(origin="/world", name="world")]


def scene(target: tuple[float, float, float], offset: tuple[float, float, float]) -> Callable[[str], list[rrb.View]]:
    """The world view with the eye at ``target + offset`` looking at ``target`` (z up).

    WHY a set eye: Rerun's automatic one framed these small scenes badly (measured 2026-11-24: it cut off the top
    of a 0.43 m arrow and of a 0.62 m transform, and zoomed onto a near-zero twist). Each type's eye is set for robot
    scale (a cell around its base frame, metres); double-click the view to go back to Rerun's own framing.
    """
    position = tuple(t + o for t, o in zip(target, offset, strict=True))

    def views(name: str) -> list[rrb.View]:
        eye = rrb.EyeControls3D(position=position, look_target=target, eye_up=(0.0, 0.0, 1.0))
        return [rrb.Spatial3DView(origin="/world", name="world", eye_controls=eye)]

    return views


# ── helpers ──────────────────────────────────────────────────────────────────


def _inner(recording: Any) -> Any:
    """The real RecordingStream behind `_engine.Timed` (which puts every log on the timeline)."""
    return getattr(recording, "_inner", recording)


def reference(recording: Any) -> None:
    """The world-origin triad, once per recording and static, so it shows at every time.

    WHY static on the inner stream: under `Timed` (a stream) every log lands on the timeline, so logging the
    reference per message would add a row each time and hide it before the first one.
    """
    inner = _inner(recording)
    if inner in _REFERENCED:
        return
    _REFERENCED.add(inner)
    origin_triad(inner)


def _num(value: float, digits: int = 3) -> str:
    text = f"{value:.{digits}g}"
    return "0" if text in ("-0", "0") else text


def _xyz(values: Any, digits: int = 3) -> str:
    return "(" + ", ".join(_num(float(v), digits) for v in values) + ")"


def _metres(values: Any) -> str:
    return "(" + ", ".join(_m(float(v)) for v in values) + ") m"


def _m(value: float) -> str:
    """Metres to the millimetre, trailing zeros dropped: 1.2, 12.345, 0.007."""
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "0", "") else text


def _label(recording: Any, path: str, text: str, at: Any = (0.0, 0.0, 0.0), colour: Any = INK) -> None:
    recording.log(
        path,
        rr.Points3D([list(at)], radii=rr.Radius.ui_points(3.0), colors=[colour], labels=[text], show_labels=True),
        static=True,
    )


def _unit(q: Any) -> np.ndarray:
    return rot.quat_normalize(np.asarray(list(q), dtype=np.float64))


def _home(recording: Any, path: str, translation: np.ndarray, unit_xyzw: np.ndarray) -> None:
    """A thin grey line from the frame back to its parent's origin, logged in the frame's own coordinates."""
    back = rot.quat_rotate(rot.quat_conjugate(unit_xyzw), -translation)
    strips = [[[0.0, 0.0, 0.0], back.tolist()]] if np.linalg.norm(translation) > 1e-9 else []
    recording.log(f"{path}/link", rr.LineStrips3D(strips, radii=THIN, colors=[GREY]), static=True)


def _parent(recording: Any, path: str, frame_id: str, translation: np.ndarray, unit_xyzw: np.ndarray) -> None:
    """The parent frame at the world origin (a triad and its name), placed by the inverse transform."""
    inverse = rot.quat_conjugate(unit_xyzw)
    recording.log(
        f"{path}/parent",
        rr.Transform3D(translation=rot.quat_rotate(inverse, -translation).tolist(), quaternion=inverse.tolist()),
        rr.TransformAxes3D(AXIS_LENGTH),
        static=True,
    )
    _label(recording, f"{path}/parent/name", frame_id or "parent")


def _frame(recording: Any, path: str, translation: Any, rotation: Any, label: str, parent: str | None = None) -> None:
    """The extras on a frame robot_msgs' stock logger already drew at `path`: its name, the line home, its parent."""
    t = np.asarray(list(translation), dtype=np.float64)
    unit = _unit(rotation)
    _label(recording, f"{path}/name", label)
    _home(recording, path, t, unit)
    if parent is not None:
        _parent(recording, path, parent, t, unit)
    reference(recording)


# ── frames ───────────────────────────────────────────────────────────────────


@log.register
def _(pose: Pose, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    name = name or "Pose"
    _base(pose, name, recording=recording)
    _frame(recording, f"world/{part(name)}", pose.position, pose.orientation, name)


@log.register
def _(stamped: PoseStamped, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    name = name or stamped.header.frame_id or "PoseStamped"  # robot_msgs' own default
    pose = stamped.pose
    _base(pose, name, recording=recording)
    _frame(recording, f"world/{part(name)}", pose.position, pose.orientation, name, parent=stamped.header.frame_id)


@log.register
def _(transform: Transform, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    name = name or "Transform"
    _base(transform, name, recording=recording)
    _frame(recording, f"world/{part(name)}", transform.translation, transform.rotation, name)


@log.register
def _(stamped: TransformStamped, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    name = name or stamped.child_frame_id or "TransformStamped"  # robot_msgs' own default
    tf = stamped.transform
    _base(tf, name, recording=recording)
    child = stamped.child_frame_id or name
    _frame(recording, f"world/{part(name)}", tf.translation, tf.rotation, child, parent=stamped.header.frame_id)


# ── a rotation on its own ────────────────────────────────────────────────────


def _arc(axis: np.ndarray, start: np.ndarray, angle: float, steps: int = 32) -> np.ndarray:
    """`start` turned about unit `axis` from 0 to `angle` (Rodrigues), as a polyline."""
    out = []
    for a in np.linspace(0.0, angle, steps):
        c, s = math.cos(a), math.sin(a)
        out.append(start * c + np.cross(axis, start) * s + axis * np.dot(axis, start) * (1 - c))
    return np.array(out)


@log.register
def _(q: Quaternion, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    name = name or "Quaternion"
    path = f"world/{part(name)}"
    _base(q, name, recording=recording)  # the bold triad, turned by q (robot_msgs' stock frame)
    unit = _unit(q)
    back = rot.quat_conjugate(unit)  # children of `path` live in the turned frame: map world vectors into it

    def local(v: Any) -> list[float]:
        return rot.quat_rotate(back, np.asarray(v, dtype=np.float64)).tolist()

    rotvec = rot.quat_to_rotvec(unit)
    angle = float(np.linalg.norm(rotvec))
    before = np.eye(3) * AXIS_LENGTH
    recording.log(
        f"{path}/after",
        rr.Arrows3D(vectors=before.tolist(), colors=list(AXES), radii=rr.Radius.ui_points(2.0)),
        static=True,
    )
    recording.log(
        f"{path}/before",
        rr.Arrows3D(vectors=[local(v) for v in before], colors=list(FAINT_AXES), radii=rr.Radius.ui_points(1.5)),
        static=True,
    )
    roll, pitch, yaw = rot.quat_to_euler(unit).tolist()
    norm = float(np.linalg.norm(q.as_array()))
    note = "" if abs(norm - 1.0) < 1e-6 else f"\n|q| = {_num(norm, 4)}, drawn normalised"
    euler = f"roll {roll * DEG:.1f}°, pitch {pitch * DEG:.1f}°, yaw {yaw * DEG:.1f}°"
    turned = angle > 1e-9
    axis = rotvec / angle if turned else np.zeros(3)
    tip = axis * AXIS_LENGTH * 1.5  # the axis is unmoved by its own rotation: local == world
    sweeps = [[local(p) for p in _arc(axis, v, angle)] for v in before] if turned else []
    recording.log(f"{path}/sweep", rr.LineStrips3D(sweeps, radii=THIN, colors=list(AXES)), static=True)
    recording.log(
        f"{path}/axis",
        rr.Arrows3D(vectors=[tip.tolist()] if turned else np.zeros((0, 3)), colors=[INK], radii=ARROW),
        static=True,
    )
    text = f"{name}: {angle * DEG:.1f}° about this axis\n{euler}" if turned else f"{name}: no rotation"
    _label(recording, f"{path}/label", text + note, at=tip)
    reference(recording)


# ── positions and vectors ────────────────────────────────────────────────────


@log.register
def _(point: Point, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    name = name or "Point"
    path = f"world/{part(name)}"
    p = point.as_array()
    recording.log(
        path,
        rr.Points3D([p.tolist()], radii=0.01, colors=[LINEAR], labels=[f"{name} {_metres(p)}"], show_labels=True),
        static=True,
    )
    floor = [p[0], p[1], 0.0]
    strips = [[[0.0, 0.0, 0.0], floor]] if np.hypot(p[0], p[1]) > 1e-9 else []
    if abs(p[2]) > 1e-9:
        strips.append([floor, p.tolist()])
    recording.log(f"{path}/drop", rr.LineStrips3D(strips, radii=THIN, colors=[GREY]), static=True)
    shadow = [floor] if strips else np.zeros((0, 3))
    recording.log(f"{path}/floor", rr.Points3D(shadow, radii=rr.Radius.ui_points(3.0), colors=[GREY]), static=True)
    reference(recording)


@log.register
def _(vector: Vector3, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    name = name or "Vector3"
    path = f"world/{part(name)}"
    v = vector.as_array()
    recording.log(
        path,
        rr.Arrows3D(
            vectors=[v.tolist()],
            colors=[INK],
            radii=ARROW,
            labels=[f"{name} {_xyz(v)}, length {_num(float(np.linalg.norm(v)))}"],
            show_labels=True,
        ),
        static=True,
    )
    corners = [np.zeros(3), np.array([v[0], 0, 0]), np.array([v[0], v[1], 0]), v]
    steps = [[corners[i].tolist(), corners[i + 1].tolist()] for i in range(3)]
    recording.log(f"{path}/steps", rr.LineStrips3D(steps, radii=THIN, colors=list(AXES)), static=True)
    reference(recording)


@log.register
def _(pose: Pose2D, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    """On the floor: a dot at (x, y), a heading arrow, and the arc from world +x round to the heading."""
    name = name or "Pose2D"
    path = f"world/{part(name)}"
    theta = float(pose.theta)  # stored unwrapped, shown unwrapped (a 400° heading draws more than a turn)
    recording.log(
        path,
        rr.Transform3D(
            translation=[pose.x, pose.y, 0.0], rotation_axis_angle=rr.RotationAxisAngle([0, 0, 1], radians=theta)
        ),
        static=True,
    )
    length = 0.3
    recording.log(f"{path}/heading", rr.Arrows3D(vectors=[[length, 0, 0]], colors=[AXES[0]], radii=ARROW), static=True)
    _label(recording, f"{path}/name", f"{name} ({_m(pose.x)}, {_m(pose.y)}) m, heading {theta * DEG:.1f}°")
    radius = 0.6 * length
    steps = min(1000, max(8, int(abs(theta) / (math.pi / 48)) + 1))
    sweep = [[radius * math.cos(a - theta), radius * math.sin(a - theta), 0.0] for a in np.linspace(0, theta, steps)]
    zero = [[0.0, 0.0, 0.0], [1.3 * radius * math.cos(-theta), 1.3 * radius * math.sin(-theta), 0.0]]
    recording.log(f"{path}/angle", rr.LineStrips3D([zero, sweep], radii=THIN, colors=[GREY, INK]), static=True)
    _home(recording, path, np.array([pose.x, pose.y, 0.0]), _unit([0, 0, math.sin(theta / 2), math.cos(theta / 2)]))
    reference(recording)


# ── pairs: velocity, acceleration, load ──────────────────────────────────────

# Per type, per part: (series field, word on screen, unit on screen, unit in the plot title, metres drawn per SI
# unit, plot multiplier). WHY fixed draw scales, not one per message: a stream would rescale every message and
# the arrows would lie about change. The labels and plots carry the true values; the length only has to read at
# robot scale (1 m/s draws 1 m, 1 rad/s 0.5 m, 1 m/s² or rad/s² 0.1 m, 100 N 1 m, 1 N·m 0.2 m).
_QUANTITIES = {
    Twist: (
        ("linear_velocity", "linear", "m/s", "m/s", 1.0, 1.0),
        ("angular_velocity", "angular", "°/s", "deg/s", 0.5, DEG),
    ),
    Accel: (
        ("linear_acceleration", "linear", "m/s²", "m/s²", 0.1, 1.0),
        ("angular_acceleration", "angular", "°/s²", "deg/s²", 0.1, DEG),
    ),
    Wrench: (
        ("force", "force", "N", "N", 0.01, 1.0),
        ("torque", "torque", "N·m", "N·m", 0.2, 1.0),
    ),
}


def _styled_once(recording: Any, path: str, *style: Any) -> None:
    """A series' name and colour, once per recording and static (the same WHY as `reference`)."""
    seen = _STYLED.setdefault(_inner(recording), set())
    if path in seen:
        return
    seen.add(path)
    _inner(recording).log(path, *style, static=True)


def _spin(drawn: np.ndarray) -> tuple[list[list[float]], list[float], list[float]]:
    """A three-quarter ring round `drawn` by the right-hand rule, and the short arrow that ends it."""
    size = float(np.linalg.norm(drawn))
    axis = drawn / size
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    centre, radius = drawn * 0.6, 0.3 * size
    ring = [centre + radius * (u * math.cos(a) + v * math.sin(a)) for a in np.linspace(0.0, 1.5 * math.pi, 36)]
    tip = ring[-1]
    tangent = (u * -math.sin(1.5 * math.pi) + v * math.cos(1.5 * math.pi)) * radius * 0.35
    return [p.tolist() for p in ring], tip.tolist(), tangent.tolist()


def _pair(msg: Any, parts: tuple[Vector3, Vector3], name: str, recording: Any) -> None:
    path = f"world/{part(name)}"
    default = type(msg).__name__
    snapshot = _inner(recording) is recording  # show()/save(): one message, no stream clock
    resting = []
    for vector, spec, colour in zip(parts, _QUANTITIES[type(msg)], (LINEAR, ANGULAR), strict=True):
        field_name, word, unit, _, scale, mult = spec
        v = vector.as_array()
        size = float(np.linalg.norm(v))
        drawn = v * scale
        moving = float(np.linalg.norm(drawn)) > 1e-6
        text = f"{word} {_num(size * mult)}{'' if unit.startswith('°') else ' '}{unit}"
        if not moving:
            resting.append(text)
        recording.log(
            f"{path}/{word}",
            rr.Arrows3D(vectors=[drawn.tolist()], colors=[colour], radii=ARROW, labels=[text], show_labels=moving),
            static=True,
        )
        if colour is ANGULAR:
            ring, tip, tangent = _spin(drawn) if moving else ([], np.zeros((0, 3)), np.zeros((0, 3)))
            recording.log(
                f"{path}/{word}/spin",
                rr.LineStrips3D([ring] if moving else [], radii=rr.Radius.ui_points(1.5), colors=[colour]),
                rr.Arrows3D(origins=[tip] if moving else tip, vectors=[tangent] if moving else tangent,
                            radii=rr.Radius.ui_points(1.5), colors=[colour]),
                static=True,
            )  # fmt: skip
        for axis_name, value, axis_colour in zip("xyz", v, AXES, strict=True):
            series = f"{field_name}/{axis_name}/{part(name)}"
            legend = axis_name if name == default else f"{name} {axis_name}"
            style = [rr.SeriesLines(names=legend, colors=[axis_colour])]
            if snapshot:  # WHY: a lone sample draws no line, so mark it with a dot
                style.append(rr.SeriesPoints(names=legend, colors=[axis_colour], marker_sizes=[4.0]))
            _styled_once(recording, series, *style)
            if snapshot:
                # WHY t = 0 and not static: a static scalar draws nothing in a time series view, and a plain log lands
                # at three different log times (x, y, z scattered along the axis). One message is one sample, at 0.
                recording.send_columns(
                    series, indexes=[rr.TimeColumn(TIMELINE, duration=[0.0])],
                    columns=rr.Scalars.columns(scalars=[float(value) * mult]),
                )  # fmt: skip
            else:
                recording.log(series, rr.Scalars(float(value) * mult), static=True)  # Timed: at the stream's time
    # Parts at rest share one label at the origin (two zero-length arrows would print theirs on top of each other).
    recording.log(
        f"{path}/at_rest",
        rr.Points3D([[0.0, 0.0, 0.0]] if resting else np.zeros((0, 3)), radii=rr.Radius.ui_points(3.0), colors=[INK],
                    labels=[", ".join(resting) or " "], show_labels=True),
        static=True,
    )  # fmt: skip
    reference(recording)


def _pair_views(message_type: type, target: tuple[float, float, float], offset: tuple[float, float, float]) -> Any:
    first, second = _QUANTITIES[message_type]
    space = scene(target, offset)

    def views(name: str) -> list[rrb.View]:
        return [
            *space(name),
            *(
                rrb.TimeSeriesView(origin=f"/{spec[0]}", name=f"{spec[0].replace('_', ' ')} ({spec[3]})")
                for spec in (first, second)
            ),
        ]

    return views


@log.register
def _(twist: Twist, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    _pair(twist, (twist.linear, twist.angular), name or "Twist", recording)


@log.register
def _(accel: Accel, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    _pair(accel, (accel.linear, accel.angular), name or "Accel", recording)


@log.register
def _(wrench: Wrench, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    _pair(wrench, (wrench.force, wrench.torque), name or "Wrench", recording)


# ── mass ─────────────────────────────────────────────────────────────────────


def equivalent_ellipsoid(inertia: Inertia) -> tuple[np.ndarray, np.ndarray] | None:
    """Semi-axes and orientation (xyzw) of the uniform solid ellipsoid with this mass and tensor, else None.

    For semi-axes a, b, c: I1 = m(b²+c²)/5 and so on, so a² = 5(I2 + I3 - I1) / 2m. A massless body or a tensor
    that breaks the triangle inequality has no such ellipsoid: None, said so in the label, never clipped to one.
    """
    if inertia.m <= 0.0:
        return None
    moments, axes = np.linalg.eigh(inertia.to_matrix())
    squares = 5.0 * (moments.sum() - 2.0 * moments) / (2.0 * inertia.m)
    if np.any(squares <= 0.0):
        return None
    if np.linalg.det(axes) < 0:
        axes[:, 2] = -axes[:, 2]
    return np.sqrt(squares), rot.quat_from_matrix(axes)


@log.register
def _(inertia: Inertia, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    """The equivalent inertia ellipsoid at the centre of mass, the mass in its label (robot_msgs has no logger)."""
    name = name or "Inertia"
    path = f"world/{part(name)}"
    com = inertia.com.as_array()
    shape = equivalent_ellipsoid(inertia)
    text = f"{name}: {_num(inertia.m)} kg"
    if shape is None:
        why = "no mass" if inertia.m <= 0 else "tensor is not physical"
        text += f"\n{why}: no equivalent ellipsoid"
    half_sizes, quaternion = shape if shape is not None else (np.zeros((0, 3)), np.zeros((0, 4)))
    one = shape is not None
    recording.log(
        f"{path}/ellipsoid",
        rr.Ellipsoids3D(
            centers=[com.tolist()] if one else np.zeros((0, 3)),
            half_sizes=[half_sizes.tolist()] if one else half_sizes,
            quaternions=[quaternion.tolist()] if one else quaternion,
            colors=[MASS],
            fill_mode=rr.components.FillMode.TransparentFillMajorWireframe,
            line_radii=THIN,
        ),
        static=True,
    )
    _label(recording, path, text, at=com)
    link = [[[0.0, 0.0, 0.0], com.tolist()]] if np.linalg.norm(com) > 1e-9 else []
    recording.log(f"{path}/link", rr.LineStrips3D(link, radii=THIN, colors=[GREY]), static=True)
    reference(recording)


# ── samples: small, deterministic, shaped like real robot data ───────────────

_STAMP = 1732492800.0  # a fixed stamp, so samples compare equal


def _ease(t: np.ndarray, rise: float) -> np.ndarray:
    """0 → 1 over `rise` seconds (sin², smooth at both ends), then held at 1."""
    return np.sin(0.5 * np.pi * np.clip(t / rise, 0.0, 1.0)) ** 2


def _base_twists() -> list[tuple[float, Twist]]:
    """A mobile base's odometry as it speeds up into a left curve: 0.5 m/s forward, turning at 0.5 rad/s."""
    t = np.linspace(0.0, 2.0, 41)
    ramp = _ease(t, 1.2)
    noise = rng(7).normal(0.0, 1.0, size=(len(t), 2))
    return [
        (float(ti), Twist(Vector3(0.5 * ramp[i] + 0.004 * noise[i, 0], 0.0, 0.0),
                          Vector3(0.0, 0.0, 0.5 * ramp[i] + 0.006 * noise[i, 1])))
        for i, ti in enumerate(t)
    ]  # fmt: skip


def _swing_accels() -> list[tuple[float, Accel]]:
    """A tool 0.5 m out on an arm whose base joint spins up at 1 rad/s², in the base frame.

    Linear = tangential (r·α) + centripetal (r·ω²), so it grows and turns with the arm; angular = α about z.
    """
    radius, alpha = 0.5, 1.0
    t = np.linspace(0.0, 1.5, 31)
    noise = rng(5).normal(0.0, 1.0, size=(len(t), 3))
    out = []
    for i, ti in enumerate(t):
        phi, omega = alpha * ti**2 / 2.0, alpha * ti
        tangential = radius * alpha * np.array([-math.sin(phi), math.cos(phi), 0.0])
        centripetal = radius * omega**2 * np.array([-math.cos(phi), -math.sin(phi), 0.0])
        linear = tangential + centripetal + 0.01 * noise[i]
        out.append((float(ti), Accel(Vector3.from_array(linear), Vector3(0.0, 0.0, alpha + 0.005 * noise[i, 2]))))
    return out


def _press_wrenches() -> list[tuple[float, Wrench]]:
    """A wrist force/torque sensor as the gripper presses into a bin floor: 15 N down within 1 s, then held.

    The contact is a few centimetres off the tool axis, so a torque builds with the force.
    """
    t = np.linspace(0.0, 2.0, 41)
    press = _ease(t, 1.0)
    noise = rng(11).normal(0.0, 1.0, size=(len(t), 6))
    out = []
    for i, ti in enumerate(t):
        force = np.array([1.2, -0.6, -15.0]) * press[i] + 0.15 * noise[i, :3]
        torque = np.array([0.45, 0.6, 0.0]) * press[i] + 0.01 * noise[i, 3:]
        out.append((float(ti), Wrench(Vector3.from_array(force), Vector3.from_array(torque))))
    return out


def _forearm() -> Inertia:
    """A 1.5 kg aluminium tube link, 0.4 m long along x, tilted 30° up and 15° round: a robot forearm's inertial."""
    mass, radius, length = 1.5, 0.04, 0.4
    along = mass * radius**2 / 2.0
    across = mass * (3 * radius**2 + length**2) / 12.0
    flat = Inertia(mass, Vector3(length / 2, 0.0, 0.0), along, 0.0, 0.0, across, 0.0, across)
    return flat.rotated(Quaternion.from_euler(0.0, -math.radians(30), math.radians(15)))


def _last(series: Callable[[], list[tuple[float, Any]]]) -> Callable[[], Any]:
    return lambda: series()[-1][1]


# ── the views ────────────────────────────────────────────────────────────────

# Eyes: look from the front right and above (x to the lower right, y to the upper right, z up), as a robot cell
# is usually photographed.
_CAMERA_TRANSFORM = Transform.from_xyz_rpy(0.05, 0.0, 0.62, 0.0, math.radians(40), 0.0)

register(TypeView(
    type=Vector3,
    summary="An arrow from the origin with its components and length, and its x, y, z steps in red, green, blue.",
    sample=lambda: Vector3(0.25, 0.20, 0.30),
    views=scene((0.12, 0.1, 0.13), (0.36, -0.56, 0.3)),
    tags=("3d",),
))  # fmt: skip

register(TypeView(
    type=Point,
    summary="A marker at the position labelled with its coordinates, with a drop line to the floor and a line home.",
    sample=lambda: Point(0.42, -0.15, 0.30),
    views=scene((0.21, -0.07, 0.13), (0.38, -0.6, 0.3)),
    tags=("3d",),
))  # fmt: skip

register(TypeView(
    type=Quaternion,
    summary="The turn: a faint triad before, the bold one after, the arcs between, the axis and its angle in degrees.",
    sample=lambda: Quaternion.from_euler(math.radians(20), math.radians(35), math.radians(60)),
    views=scene((0.02, 0.01, 0.06), (0.13, -0.18, 0.1)),
    tags=("3d",),
))  # fmt: skip

register(TypeView(
    type=Pose,
    summary="A coordinate frame: red, green, blue axes (x, y, z) at the position, turned by the orientation.",
    sample=lambda: Pose.from_xyz_rpy(0.4, 0.1, 0.3, 0.0, 0.5, 0.8),
    views=scene((0.2, 0.05, 0.15), (0.38, -0.6, 0.3)),
    tags=("3d",),
))  # fmt: skip

register(TypeView(
    type=Pose2D,
    summary="A position on the floor with a heading arrow, the arc from +x to it, and the heading in degrees.",
    sample=lambda: Pose2D(1.2, 0.6, math.radians(35)),
    views=scene((0.64, 0.33, 0.0), (0.22, -0.95, 1.05)),
    tags=("3d",),
))  # fmt: skip

register(TypeView(
    type=PoseStamped,
    summary="A named coordinate frame linked to its parent frame (header.frame_id), drawn and named at the origin.",
    sample=lambda: PoseStamped(Header(_STAMP, "base_link"), Pose.from_xyz_rpy(0.45, -0.12, 0.32, math.pi, 0.0, 0.4)),
    views=scene((0.22, -0.06, 0.15), (0.38, -0.6, 0.3)),
    tags=("3d",),
))  # fmt: skip

register(TypeView(
    type=Transform,
    summary="Where a child frame sits in its parent: the child's axes, and a line back to the parent's origin.",
    sample=lambda: _CAMERA_TRANSFORM,
    views=scene((0.03, 0.0, 0.33), (0.65, -0.7, 0.25)),
    tags=("3d",),
))  # fmt: skip

register(TypeView(
    type=TransformStamped,
    summary="One edge of the transform tree: parent and child frames, each named, joined by a line.",
    sample=lambda: TransformStamped(Header(_STAMP, "base_link"), "camera_link", _CAMERA_TRANSFORM),
    views=scene((0.03, 0.0, 0.33), (0.65, -0.7, 0.25)),
    tags=("3d",),
))  # fmt: skip

register(TypeView(
    type=Twist,
    summary="Linear (orange) and angular (purple, with its spin) velocity arrows, and both over time (angular in deg/s).",
    sample=_last(_base_twists),
    series=_base_twists,
    views=_pair_views(Twist, (0.22, 0.0, 0.1), (0.27, -0.45, 0.22)),
    tags=("3d", "plot", "stream"),
))  # fmt: skip

register(TypeView(
    type=Accel,
    summary="Linear (orange) and angular (purple) acceleration arrows, and both over time; angular in deg/s².",
    sample=_last(_swing_accels),
    series=_swing_accels,
    views=_pair_views(Accel, (-0.04, -0.03, 0.04), (0.14, -0.22, 0.12)),
    tags=("3d", "plot", "stream"),
))  # fmt: skip

register(TypeView(
    type=Wrench,
    summary="Force (orange) and torque (purple, with its twist) arrows at the sensor, and both over time.",
    sample=_last(_press_wrenches),
    series=_press_wrenches,
    views=_pair_views(Wrench, (0.04, 0.04, -0.06), (0.2, -0.3, 0.15)),
    tags=("3d", "plot", "stream"),
))  # fmt: skip

register(TypeView(
    type=Inertia,
    summary="The equivalent inertia ellipsoid (same mass and moments) at the centre of mass, labelled with the mass.",
    sample=_forearm,
    views=scene((0.16, 0.04, 0.09), (0.36, -0.55, 0.26)),
    tags=("3d",),
))  # fmt: skip
