"""Stock views of robot_msgs.trajectory types (docs/CONTRACT.md: one TypeView per type, registered on import).

What each view shows:

- Trajectory: every channel over time in one plot, the legend naming the channel.
- TrajectoryPoint / JointTrajectoryPoint: a table of the sample's values at its instant.
- JointTrajectory: one plot per field (position, velocity, acceleration in degrees), one line per joint in a fixed
  colour per joint, the limits the motion comes near as thin lines, and a table (which colour is which joint, its
  range, peak speed and acceleration against its limits, any violation).
- PoseTrajectory: the path in 3D with frames along it and the frame moving on the timeline, beside its position (mm)
  and roll / pitch / yaw (deg) over time. PoseTrajectoryPoint: that one frame, with its velocity arrow.
- JointLimits: each joint's position range as a bar on one degree scale, beside a table of every bound and cap.
- JointLimitViolation: the record as one sentence card, in degrees.

WHY overrides of robot_msgs' stock logger here (all registered on `robot_msg_viz._engine.log`):
- JointTrajectory: the stock logs radians, its colours follow the entity path (so one joint changes colour from plot
  to plot) and its legend names the run, not the joint; here one plot holds every joint, and the table names the colours.
  WHY degrees for every joint: Zach's rule (angles on screen in degrees) and robot_msgs carries no joint type, so every
  joint is read as revolute, as the stock's ``degrees=True`` and traj_compare's export do; the table says so.
  The URDF dashboard (``robot=``) and every other stock option still go to robot_msgs' logger untouched. The babble
  round in traj_compare/docs/proposals/rerun-joint-trajectory-2026-11-24/ picks the stock JointTrajectory look (the
  URDF arm moving); its pick replaces this view.
- PoseTrajectory: the stock plots raw quaternion components (unreadable) at ``/positions``, where they would share a
  plot with a JointTrajectory's joints; here position is in mm at ``/position_mm`` and orientation is roll / pitch /
  yaw in degrees at ``/orientation_deg``.
- Trajectory: the stock legend names every channel's line after the run, so one plot of channels was unreadable.
- The points, JointLimits and JointLimitViolation have no stock logger (they were drawn as their repr).
"""

from __future__ import annotations

# WHY empty (2026-11-25): this module registers its TypeViews on import; its functions are sample data and helpers,
# not API. The public API is robot_msg_viz's own __all__ (show/save/stream/blueprint), and robot-sdk's scanner reads
# __all__ to decide what a package publishes.
__all__: list[str] = []

import dataclasses
import math
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from robot_msgs.geometry.rotations import quat_from_euler, quat_to_euler, quat_to_matrix
from robot_msgs.std import Header
from robot_msgs.trajectory import (
    JointLimits,
    JointLimitViolation,
    JointTrajectory,
    JointTrajectoryPoint,
    PoseTrajectory,
    PoseTrajectoryPoint,
    Trajectory,
    TrajectoryPoint,
)
from robot_msgs.trajectory.joint_limit_violation import joint_limit_violations
from robot_msgs.viz import rerun as stock

from robot_msg_viz._engine import TIMELINE, log, part
from robot_msg_viz.registry import TypeView, register
from robot_msg_viz.views._common import DEG, origin_triad, rng, world

# ── style ────────────────────────────────────────────────────────────────────────────────────────────────────────────

# WHY these eight, in this order (the dataviz skill's categorical palette, its dark-surface steps: Rerun's canvas is
# dark): a joint keeps its colour in every plot and in the JointLimits chart, and a 9th joint gets Rerun's own colour
# rather than a cycled (repeated) one.
JOINT_COLORS = (
    (0x39, 0x87, 0xE5),  # blue
    (0xD9, 0x59, 0x26),  # orange
    (0x19, 0x9E, 0x70),  # aqua
    (0xC9, 0x85, 0x00),  # yellow
    (0xD5, 0x51, 0x81),  # magenta
    (0x00, 0x83, 0x00),  # green
    (0x90, 0x85, 0xE9),  # violet
    (0xE6, 0x67, 0x67),  # red
)
COLOR_NAMES = ("blue", "orange", "aqua", "yellow", "magenta", "green", "violet", "red")
AXIS_COLORS = ((214, 62, 62), (52, 150, 72), (48, 98, 214))  # x / y / z and roll / pitch / yaw: Rerun's own
LIMIT_GREY = (150, 150, 150)
ACCENT = (240, 140, 0)  # robot_msgs' tool-path orange
START = (90, 140, 230)
END = (220, 120, 20)

# (field, singular, unit on screen, scale from SI to that unit, the JointLimits bounds that apply: (low, high, magnitude))
JOINT_FIELDS = (
    ("positions", "position", "deg", DEG, ("min_position", "max_position", False)),
    ("velocities", "velocity", "deg/s", DEG, ("max_velocity", "max_velocity", True)),
    ("accelerations", "acceleration", "deg/s²", DEG, ("max_acceleration", "max_acceleration", True)),
)
EFFORT_UNIT = "N·m"


def joint_color(index: int) -> tuple[int, int, int] | None:
    return JOINT_COLORS[index] if index < len(JOINT_COLORS) else None


def _faded(color: tuple[int, int, int] | None) -> tuple[int, int, int]:
    """A limit line in its joint's hue, quieter. WHY opaque, not alpha: in 0.40 an alpha line drew darker, not fainter."""
    if color is None:
        return LIMIT_GREY
    return tuple(int(0.55 * c + 0.45 * g) for c, g in zip(color, LIMIT_GREY, strict=True))  # type: ignore[return-value]


def _num(value: float, digits: int = 4) -> str:
    """A value as shown in a table: ``∞`` / ``−∞`` for unbounded, ``—`` for NaN, a true minus sign."""
    if value is None:
        return "—"
    value = float(value)
    if math.isnan(value):
        return "—"
    if math.isinf(value):
        return "∞" if value > 0 else "−∞"
    text = f"{value:.{digits}g}"
    return text.replace("-", "−")


def _legend(label: str, name: str, default: str) -> str:
    """The legend of one line: what it is, and the run's name when the caller gave one (an overlay)."""
    return label if name == default else f"{label} · {name}"


def _series(
    recording: Any, path: str, seconds: np.ndarray, values: np.ndarray, *, label: str, color: Any = None,
    width: float | None = None,
) -> None:  # fmt: skip
    style = rr.SeriesLines(names=label, colors=color, widths=width)
    recording.log(path, style, static=True)
    recording.send_columns(
        path,
        indexes=[rr.TimeColumn(TIMELINE, duration=np.asarray(seconds, dtype=np.float64))],
        columns=rr.Scalars.columns(scalars=np.asarray(values, dtype=np.float64)),
    )


def _stock(msg: Any, name: str | None, recording: Any, **options: Any) -> None:
    """robot_msgs' own logger, forgetting its auto-layout paths as `_engine.log` does (else a long stream grows them)."""
    stock.log(msg, name, recording=recording, **options)
    paths = getattr(stock, "_PATHS", {})
    paths.pop(id(recording), None)
    paths.pop(id(getattr(recording, "_inner", None)), None)


def _seconds(traj: Trajectory) -> np.ndarray:
    t = np.asarray(traj.t, dtype=np.float64)
    return t - t[0] if t.size else t


def _channel_columns(values: np.ndarray) -> list[str]:
    """robot_msgs' column names for a channel (so the paths overlay with its logger and traj_compare)."""
    width = int(np.prod(values.shape[1:], dtype=int)) if values.ndim > 1 else 1
    if width == 3:
        return ["x", "y", "z"]
    return [str(i) for i in range(width)]


def _log_channels(traj: Trajectory, name: str, recording: Any, seconds: np.ndarray) -> None:
    """Every numeric channel at ``/channels/<key>/<column>/<name>``, its line named after the channel."""
    for index, (key, raw) in enumerate((traj.channels or {}).items()):
        values = np.asarray(raw)
        if values.dtype.kind not in "fiub" or values.shape[:1] != seconds.shape:
            continue
        flat = values.reshape(len(seconds), -1).astype(np.float64)
        columns = _channel_columns(values)
        for column, column_name in enumerate(columns):
            label = key if len(columns) == 1 else f"{key}[{column_name}]"
            color = joint_color(index) if len(columns) == 1 else None
            _series(recording, f"channels/{part(key)}/{part(column_name)}/{part(name)}", seconds, flat[:, column],
                    label=_legend(label, name, type(traj).__name__), color=color)  # fmt: skip


# WHY the whole run on the x axis, always: left to itself the viewer opened a 10-minute series at its end on only its
# last ~50 s (measured in the 0.40 web viewer, 2026-11-24)
WHOLE_RUN = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
)


def _plot(origin: str, title: str, *, legend: bool = True) -> rrb.View:
    # WHY bottom right: a run's lines start together at the left (a warm-up, a pose leaving home), where the default
    # bottom-left legend covered them
    legend_box = rrb.PlotLegend(corner=rrb.Corner2D.RightBottom, visible=legend)
    return rrb.TimeSeriesView(origin=origin, name=title, axis_x=WHOLE_RUN, plot_legend=legend_box)


def _notes(recording: Any, name: str, text: str) -> None:
    recording.log(f"notes/{part(name)}", rr.TextDocument(text, media_type="text/markdown"), static=True)


def _notes_view(name: str, title: str | None = None) -> rrb.View:
    return rrb.TextDocumentView(origin=f"/notes/{part(name)}", name=title or name)


# ── Trajectory: channels over time ───────────────────────────────────────────────────────────────────────────────────


@log.register
def _(traj: Trajectory, name: str | None = None, *, recording: Any, **options: Any) -> None:
    # WHY exact type only: a Trajectory subclass this module has no view for keeps robot_msgs' own logger
    if type(traj) is not Trajectory or options:
        _stock(traj, name, recording, **options)
        return
    _log_channels(traj, name or "Trajectory", recording, _seconds(traj))


def motor_temperatures() -> Trajectory:
    """Ten minutes of three motors warming up in a pick cell: a first-order rise, a slow swell with each 90 s batch of
    picks, sensor noise, logged every 2 s to 0.01 °C."""
    t = np.arange(0.0, 600.0 + 1e-9, 2.0)
    noise = rng(7)
    channels = {}
    for key, rise, tau, swell in (("shoulder_motor_C", 29.0, 170.0, 0.8), ("elbow_motor_C", 23.0, 140.0, 0.6),
                                  ("wrist_motor_C", 13.0, 110.0, 0.4)):  # fmt: skip
        warm = 23.5 + rise * (1.0 - np.exp(-t / tau))
        batch = swell * np.sin(2.0 * np.pi * t / 90.0 + 0.4) * (1.0 - np.exp(-t / 60.0))
        channels[key] = np.round(warm + batch + noise.normal(0.0, 0.08, t.size), 2)
    return Trajectory(t=t, channels=channels, header=Header(stamp=1_761_600_000.0))


register(TypeView(
    type=Trajectory,
    summary="Every channel over time in one plot, one line per channel, the legend naming it.",
    sample=motor_temperatures,
    views=lambda name: [_plot("/channels", "channels")],
    tags=("plot",),
))  # fmt: skip


# ── points: the values at one instant ────────────────────────────────────────────────────────────────────────────────


def _value_cell(value: Any) -> str:
    if value is None:
        return "—"
    array = np.asarray(value)
    if array.dtype.kind in "fiu" and array.ndim == 0:
        return _num(float(array), 6)
    if array.dtype.kind in "fiu":
        return "[" + ", ".join(_num(v, 6) for v in array.reshape(-1)) + "]"
    return str(value)


def _point_heading(point: TrajectoryPoint) -> str:
    frame = point.header.frame_id
    return f"**t = {_num(point.t, 6)} s**" + (f" · frame `{frame}`" if frame else "")


@log.register
def _(point: TrajectoryPoint, name: str | None = None, *, recording: Any) -> None:
    """Any point (a generated one too): a row per typed field and channel, values as stored."""
    rows = [_point_heading(point), "", "| channel | value |", "|---|---|"]
    for f in dataclasses.fields(point):
        if f.name in ("t", "channels", "header"):
            continue
        rows.append(f"| {f.name} | {_value_cell(getattr(point, f.name))} |")
    for key, value in (point.channels or {}).items():
        rows.append(f"| {key} | {_value_cell(value)} |")
    _notes(recording, name or type(point).__name__, "\n".join(rows))


register(TypeView(
    type=TrajectoryPoint,
    summary="A table of the sample's channel values at its instant t.",
    sample=lambda: motor_temperatures()[150],
    views=lambda name: [_notes_view(name)],
    tags=("text",),
))  # fmt: skip


@log.register
def _(point: JointTrajectoryPoint, name: str | None = None, *, recording: Any) -> None:
    """One row per joint (the point carries no joint names, so the row is the joint's index), angles in degrees."""
    columns: list[tuple[str, np.ndarray]] = []
    for field_name, singular, unit, scale, _ in JOINT_FIELDS:
        values = getattr(point, field_name, None)
        if values is not None:
            columns.append((f"{singular} ({unit})", np.asarray(values, dtype=np.float64).reshape(-1) * scale))
    if point.effort is not None:
        columns.append((f"effort ({EFFORT_UNIT})", np.asarray(point.effort, dtype=np.float64).reshape(-1)))
    count = max((len(values) for _, values in columns), default=0)
    rows = [_point_heading(point) + f" · {count} joints", "",
            "| joint | " + " | ".join(title for title, _ in columns) + " |",
            "|---|" + "---|" * len(columns)]  # fmt: skip
    for joint in range(count):
        rows.append(f"| {joint} | " + " | ".join(_num(values[joint], 4) for _, values in columns) + " |")
    for key, value in (point.channels or {}).items():
        rows.append(f"\n{key}: {_value_cell(value)}")
    rows.append("\n*Angles in degrees, every joint read as revolute (robot_msgs carries no joint type).*")
    _notes(recording, name or "JointTrajectoryPoint", "\n".join(rows))


# ── JointTrajectory: one plot per field, one line per joint ─────────────────────────────────────────────────────────


def _near(values: np.ndarray, bound: float) -> bool:
    """A limit is drawn when the joint comes within one of its own motion spans of it (or past it).

    WHY (robot_msgs' R3 strip rule, applied per joint): a ±360° limit drawn beside a 60° move flattens every line in a
    plot that holds all joints; the table keeps every limit and its margin, so nothing is hidden."""
    values = values[np.isfinite(values)]
    if values.size == 0 or not np.isfinite(bound):
        return False
    low, high = float(values.min()), float(values.max())
    span = max(high - low, 1e-9)
    return low - span <= bound <= high + span


def _bound(limits: JointLimits | None, attribute: str, scale: float, magnitude: bool, side: str) -> np.ndarray | None:
    values = None if limits is None else getattr(limits, attribute)
    if values is None:
        return None
    values = np.asarray(values, dtype=np.float64) * scale
    return -values if magnitude and side == "lower" else values


def _log_limit_lines(traj: JointTrajectory, recording: Any, seconds: np.ndarray) -> None:
    """Each limit the motion comes near: one line per distinct value, grey when several joints share it."""
    names = [str(n) for n in traj.joint_names]
    span = np.array([seconds[0], seconds[-1]]) if seconds.size else np.zeros(2)
    for field_name, _, unit, scale, (low_name, high_name, magnitude) in JOINT_FIELDS:
        values = getattr(traj, field_name)
        if values is None:
            continue
        shown = np.asarray(values, dtype=np.float64) * scale
        for side, attribute in (("lower", low_name), ("upper", high_name)):
            bounds = _bound(traj.limits, attribute, scale, magnitude, side)
            if bounds is None:
                continue
            groups: dict[float, list[int]] = {}
            for index, value in enumerate(bounds):
                if np.isfinite(value):
                    groups.setdefault(round(float(value), 9), []).append(index)
            for value, members in groups.items():
                if not any(_near(shown[:, i], value) for i in members):
                    continue
                who = "all joints" if len(members) == len(names) else ", ".join(names[i] for i in members)
                color = LIMIT_GREY if len(members) > 1 else _faded(joint_color(members[0]))
                label = f"{side} limit {_num(value)} {unit} ({who})"
                _series(recording, f"{field_name}/{part(names[members[0]])}/{side}_limit", span, np.full(2, value),
                        label=label, color=color, width=1.0)  # fmt: skip


def _joint_summary(traj: JointTrajectory) -> str:
    """The table beside the plots: each joint's range, peaks against its limits, and every violation."""
    names = [str(n) for n in traj.joint_names]
    limits = traj.limits
    duration = float(traj.t[-1] - traj.t[0]) if len(traj) else 0.0
    head = f"**{_num(duration, 4)} s · {len(traj)} samples · {len(names)} joints**"
    if traj.header.frame_id:
        head += f" · frame `{traj.header.frame_id}`"
    lines = [head, ""]
    if limits is not None:
        violations = joint_limit_violations(traj)
        if violations:
            lines.append(f"**{len(violations)} limit violation{'s' if len(violations) != 1 else ''}**")
            lines += [f"- {violation_sentence(v)}" for v in violations]
        else:
            lines.append("Within every stated limit.")
        lines.append("")

    def peak(field_name: str) -> np.ndarray | None:
        values = getattr(traj, field_name)
        return None if values is None else np.nanmax(np.abs(np.asarray(values, dtype=np.float64)), axis=0) * DEG

    def of_limit(value: float, attribute: str, index: int) -> str:
        bound = None if limits is None or getattr(limits, attribute) is None else float(getattr(limits, attribute)[index])
        if bound is None or not np.isfinite(bound) or bound <= 0:
            return _num(value, 3)
        return f"{_num(value, 3)} · {100.0 * value / (bound * DEG):.0f}%"

    q = np.asarray(traj.positions, dtype=np.float64) * DEG
    speed, accel = peak("velocities"), peak("accelerations")
    # WHY a colour column and no plot legends: one plot holds every joint, and a six-joint legend covered a third of
    # each plot; the colour stays the joint's in all three plots, and hovering a line names it
    header = ["joint", "line", "range (deg)"]
    if speed is not None:
        header.append("peak speed (deg/s)")
    if accel is not None:
        header.append("peak accel (deg/s²)")
    positions_bounded = limits is not None and (limits.min_position is not None or limits.max_position is not None)
    if positions_bounded:
        header.append("to limit (deg)")
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for index, joint in enumerate(names):
        cells = [joint, COLOR_NAMES[index] if index < len(COLOR_NAMES) else "Rerun's",
                 f"{_num(np.nanmin(q[:, index]), 3)} … {_num(np.nanmax(q[:, index]), 3)}"]  # fmt: skip
        if speed is not None:
            cells.append(of_limit(speed[index], "max_velocity", index))
        if accel is not None:
            cells.append(of_limit(accel[index], "max_acceleration", index))
        if positions_bounded:
            low = -np.inf if limits.min_position is None else float(limits.min_position[index]) * DEG
            high = np.inf if limits.max_position is None else float(limits.max_position[index]) * DEG
            to_low, to_high = np.nanmin(q[:, index]) - low, high - np.nanmax(q[:, index])
            if not (np.isfinite(to_low) or np.isfinite(to_high)):
                cells.append("none stated")
            elif to_low <= to_high:
                cells.append(f"{_num(to_low, 3)} to {_num(low, 4)}")
            else:
                cells.append(f"{_num(to_high, 3)} to {_num(high, 4)}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("*Angles in degrees, every joint read as revolute (robot_msgs carries no joint type). % = of the "
                 "joint's own limit. A thin line is a limit the joint comes within one of its own motion spans of "
                 "(grey: shared by several joints).*")  # fmt: skip
    return "\n".join(lines)


@log.register
def _(traj: JointTrajectory, name: str | None = None, *, recording: Any, **options: Any) -> None:
    if options:  # robot= (the URDF dashboard), degrees=, color=, prefix=, t0=, label=: robot_msgs' own logger
        _stock(traj, name, recording, **options)
        return
    name = name or "JointTrajectory"
    seconds = _seconds(traj)
    for field_name, _, unit, scale, _ in JOINT_FIELDS:
        values = getattr(traj, field_name)
        if values is None:
            continue
        shown = np.asarray(values, dtype=np.float64) * scale
        for index, joint in enumerate(traj.joint_names):
            _series(recording, f"{field_name}/{part(joint)}/{part(name)}", seconds, shown[:, index],
                    label=_legend(f"{joint} ({unit})", name, "JointTrajectory"), color=joint_color(index), width=2.0)
    if traj.effort is not None:  # logged (robot_msgs' layout, N·m), not in the stock views: a plan rarely carries it
        effort = np.asarray(traj.effort, dtype=np.float64)
        for index, joint in enumerate(traj.joint_names):
            _series(recording, f"effort/{part(joint)}/{part(name)}", seconds, effort[:, index],
                    label=_legend(f"{joint} ({EFFORT_UNIT})", name, "JointTrajectory"), color=joint_color(index))
    _log_limit_lines(traj, recording, seconds)
    _log_channels(traj, name, recording, seconds)
    _notes(recording, name, _joint_summary(traj))


ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
              "wrist_3_joint")  # fmt: skip


def arm_limits() -> JointLimits:
    """A UR5e-class arm as an integrator leaves it in a cell: the shoulder pan fenced to ±60° (a wall), the lift kept
    above the table, ±180°/s on every joint, a planner's 5 rad/s² (286°/s²), the datasheet's torques."""
    return JointLimits(
        joint_names=ARM_JOINTS,
        min_position=np.radians([-60.0, -180.0, -160.0, -360.0, -360.0, -360.0]),
        max_position=np.radians([60.0, 0.0, 160.0, 360.0, 360.0, 360.0]),
        max_velocity=np.full(6, np.pi),
        max_acceleration=np.full(6, 5.0),
        max_effort=[150.0, 150.0, 150.0, 28.0, 28.0, 28.0],
        position_kind=("software",) * 6,
    )


def _min_jerk(s: np.ndarray) -> np.ndarray:
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


# A pick: home → above the bin → down to the grasp → back up → over to the place pose (degrees, holding between moves).
PICK_WAYPOINTS = (
    (0.0, -90.0, 90.0, -90.0, -90.0, 0.0),
    (35.0, -75.0, 100.0, -115.0, -88.0, 20.0),
    (35.0, -62.0, 108.0, -136.0, -92.0, 20.0),
    (35.0, -75.0, 100.0, -115.0, -88.0, 20.0),
    (-40.0, -85.0, 95.0, -100.0, -84.0, -70.0),
)
PICK_DURATIONS = (1.4, 0.7, 0.7, 1.6)  # s per move
PICK_DWELL = 0.25  # s held at each stop (the gripper closing)


def pick_move(rate_hz: float = 100.0) -> JointTrajectory:
    """A 6-joint arm's pick, minimum-jerk between stops, sampled at `rate_hz`, derivatives by `with_derivatives()`."""
    pieces_t, pieces_q, clock = [], [], 0.0
    for start, end, duration in zip(PICK_WAYPOINTS[:-1], PICK_WAYPOINTS[1:], PICK_DURATIONS, strict=True):
        steps = int(round(duration * rate_hz))
        s = np.arange(steps) / steps
        pieces_t.append(clock + s * duration)
        pieces_q.append(np.radians(start) + _min_jerk(s)[:, None] * (np.radians(end) - np.radians(start)))
        clock += duration
        hold = int(round(PICK_DWELL * rate_hz))
        pieces_t.append(clock + np.arange(hold) / rate_hz)
        pieces_q.append(np.repeat(np.radians(end)[None, :], hold, axis=0))
        clock += PICK_DWELL
    t = np.concatenate([*pieces_t, [clock]])
    q = np.concatenate([*pieces_q, np.radians(PICK_WAYPOINTS[-1])[None, :]])
    # WHY stamp 0 (a plan's times are from its start): robot_msgs.compare's split_phases scales its boundary tolerance
    # with the absolute time (1e-9 × 1.76e9 s = 1.76 s on an epoch stamp), so check() of an epoch-stamped run starts
    # the goal phase 1.76 s early and the Outcome sample would miss a real path breach (found 2026-11-24; robot_msgs')
    traj = JointTrajectory(t=t, joint_names=ARM_JOINTS, positions=q, limits=arm_limits(),
                           header=Header(stamp=0.0, frame_id="base_link"))  # fmt: skip
    return traj.with_derivatives()


def _joint_views(name: str) -> list[Any]:
    return [
        _plot("/positions", "joint position (deg)", legend=False),
        _plot("/velocities", "joint velocity (deg/s)", legend=False),
        _plot("/accelerations", "joint acceleration (deg/s²)", legend=False),
        _notes_view(name),
    ]


register(TypeView(
    type=JointTrajectory,
    summary=("Position, velocity and acceleration in degrees, one plot each, one colour per joint; the limits the "
             "motion comes near as thin lines; a table of each joint's range and peaks against its limits."),
    sample=pick_move,
    views=_joint_views,
    tags=("plot",),
))  # fmt: skip

register(TypeView(
    type=JointTrajectoryPoint,
    summary="A table of one sample: each joint's position, velocity and acceleration in degrees, at its instant t.",
    sample=lambda: pick_move()[45],
    views=lambda name: [_notes_view(name)],
    tags=("text",),
))  # fmt: skip


# ── PoseTrajectory: the path in 3D, frames along it, position and orientation over time ─────────────────────────────


def _axes_arrows(positions: np.ndarray, quaternions: np.ndarray, length: float) -> rr.Arrows3D:
    """A red/green/blue (x/y/z) triad at each pose, as one batch of arrows."""
    rotations = quat_to_matrix(quaternions)  # (K, 3, 3), columns = the frame's axes
    origins = np.repeat(positions, 3, axis=0)
    vectors = np.concatenate([rotations[:, :, axis][:, None, :] for axis in range(3)], axis=1).reshape(-1, 3) * length
    colors = np.tile(np.array(AXIS_COLORS, dtype=np.uint8), (len(positions), 1))
    return rr.Arrows3D(origins=origins, vectors=vectors, colors=colors, radii=rr.Radius.ui_points(1.5))


def _frame_length(positions: np.ndarray) -> float:
    """A triad size that reads at the path's own scale: 8 % of its extent, between 1 cm and 10 cm."""
    finite = positions[np.all(np.isfinite(positions), axis=1)]
    extent = float(np.linalg.norm(np.ptp(finite, axis=0))) if len(finite) else 0.0
    return float(np.clip(0.08 * extent, 0.01, 0.1))


def unwrapped_rpy_deg(quaternions: np.ndarray) -> np.ndarray:
    """Roll, pitch, yaw in degrees, unwrapped along time.

    WHY unwrapped: a tool pointing down sits at roll ±180°, and the raw angle flips between +180 and −180 sample to
    sample; the unwrapped one is the same orientation drawn as a continuous line."""
    rpy = quat_to_euler(np.asarray(quaternions, dtype=np.float64))
    return np.degrees(np.unwrap(rpy, axis=0)) if len(rpy) > 1 else np.degrees(rpy)


@log.register
def _(traj: PoseTrajectory, name: str | None = None, *, recording: Any, **options: Any) -> None:
    if options:
        _stock(traj, name, recording, **options)
        return
    name = name or "PoseTrajectory"
    seconds = _seconds(traj)
    positions = np.asarray(traj.positions, dtype=np.float64)
    quaternions = np.asarray(traj.orientations, dtype=np.float64)  # xyzw, as Rerun takes it
    base = f"world/{part(name)}"
    origin_triad(recording)
    length = _frame_length(positions)
    recording.log(f"{base}/path", rr.LineStrips3D([positions], colors=[ACCENT], radii=rr.Radius.ui_points(2.0)),
                  static=True)  # fmt: skip
    along = np.unique(np.linspace(0, len(traj) - 1, min(len(traj), 9)).round().astype(int))
    recording.log(f"{base}/frames", _axes_arrows(positions[along], quaternions[along], length), static=True)
    ends = rr.Points3D([positions[0], positions[-1]], colors=[START, END], radii=rr.Radius.ui_points(5.0),
                       labels=["start", "end"], show_labels=True)  # fmt: skip
    recording.log(f"{base}/ends", ends, static=True)
    recording.log(f"{base}/frame", rr.TransformAxes3D(1.6 * length), static=True)
    recording.send_columns(
        f"{base}/frame",
        indexes=[rr.TimeColumn(TIMELINE, duration=seconds)],
        columns=rr.Transform3D.columns(translation=positions, quaternion=quaternions),
    )
    for axis, label in enumerate("xyz"):
        _series(recording, f"position_mm/{label}/{part(name)}", seconds, positions[:, axis] * 1000.0,
                label=_legend(f"{label} (mm)", name, "PoseTrajectory"), color=AXIS_COLORS[axis], width=2.0)
    rpy = unwrapped_rpy_deg(quaternions)
    for axis, label in enumerate(("roll", "pitch", "yaw")):
        _series(recording, f"orientation_deg/{label}/{part(name)}", seconds, rpy[:, axis],
                label=_legend(f"{label} (deg)", name, "PoseTrajectory"), color=AXIS_COLORS[axis], width=2.0)
    _log_channels(traj, name, recording, seconds)


# A pick seen at the tool: from home over to the bin, down, up, across to the place pose (x, y, z in m, yaw in deg),
# the tool pointing down the whole way (roll 180°).
TOOL_WAYPOINTS = (
    (0.30, -0.25, 0.45, 0.0),
    (0.45, 0.08, 0.28, 30.0),
    (0.45, 0.08, 0.12, 30.0),
    (0.45, 0.08, 0.28, 30.0),
    (0.12, 0.38, 0.24, 120.0),
)
TOOL_DURATIONS = (1.4, 0.7, 0.7, 1.6)


def tool_path(rate_hz: float = 50.0) -> PoseTrajectory:
    """The tool's pick path, minimum-jerk between stops, with its twist from `with_derivatives()`."""
    pieces_t, pieces_p, clock = [], [], 0.0
    for start, end, duration in zip(TOOL_WAYPOINTS[:-1], TOOL_WAYPOINTS[1:], TOOL_DURATIONS, strict=True):
        steps = int(round(duration * rate_hz))
        s = np.arange(steps) / steps
        pieces_t.append(clock + s * duration)
        pieces_p.append(np.asarray(start) + _min_jerk(s)[:, None] * (np.asarray(end) - np.asarray(start)))
        clock += duration
        hold = int(round(PICK_DWELL * rate_hz))
        pieces_t.append(clock + np.arange(hold) / rate_hz)
        pieces_p.append(np.repeat(np.asarray(end, dtype=np.float64)[None, :], hold, axis=0))
        clock += PICK_DWELL
    t = np.concatenate([*pieces_t, [clock]])
    xyzyaw = np.concatenate([*pieces_p, np.asarray(TOOL_WAYPOINTS[-1], dtype=np.float64)[None, :]])
    quaternions = quat_from_euler(np.full(len(t), np.pi), np.zeros(len(t)), np.radians(xyzyaw[:, 3]))
    traj = PoseTrajectory(t=t, positions=xyzyaw[:, :3], orientations=quaternions, child_frame_id="tool0",
                          header=Header(stamp=0.0, frame_id="base_link"))  # fmt: skip
    return traj.with_derivatives()


def _pose_views(name: str) -> list[Any]:
    return [
        *world(name),
        _plot("/position_mm", "position (mm)"),
        _plot("/orientation_deg", "roll / pitch / yaw (deg)"),
    ]


register(TypeView(
    type=PoseTrajectory,
    summary=("The path in 3D with x/y/z frames along it and one moving with the timeline, beside its position (mm) "
             "and roll / pitch / yaw (deg) over time."),
    sample=tool_path,
    views=_pose_views,
    tags=("3d", "plot"),
))  # fmt: skip


@log.register
def _(point: PoseTrajectoryPoint, name: str | None = None, *, recording: Any) -> None:
    name = name or "PoseTrajectoryPoint"
    base = f"world/{part(name)}"
    origin_triad(recording)
    position = np.asarray(point.position, dtype=np.float64)
    quaternion = np.asarray(point.orientation, dtype=np.float64)
    recording.log(f"{base}/frame", rr.Transform3D(translation=position, quaternion=quaternion),
                  rr.TransformAxes3D(stock.AXIS_LENGTH), static=True)  # fmt: skip
    roll, pitch, yaw = unwrapped_rpy_deg(quaternion[None, :])[0]
    label = f"t = {_num(point.t, 4)} s · roll {_num(roll, 3)}°, pitch {_num(pitch, 3)}°, yaw {_num(yaw, 3)}°"
    recording.log(f"{base}/at", rr.Points3D([position], colors=[ACCENT], radii=rr.Radius.ui_points(4.0)), static=True)
    # WHY an anchor above the frame: on the frame itself the label covered the triad it describes
    above = position + np.array([0.0, 0.0, 0.6 * stock.AXIS_LENGTH])
    recording.log(f"{base}/label", rr.Points3D([above], colors=[ACCENT], radii=0.0, labels=[label], show_labels=True),
                  static=True)  # fmt: skip
    if point.linear_velocity is not None:
        velocity = np.asarray(point.linear_velocity, dtype=np.float64)
        # WHY 0.5 s of travel: the arrow is drawn in metres, so a speed needs a time to become a length; the label
        # carries the speed itself
        recording.log(f"{base}/velocity", rr.Arrows3D(origins=[position], vectors=[0.5 * velocity], colors=[ACCENT],
                      radii=rr.Radius.ui_points(2.0), labels=[f"{np.linalg.norm(velocity) * 1000:.0f} mm/s"],
                      show_labels=True), static=True)  # fmt: skip


register(TypeView(
    type=PoseTrajectoryPoint,
    summary="One sample of a pose trajectory: its x/y/z frame, its time and orientation, and its velocity arrow.",
    sample=lambda: tool_path()[40],
    views=world,
    tags=("3d",),
))  # fmt: skip


# ── JointLimits: a chart of every joint's range and caps ─────────────────────────────────────────────────────────────

# (bound, column title, scale from SI, unit)
CAPS = (
    ("max_velocity", "velocity", DEG, "deg/s"),
    ("max_acceleration", "accel", DEG, "deg/s²"),
    ("max_jerk", "jerk", DEG, "deg/s³"),
    ("max_effort", "effort", 1.0, EFFORT_UNIT),
)


def _position_bounds(limits: JointLimits) -> tuple[np.ndarray, np.ndarray]:
    """Each joint's (low, high) position bound in degrees; an unstated side is ±∞ (unbounded)."""
    count = len(limits.joint_names)
    low = np.full(count, -np.inf) if limits.min_position is None else np.asarray(limits.min_position) * DEG
    high = np.full(count, np.inf) if limits.max_position is None else np.asarray(limits.max_position) * DEG
    return low, high


def _limits_table(limits: JointLimits) -> str:
    """Every bound, exactly: position range and kind, then each cap (max |value|) in its unit."""
    names = [str(n) for n in limits.joint_names]
    has_position = limits.min_position is not None or limits.max_position is not None
    present = [cap for cap in CAPS if getattr(limits, cap[0]) is not None]
    header = ["joint"]
    if has_position:
        header.append("position (deg)")
    if limits.position_kind is not None:
        header.append("kind")
    header += [f"{title} ({unit})" for _, title, _, unit in present]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    low, high = _position_bounds(limits)
    for index, joint in enumerate(names):
        cells = [joint]
        if has_position:
            cells.append(f"{_num(low[index])} … {_num(high[index])}")
        if limits.position_kind is not None:
            cells.append(limits.position_kind[index] or "not stated")
        for attribute, _, scale, _ in present:
            cells.append(_num(float(getattr(limits, attribute)[index]) * scale))
        lines.append("| " + " | ".join(cells) + " |")
    missing = [title for attribute, title, _, _ in CAPS if getattr(limits, attribute) is None]
    if not has_position:
        missing.insert(0, "position")
    lines.append("")
    if present:
        lines.append("Caps are magnitudes: |velocity| ≤ the velocity cap, and so on.")
    if missing:
        lines.append(f"Not stated (unbounded): {', '.join(missing)}.")
    lines += ["", "*Angles in degrees, every joint read as revolute (robot_msgs carries no joint type).*"]
    return "\n".join(lines)


# The chart's own coordinates (the 2D view's scene): degrees map onto x in [-1, 1], the rows fill y in [0, CHART_HEIGHT]
# whatever the joint count, so one fixed `VisualBounds2D` frames every JointLimits. WHY fixed: labels are drawn at a
# fixed size in pixels, and a scene that grew with the joint count shrank the rows under them.
CHART_HEIGHT = 1.6
CHART_BOUNDS = rrb.VisualBounds2D(x_range=[-1.3, 1.3], y_range=[-0.08, CHART_HEIGHT + 0.2])
TICK_STEPS = (15.0, 30.0, 45.0, 90.0, 180.0, 360.0, 720.0)  # degrees: quarter and half turns read at a glance
TICK_GREY = (120, 120, 120)


def _log_range_chart(limits: JointLimits, name: str, recording: Any) -> None:
    """A row per joint: its position range as a bar on one shared degree scale, labelled with the numbers and kind."""
    base = f"limits/{part(name)}"
    names = [str(n) for n in limits.joint_names]
    count = max(len(names), 1)
    low, high = _position_bounds(limits)
    finite = np.abs(np.concatenate([low[np.isfinite(low)], high[np.isfinite(high)]]))
    reach = float(finite.max()) if finite.size else 180.0
    step = next((s for s in TICK_STEPS if s >= reach / 2.0), TICK_STEPS[-1])
    edge = step * math.ceil(reach / step)
    pitch = CHART_HEIGHT / count

    def x(value: float) -> float:  # an unbounded side runs a little past the last tick
        return float(np.clip(value, -1.04 * edge, 1.04 * edge) / edge)

    bars, colors, label_at, label_text, label_colors = [], [], [], [], []
    for index, joint in enumerate(names):
        y = (index + 0.62) * pitch
        left, right = x(low[index]), x(high[index])
        bars.append([(left, y), (right, y)])
        colors.append(joint_color(index) or LIMIT_GREY)
        kind = "" if limits.position_kind is None or limits.position_kind[index] is None else f"  ({limits.position_kind[index]})"
        label_at.append((0.5 * (left + right), y - 0.4 * pitch))
        label_text.append(f"{joint}   {_num(low[index])} … {_num(high[index])}°{kind}")
        label_colors.append(joint_color(index) or LIMIT_GREY)
    grid, grid_colors = [], []
    for value in np.arange(-edge, edge + 0.5 * step, step):
        grid.append([(x(value), 0.0), (x(value), CHART_HEIGHT)])
        grid_colors.append(LIMIT_GREY if abs(value) < 1e-9 else (215, 215, 215))
        label_at.append((x(value), CHART_HEIGHT + 0.08))
        label_text.append(f"{_num(value)}°")
        label_colors.append(TICK_GREY)
    recording.log(f"{base}/grid", rr.LineStrips2D(grid, colors=grid_colors, radii=rr.Radius.ui_points(0.5),
                                                   draw_order=1.0), static=True)  # fmt: skip
    recording.log(f"{base}/ranges", rr.LineStrips2D(bars, colors=colors, radii=rr.Radius.ui_points(5.0),
                                                     draw_order=2.0), static=True)  # fmt: skip
    recording.log(f"{base}/labels", rr.Points2D(label_at, radii=0.0, colors=label_colors, labels=label_text,
                                                 show_labels=True, draw_order=3.0), static=True)  # fmt: skip


@log.register
def _(limits: JointLimits, name: str | None = None, *, recording: Any) -> None:
    name = name or "JointLimits"
    _log_range_chart(limits, name, recording)
    _notes(recording, name, _limits_table(limits))


def _limits_views(name: str) -> list[Any]:
    return [
        rrb.Spatial2DView(origin=f"/limits/{part(name)}", name="position range (deg)", visual_bounds=CHART_BOUNDS),
        _notes_view(name, "every bound"),
    ]


register(TypeView(
    type=JointLimits,
    summary=("Each joint's position range as a bar on one degree scale, beside a table of every bound: the range and "
             "its kind, and the velocity, acceleration, jerk and effort caps."),
    sample=arm_limits,
    views=_limits_views,
    tags=("2d", "text"),
))  # fmt: skip


# ── JointLimitViolation: one sentence ────────────────────────────────────────────────────────────────────────────────

_UNITS = {"position": "deg", "velocity": "deg/s", "acceleration": "deg/s²", "jerk": "deg/s³", "effort": EFFORT_UNIT}


def violation_sentence(violation: JointLimitViolation) -> str:
    """robot_msgs' own sentence (`str(violation)`), in degrees: ``elbow: |velocity| reaches 205 deg/s > 180 deg/s …``."""
    scale = 1.0 if violation.quantity == "effort" else DEG
    unit = _UNITS[violation.quantity]
    magnitude = violation.quantity != "position"
    label = f"|{violation.quantity}|" if magnitude else violation.quantity
    relation = "<" if violation.side == "min" else ">"
    shown = abs(violation.value) if magnitude else violation.value
    return (f"{violation.joint}: {label} reaches {_num(shown * scale)} {unit} {relation} "
            f"{_num(violation.limit * scale)} {unit} at t = {_num(violation.t)} s")  # fmt: skip


@log.register
def _(violation: JointLimitViolation, name: str | None = None, *, recording: Any) -> None:
    scale = 1.0 if violation.quantity == "effort" else DEG
    unit = _UNITS[violation.quantity]
    kind = f" ({violation.kind} limit)" if violation.kind else ""
    lines = [
        f"### {violation_sentence(violation)}",
        "",
        f"- past the {violation.side} {violation.quantity} limit{kind} by {_num(violation.excess * scale)} {unit}",
        f"- from t = {_num(violation.t_first)} s to t = {_num(violation.t_last)} s, {violation.samples} sample"
        f"{'s' if violation.samples != 1 else ''} past it",
        "",
        "*Angles in degrees, the joint read as revolute (robot_msgs carries no joint type).*",
    ]
    _notes(recording, name or "JointLimitViolation", "\n".join(lines))


def fast_pick_violation() -> JointLimitViolation:
    """The pick run twice as fast: the wrist 3's turn to the place pose goes past its 180°/s cap."""
    fast = pick_move().time_scaled(0.5)
    found = [v for v in joint_limit_violations(fast) if v.quantity == "velocity"]
    return found[0]


register(TypeView(
    type=JointLimitViolation,
    summary="The violation as one sentence in degrees: which joint, which bound, how far past it, when and for how long.",
    sample=fast_pick_violation,
    views=lambda name: [_notes_view(name)],
    tags=("text",),
))  # fmt: skip
