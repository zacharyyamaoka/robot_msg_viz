"""Stock views of robot_msgs' recorded streams (docs/CONTRACT.md: one TypeView per type, registered on import).

    ServoStateTrajectory    a drive log: position, velocity, effort, temperature, voltage and fault code per servo
                            over time (one tab each), beside a summary table per servo
    ServoCommandTrajectory  what was sent: position / velocity targets, feed-forward, stiffness, damping, effort cap
                            per servo, and each command's latency (ms)
    ImuTrajectory           the sensor board turning in 3D on the timeline, beside acceleration (m/s²), angular
                            velocity (deg/s) and roll / pitch / yaw (deg), x/y/z in red/green/blue
    WrenchTrajectory        the force and torque arrows at the sensor on the timeline, beside force (N) and torque
                            (N·m) over time and a contact summary
    ImageTrajectory         the frames on the timeline, beside any per-frame channels (exposure, a score)
    …TrajectoryPoint        one sample: the servo points as a table per joint (degrees), the IMU / wrench / image
                            points through their single message's own view (Imu, Wrench, Image)

Entity paths are the single messages' (control.py's ServoState / ServoCommand, sensor.py's Imu, geometry.py's
Wrench), so a stream overlays the live messages it was recorded from: ``/positions/<joint>/<name>`` (deg),
``/velocities/…`` (deg/s), ``/effort/…`` (N·m; a command's feed-forward), ``/temperature/…`` (°C), ``/voltage/…`` (V),
``/fault/…``, ``/latency/<name>`` (ms), ``/linear_acceleration|angular_velocity|orientation/<axis>/<name>``,
``/force|torque/<axis>/<name>``, ``/image/<name>``, ``/world/<name>``.

Several streams of one type shown together (``viz.show(sim=a, real=b)``) share every plot: each joint keeps its hue
and each source after the first gets a lighter (then darker) shade of it, the legend naming the source. WHY by the
order the sources were logged into the recording, not by their names: any names work (sim / real, run_3 / run_4),
and robot_msgs' Run takes its first recording as the reference too.

Every logger here also takes a ``root`` (an entity-path prefix): runs.py nests one Run's streams under
``/streams/<stream>/`` (and a Dataset's under ``/runs/<run>/streams/<stream>/``) so each run can be opened on its own.

WHY overrides of robot_msgs' stock logger: the stock plots these streams in radians with one colour per path and puts
the IMU frame and the wrench at the stock series paths (``/orientations/x``, a quaternion component); here angles are
degrees (Zach's rule), joints keep the JointTrajectory view's colours, and the paths overlay the single messages'.
"""

from __future__ import annotations

# WHY empty: this module registers its TypeViews on import; its functions are helpers, not API (see compare.py).
__all__: list[str] = []

from collections.abc import Sequence
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from robot_msgs.geometry import Wrench
from robot_msgs.sensor import Image, Imu
from robot_msgs.trajectory import (
    ImageTrajectory,
    ImageTrajectoryPoint,
    ImuTrajectory,
    ImuTrajectoryPoint,
    ServoCommandTrajectory,
    ServoCommandTrajectoryPoint,
    ServoStateTrajectory,
    ServoStateTrajectoryPoint,
    WrenchTrajectory,
    WrenchTrajectoryPoint,
)

from robot_msg_viz._engine import TIMELINE, log, part
from robot_msg_viz.registry import TypeView, register, view_for
from robot_msg_viz.views import _sim2real as sim2real
from robot_msg_viz.views._common import DEG
from robot_msg_viz.views.geometry import ANGULAR, LINEAR
from robot_msg_viz.views.sensor import AXES, GREY, IMU_EYE, INK, THIN, _IMU_BOARD
from robot_msg_viz.views.trajectory import (
    EFFORT_UNIT,
    JOINT_COLORS,
    LIMIT_GREY,
    _num,
    _plot,
    _point_heading,
    _series,
    unwrapped_rpy_deg,
)

# ── shared: shades per source, the page, the notes ───────────────────────────────────────────────────────────────────

_SOURCES: dict[tuple[int, str], list[str]] = {}
_SOURCES_KEPT = 256  # recordings remembered (a long session would otherwise grow the table forever)


def source_index(recording: Any, leaf: str, root: str = "") -> int:
    """0 for the first source logged under `root` into this recording, 1 for the next, … (the same leaf keeps its)."""
    key = (id(getattr(recording, "_inner", recording)), root)
    seen = _SOURCES.setdefault(key, [])
    if leaf not in seen:
        seen.append(leaf)
    while len(_SOURCES) > _SOURCES_KEPT:
        _SOURCES.pop(next(iter(_SOURCES)))
    return seen.index(leaf)


def shade(colour: Sequence[int] | None, index: int) -> tuple[int, int, int] | None:
    """`colour` for the first source; lighter, then darker, then lighter still for the ones after it."""
    if colour is None:
        return None
    mix = (0.0, 0.5, -0.4, 0.75, -0.6)[index % 5]
    target = 255 if mix > 0 else 0
    return tuple(int(round(c + (target - c) * abs(mix))) for c in colour)  # type: ignore[return-value]


def _width(index: int) -> float:
    return 2.0 if index == 0 else 1.5


def series_group(recording: Any, root: str, field: str, leaf: str, seconds: np.ndarray, values: np.ndarray,
                 columns: Sequence[str], labels: Sequence[str], colors: Sequence[Any], width: float) -> None:  # fmt: skip
    """Several columns of one field: ``<field>/<column>/<leaf>`` each at the top level (the contract's layout, so a
    stream overlays its single messages), but ONE entity ``<root><field>/<leaf>`` with a line per column when nested
    under a `root` (a run of a Dataset).

    WHY one entity when nested: an .rrd pays ~9 KB of schema per entity whatever its length, so a 4-run sweep of 6-joint
    servo logs was 10 MB as one entity per joint and field (measured 2026-11-25); a batch of 6 lines is one entity
    (21 KB) instead of six (57 KB), and nested runs never overlay a lone message's paths anyway."""
    values = np.asarray(values, dtype=np.float64).reshape(len(seconds), -1)
    if not root:
        for index, column in enumerate(columns):
            _series(recording, f"{field}/{part(column)}/{leaf}", seconds, values[:, index], label=labels[index],
                    color=colors[index], width=width)  # fmt: skip
        return
    path = f"{root}{field}/{leaf}"
    filled = [LIMIT_GREY if colour is None else colour for colour in colors]
    recording.log(path, rr.SeriesLines(names=list(labels), colors=filled, widths=[width] * len(labels)), static=True)
    recording.send_columns(path, indexes=[rr.TimeColumn(TIMELINE, duration=np.asarray(seconds, dtype=np.float64))],
                           columns=rr.Scalars.columns(scalars=values))  # fmt: skip


def joint_hue(index: int) -> tuple[int, int, int] | None:
    return JOINT_COLORS[index] if index < len(JOINT_COLORS) else None


def _seconds(t: Any) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    return t - t[0] if t.size else t


def _who(subject: str, leaf_name: str, default: str) -> str:
    """A legend entry: ``wrist_1_joint`` alone, ``wrist_1_joint · real`` when the caller named the message."""
    return subject if leaf_name == default else f"{subject} · {leaf_name}"


def _note(recording: Any, path: str, text: str) -> None:
    recording.log(path, rr.TextDocument(text, media_type="text/markdown"), static=True)


def _panels(root: Any) -> rrb.Blueprint:
    """A page: one root container, side panels collapsed, the time panel collapsed (it still shows the scrubber)."""
    return rrb.Blueprint(root, rrb.BlueprintPanel(state="collapsed"), rrb.SelectionPanel(state="collapsed"),
                         rrb.TimePanel(state="collapsed"))  # fmt: skip


def _tabs(views: list[Any], name: str | None = None) -> Any:
    return views[0] if len(views) == 1 else rrb.Tabs(*views, name=name)


def side_by_side(main: Any, side: Any, shares: tuple[int, int] = (3, 2), *, name: str | None = None) -> Any:
    return rrb.Horizontal(main, side, column_shares=list(shares), name=name)


def _log_channels(channels: Any, seconds: np.ndarray, recording: Any, root: str, leaf: str, label: str,
                  skip: Sequence[str] = ()) -> list[str]:  # fmt: skip
    """Every numeric 1-D channel at ``<root>channels/<key>/<leaf>``; returns the keys logged."""
    logged = []
    for key, raw in (channels or {}).items():
        values = np.asarray(raw)
        if key in skip or values.dtype.kind not in "fiub" or values.shape != seconds.shape:
            continue
        _series(recording, f"{root}channels/{part(key)}/{leaf}", seconds, values.astype(np.float64),
                label=f"{key} · {label}" if label else key, color=INK, width=1.5)  # fmt: skip
        logged.append(str(key))
    return logged


def _range(values: np.ndarray) -> str:
    finite = np.round(values[np.isfinite(values)], 9) + 0.0  # WHY rounded: 2.5e-17 is a zero that lost its sign
    if not finite.size:
        return "—"
    return f"{_num(finite.min(), 3)} … {_num(finite.max(), 3)}"


def _peak(values: np.ndarray) -> str:
    finite = np.abs(values[np.isfinite(values)])
    return _num(finite.max(), 3) if finite.size else "—"


def _span(t: np.ndarray) -> str:
    seconds = _seconds(t)
    if len(seconds) < 2:
        return f"{len(seconds)} sample"
    rate = (len(seconds) - 1) / seconds[-1] if seconds[-1] > 0 else float("nan")
    return f"{len(seconds)} samples · {_num(seconds[-1], 3)} s · {_num(rate, 3)} Hz"


# ── servo streams: a column per joint, one series per joint and field ────────────────────────────────────────────────

# (field on the stream, entity-path field, word, unit, scale from SI): the paths ServoState / ServoCommand use
SERVO_STATE_FIELDS = (
    ("positions", "positions", "position", "deg", DEG),
    ("velocities", "velocities", "velocity", "deg/s", DEG),
    ("effort", "effort", "effort", EFFORT_UNIT, 1.0),
    ("temperature", "temperature", "temperature", "°C", 1.0),
    ("voltage", "voltage", "voltage", "V", 1.0),
    ("fault", "fault", "fault code", "", 1.0),
)
SERVO_COMMAND_FIELDS = (
    ("positions", "positions", "position target", "deg", DEG),
    ("velocities", "velocities", "velocity target", "deg/s", DEG),
    ("feedforward_effort", "effort", "feed-forward", EFFORT_UNIT, 1.0),
    ("stiffness", "stiffness", "stiffness", "N·m/rad", 1.0),
    ("damping", "damping", "damping", "N·m·s/rad", 1.0),
    ("max_effort", "max_effort", "effort cap", EFFORT_UNIT, 1.0),
)
_COMMAND_LIGHTEN = 0.45  # WHY (control.py's ServoCommand): a target overlaid on its measurement reads second


def _log_joint_fields(traj: Any, fields: Sequence[tuple[str, str, str, str, float]], recording: Any, root: str,
                      leaf: str, default: str, *, command: bool) -> list[str]:  # fmt: skip
    seconds = _seconds(traj.t)
    index = source_index(recording, leaf, root)
    logged = []
    for attribute, field, word, unit, scale in fields:
        values = getattr(traj, attribute)
        if values is None:
            continue
        shown = np.asarray(values, dtype=np.float64) * scale
        labels, colors = [], []
        for joint, servo in enumerate(traj.joint_names):
            hue = joint_hue(joint)
            if command and hue is not None:
                hue = tuple(int(c + (255 - c) * _COMMAND_LIGHTEN) for c in hue)
            what = f"{servo} {word}" if command else str(servo)
            labels.append(_who(what, leaf, default) + (f" ({unit})" if unit else ""))
            colors.append(shade(hue, index))
        series_group(recording, root, field, leaf, seconds, shown, [str(n) for n in traj.joint_names], labels, colors,
                     _width(index) - (0.5 if command else 0.0))  # fmt: skip
        logged.append(field)
    return logged


# WHY short headers: six columns must fit the side panel beside the plots (the long ones were cut off at 1280 px)
_HEADERS = {
    "positions": "range (deg)", "velocities": "max \\|v\\| (deg/s)", "effort": "max \\|τ\\| (N·m)",
    "temperature": "°C start → end", "voltage": "V start → end", "fault": "fault",
    "feedforward_effort": "max \\|τ ff\\| (N·m)", "stiffness": "kp (N·m/rad)", "damping": "kd (N·m·s/rad)",
    "max_effort": "cap (N·m)",
}  # fmt: skip


def _servo_summary(traj: Any, fields: Sequence[tuple[str, str, str, str, float]], heading: str) -> str:
    """One row per servo (a 6-axis arm's long joint names fit a narrow panel that way), one column per field."""
    columns: list[tuple[str, list[str]]] = []
    count = len(traj.joint_names)
    for attribute, _, word, unit, scale in fields:
        values = getattr(traj, attribute)
        if values is None:
            continue
        shown = np.asarray(values, dtype=np.float64) * scale
        header = _HEADERS.get(attribute, f"{word} ({unit})" if unit else word)
        if attribute == "fault":
            codes = np.asarray(values)
            cells = [", ".join(str(int(c)) for c in np.unique(codes[:, j]) if c) or "0" for j in range(count)]
            columns.append((header, [f"**{c}**" if c != "0" else c for c in cells]))
        elif attribute in ("temperature", "voltage"):
            columns.append((header, [f"{_num(shown[0, j], 3)} → {_num(shown[-1, j], 3)}" for j in range(count)]))
        elif attribute == "positions":
            columns.append((header, [_range(shown[:, j]) for j in range(count)]))
        else:
            columns.append((header, [_peak(shown[:, j]) for j in range(count)]))
    lines = [heading, "", "| servo | " + " | ".join(title for title, _ in columns) + " |",
             "|---|" + "---:|" * len(columns)]  # fmt: skip
    for j, servo in enumerate(traj.joint_names):
        lines.append(f"| {servo} | " + " | ".join(cells[j] for _, cells in columns) + " |")
    return "\n".join(lines)


def _fault_line(traj: ServoStateTrajectory) -> str:
    if traj.fault is None:
        return "fault codes not reported"
    codes = np.asarray(traj.fault)
    seconds = _seconds(traj.t)
    faulted = []
    for joint, servo in enumerate(traj.joint_names):
        hits = np.nonzero(codes[:, joint])[0]
        if hits.size:
            seen = ", ".join(str(int(c)) for c in np.unique(codes[hits, joint]))
            faulted.append(f"{servo} code {seen} at {_num(seconds[hits[0]], 3)}–{_num(seconds[hits[-1]], 3)} s")
    return f"**faulted: {'; '.join(faulted)}**" if faulted else "no faults"


def log_servo_states(traj: ServoStateTrajectory, recording: Any, *, root: str = "", name: str | None = None) -> None:
    name = name or "ServoStateTrajectory"
    leaf = part(name)
    _log_joint_fields(traj, SERVO_STATE_FIELDS, recording, root, leaf, "ServoStateTrajectory", command=False)
    _log_channels(traj.channels, _seconds(traj.t), recording, root, leaf, name)
    heading = f"**{name}** · {_span(traj.t)} · {_fault_line(traj)}"
    text = _servo_summary(traj, SERVO_STATE_FIELDS, heading) + "\n\nfault 0 = no fault · angles in degrees"
    _note(recording, f"{root}notes/{leaf}", text)


def log_servo_commands(
    traj: ServoCommandTrajectory, recording: Any, *, root: str = "", name: str | None = None
) -> None:
    name = name or "ServoCommandTrajectory"
    leaf = part(name)
    _log_joint_fields(traj, SERVO_COMMAND_FIELDS, recording, root, leaf, "ServoCommandTrajectory", command=True)
    seconds = _seconds(traj.t)
    latency = (traj.channels or {}).get("latency")
    heading = f"**{name}** · {_span(traj.t)}"
    if latency is not None:
        ms = np.asarray(latency, dtype=np.float64) * 1000.0
        _series(recording, f"{root}latency/{leaf}", seconds, ms, label=f"latency (ms) · {name}",
                color=shade(INK, source_index(recording, leaf, root)), width=1.5)  # fmt: skip
        finite = ms[np.isfinite(ms)]
        if finite.size:
            heading += f" · latency median {_num(np.median(finite), 3)} ms, worst {_num(finite.max(), 3)} ms"
    _log_channels(traj.channels, seconds, recording, root, leaf, name, skip=("latency",))
    text = _servo_summary(traj, SERVO_COMMAND_FIELDS, heading) + "\n\nNaN position target = none sent · angles in degrees"
    _note(recording, f"{root}notes/{leaf}", text)


def _servo_views(fields: Sequence[tuple[str, str, str, str, float]], root: str = "",
                 present: set[str] | None = None) -> list[Any]:  # fmt: skip
    views = []
    for attribute, field, word, unit, _ in fields:
        if present is not None and attribute not in present:
            continue
        views.append(_plot(f"/{root}{field}", f"{word}{f' ({unit})' if unit else ''}"))
    return views


def servo_state_views(root: str = "", traj: ServoStateTrajectory | None = None) -> list[Any]:
    present = None if traj is None else {f[0] for f in SERVO_STATE_FIELDS if getattr(traj, f[0]) is not None}
    return _servo_views(SERVO_STATE_FIELDS, root, present)


def servo_command_views(root: str = "", traj: ServoCommandTrajectory | None = None) -> list[Any]:
    present = None if traj is None else {f[0] for f in SERVO_COMMAND_FIELDS if getattr(traj, f[0]) is not None}
    views = _servo_views(SERVO_COMMAND_FIELDS, root, present)
    if traj is None or (traj.channels or {}).get("latency") is not None:
        views.append(_plot(f"/{root}latency", "latency (ms)"))
    return views


@log.register
def _(traj: ServoStateTrajectory, name: str | None = None, *, recording: Any) -> None:
    log_servo_states(traj, recording, name=name)


@log.register
def _(traj: ServoCommandTrajectory, name: str | None = None, *, recording: Any) -> None:
    log_servo_commands(traj, recording, name=name)


def _notes_view(root: str, name: str, title: str | None = None) -> rrb.View:
    return rrb.TextDocumentView(origin=f"/{root}notes/{part(name)}", name=title or name)


register(TypeView(
    type=ServoStateTrajectory,
    summary=("A drive log: each servo's position (deg), velocity (deg/s), effort (N·m), temperature (°C), voltage (V) "
             "and fault code over time, one tab each, beside its range, peaks and faults per servo."),
    sample=lambda: sim2real.servo_states("real"),
    views=lambda name: [*servo_state_views(), _notes_view("", name)],
    page=lambda traj, name, **_: _panels(side_by_side(rrb.Tabs(*servo_state_views("", traj)), _notes_view("", name))),
    code=(
        "import robot_msg_viz as viz\nfrom robot_msgs.trajectory import ServoStateTrajectory\n\n"
        "log = ServoStateTrajectory.from_servo_states(states)   # a drive's ServoState log\n"
        "viz.show(log)                                          # one tab per field, a table per servo\n"
        "viz.show(sim=sim_log, real=real_log)                   # overlaid: each joint's hue, real a lighter shade"
    ),
    tags=("plot", "stream"),
))  # fmt: skip

register(TypeView(
    type=ServoCommandTrajectory,
    summary=("What the controller sent: position and velocity targets (deg, deg/s), feed-forward, stiffness, damping "
             "and effort cap per servo over time, and each command's latency (ms)."),
    sample=lambda: sim2real.servo_commands("real"),
    views=lambda name: [*servo_command_views(), _notes_view("", name)],
    page=lambda traj, name, **_: _panels(side_by_side(rrb.Tabs(*servo_command_views("", traj)), _notes_view("", name))),
    tags=("plot", "stream"),
))  # fmt: skip


# ── IMU stream: the board turning in 3D, and its three signals ───────────────────────────────────────────────────────

_BOARD_SPACING = 0.12  # m between the boards of several sources in one world view (side by side, not on top)


def _imu_board(traj: ImuTrajectory, recording: Any, leaf: str, name: str, index: int, seconds: np.ndarray) -> None:
    """The board: at rest (a grey level outline, fixed) and turned by each sample's orientation, on the timeline.

    Only at the top level, as `_wrench_arrows`."""
    hx, hy = _IMU_BOARD[0], _IMU_BOARD[1]
    offset = np.array([_BOARD_SPACING * index, 0.0, 0.0])
    outline = np.array([[-hx, -hy, 0.0], [hx, -hy, 0.0], [hx, hy, 0.0], [-hx, hy, 0.0], [-hx, -hy, 0.0]]) + offset
    recording.log(f"world/{leaf}_level", rr.LineStrips3D([outline], colors=[GREY], radii=THIN), static=True)
    body = f"world/{leaf}"
    recording.log(body, rr.TransformAxes3D(0.08), static=True)
    recording.send_columns(
        body,
        indexes=[rr.TimeColumn(TIMELINE, duration=seconds)],
        columns=rr.Transform3D.columns(translation=np.tile(offset, (len(seconds), 1)),
                                       quaternion=np.asarray(traj.orientations, dtype=np.float64)),  # fmt: skip
    )
    recording.log(f"{body}/board", rr.Boxes3D(half_sizes=[_IMU_BOARD], colors=[(*shade(INK, index), 230)],
                                              fill_mode="solid", labels=[name], show_labels=index > 0 or name != "ImuTrajectory"),
                  static=True)  # fmt: skip


def log_imu(traj: ImuTrajectory, recording: Any, *, root: str = "", name: str | None = None) -> None:
    name = name or "ImuTrajectory"
    leaf = part(name)
    seconds = _seconds(traj.t)
    index = source_index(recording, leaf, root)
    groups = (
        ("linear_acceleration", ("x", "y", "z"), np.asarray(traj.linear_accelerations, dtype=np.float64), "m/s²"),
        ("angular_velocity", ("x", "y", "z"), np.asarray(traj.angular_velocities, dtype=np.float64) * DEG, "deg/s"),
        ("orientation", ("roll", "pitch", "yaw"), unwrapped_rpy_deg(np.asarray(traj.orientations)), "deg"),
    )
    for field, columns, values, unit in groups:
        series_group(recording, root, field, leaf, seconds, values, columns,
                     [_who(f"{column} ({unit})", name, "ImuTrajectory") for column in columns],
                     [shade(colour, index) for colour in AXES], _width(index))  # fmt: skip
    _log_channels(traj.channels, seconds, recording, root, leaf, name)
    if not root:
        _imu_board(traj, recording, leaf, name, index, seconds)
    accel = np.asarray(traj.linear_accelerations, dtype=np.float64)
    gyro = np.asarray(traj.angular_velocities, dtype=np.float64) * DEG
    rpy = groups[2][2]
    rows = [f"**{name}** · {_span(traj.t)} · frame `{traj.header.frame_id or '—'}`", "",
            "| | x / roll | y / pitch | z / yaw |", "|---|---:|---:|---:|",
            "| mean acceleration (m/s²) | " + " | ".join(_num(v, 3) for v in accel.mean(axis=0)) + " |",
            "| peak \\|angular velocity\\| (deg/s) | " + " | ".join(_peak(gyro[:, a]) for a in range(3)) + " |",
            "| orientation range (deg) | " + " | ".join(_range(rpy[:, a]) for a in range(3)) + " |",
            "", f"mean \\|a\\| = {_num(np.linalg.norm(accel, axis=1).mean(), 4)} m/s² (gravity alone reads 9.807)"]  # fmt: skip
    _note(recording, f"{root}notes/{leaf}", "\n".join(rows))


def imu_views(root: str = "") -> list[Any]:
    """The sensor in 3D and its three plots; nested under a `root` (a run), the plots only (see `log_wrench`)."""
    world = [] if root else [rrb.Spatial3DView(origin="/world", name="sensor", eye_controls=IMU_EYE)]
    return [
        *world,
        _plot(f"/{root}linear_acceleration", "acceleration (m/s²)"),
        _plot(f"/{root}angular_velocity", "angular velocity (deg/s)"),
        _plot(f"/{root}orientation", "roll / pitch / yaw (deg)"),
    ]


@log.register
def _(traj: ImuTrajectory, name: str | None = None, *, recording: Any) -> None:
    log_imu(traj, recording, name=name)


def _imu_page(traj: ImuTrajectory, name: str, **_: Any) -> rrb.Blueprint:
    world, *plots = imu_views()
    return _panels(side_by_side(rrb.Vertical(world, _notes_view("", name), row_shares=[3, 2]), rrb.Vertical(*plots),
                                (2, 3)))  # fmt: skip


register(TypeView(
    type=ImuTrajectory,
    summary=("An IMU log: the sensor board turning in 3D on the timeline, beside acceleration (m/s²), angular "
             "velocity (deg/s) and roll / pitch / yaw (deg) over time, x/y/z in red/green/blue."),
    sample=lambda: sim2real.imu("real"),
    views=lambda name: [*imu_views(), _notes_view("", name)],
    page=_imu_page,
    tags=("3d", "plot", "stream"),
))  # fmt: skip


# ── wrench stream: force and torque over time, and the arrows at the sensor ──────────────────────────────────────────

_FORCE_DRAW = 0.01  # m per N (geometry.py's Wrench: 100 N draws 1 m)
_TORQUE_DRAW = 0.2  # m per N·m
_CONTACT_N = 2.0  # |F| above this counts as touching something


def _wrench_arrows(forces: np.ndarray, torques: np.ndarray, recording: Any, leaf: str, index: int,
                   seconds: np.ndarray) -> None:  # fmt: skip
    """The force and torque arrows at the sensor on the timeline, and each one's peak, faint and fixed.

    WHY only at the top level: nested under a run they cost ~10 entities per source (an .rrd pays ~9 KB of schema per
    entity) for a picture the plots already carry; a run's drill-down is about the error, so it shows the plots."""
    offset = np.array([0.0, 0.12 * index, 0.0])
    origins = np.tile(offset, (len(seconds), 1))
    recording.log(f"world/{leaf}", rr.TransformAxes3D(0.04), rr.Transform3D(translation=offset), static=True)
    for word, values, scale, colour in (("force", forces, _FORCE_DRAW, LINEAR), ("torque", torques, _TORQUE_DRAW, ANGULAR)):
        path = f"world/{leaf}_{word}"
        recording.log(path, rr.Arrows3D.from_fields(colors=[shade(colour, index)], radii=rr.Radius.ui_points(2.5)),
                      static=True)  # fmt: skip
        recording.send_columns(
            path,
            indexes=[rr.TimeColumn(TIMELINE, duration=seconds)],
            columns=rr.Arrows3D.columns(vectors=values * scale, origins=origins).partition([1] * len(seconds)),
        )
        # the peak, faint and fixed: a still or a paused moment still shows how hard the contact was
        peak = int(np.argmax(np.linalg.norm(values, axis=1)))
        unit = "N" if word == "force" else "N·m"
        recording.log(f"{path}_peak", rr.Arrows3D(origins=[offset], vectors=[values[peak] * scale],
                                                  colors=[shade(colour, index + 3)], radii=rr.Radius.ui_points(1.0),
                                                  labels=[f"peak {word} {_num(np.linalg.norm(values[peak]), 3)} {unit}"],
                                                  show_labels=True), static=True)  # fmt: skip


def log_wrench(traj: WrenchTrajectory, recording: Any, *, root: str = "", name: str | None = None) -> None:
    name = name or "WrenchTrajectory"
    leaf = part(name)
    seconds = _seconds(traj.t)
    index = source_index(recording, leaf, root)
    forces = np.asarray(traj.forces, dtype=np.float64)
    torques = np.asarray(traj.torques, dtype=np.float64)
    for field, values, unit in (("force", forces, "N"), ("torque", torques, "N·m")):
        norm = np.linalg.norm(values, axis=1)[:, None]
        labels = [_who(f"{column} ({unit})", name, "WrenchTrajectory") for column in "xyz"]
        labels.append(_who(f"|{field[0].upper()}| ({unit})", name, "WrenchTrajectory"))
        series_group(recording, root, field, leaf, seconds, np.hstack([values, norm]), ["x", "y", "z", "norm"], labels,
                     [shade(colour, index) for colour in (*AXES, LIMIT_GREY)], _width(index))  # fmt: skip
    _log_channels(traj.channels, seconds, recording, root, leaf, name)
    if not root:
        _wrench_arrows(forces, torques, recording, leaf, index, seconds)
    magnitude = np.linalg.norm(forces, axis=1)
    touching = np.nonzero(magnitude > _CONTACT_N)[0]
    contact = (f"in contact (|F| > {_num(_CONTACT_N)} N) {_num(seconds[touching[0]], 3)}–{_num(seconds[touching[-1]], 3)} s"
               if touching.size else f"never above {_num(_CONTACT_N)} N")  # fmt: skip
    rows = [f"**{name}** · {_span(traj.t)} · frame `{traj.header.frame_id or '—'}` · {contact}", "",
            "| | x | y | z | \\|·\\| |", "|---|---:|---:|---:|---:|"]  # fmt: skip
    for word, values, unit in (("force", forces, "N"), ("torque", torques, "N·m")):
        norm = np.linalg.norm(values, axis=1)
        rows.append(f"| {word} range ({unit}) | " + " | ".join(_range(values[:, a]) for a in range(3)) + f" | peak {_num(norm.max(), 3)} |")
        rows.append(f"| {word} at start ({unit}) | " + " | ".join(_num(values[0, a], 3) for a in range(3)) + f" | {_num(norm[0], 3)} |")
    rows += ["", "arrows: force orange (1 cm per N), torque purple (20 cm per N·m), at the sensor frame; the thin ones "
             "are each one's peak"]
    _note(recording, f"{root}notes/{leaf}", "\n".join(rows))


_WRENCH_EYE = rrb.EyeControls3D(position=(0.2, -0.3, 0.06), look_target=(0.0, 0.03, -0.1), eye_up=(0.0, 0.0, 1.0))


def wrench_views(root: str = "") -> list[Any]:
    """The arrows in 3D and the two plots; nested under a `root` (a run), the plots only."""
    world = [] if root else [rrb.Spatial3DView(origin="/world", name="sensor", eye_controls=_WRENCH_EYE)]
    return [
        *world,
        _plot(f"/{root}force", "force (N)"),
        _plot(f"/{root}torque", "torque (N·m)"),
    ]


@log.register
def _(traj: WrenchTrajectory, name: str | None = None, *, recording: Any) -> None:
    log_wrench(traj, recording, name=name)


def _wrench_page(traj: WrenchTrajectory, name: str, **_: Any) -> rrb.Blueprint:
    world, *plots = wrench_views()
    return _panels(side_by_side(rrb.Vertical(*plots), rrb.Vertical(world, _notes_view("", name), row_shares=[3, 2])))


register(TypeView(
    type=WrenchTrajectory,
    summary=("An F/T sensor log: force (N) and torque (N·m) over time, x/y/z in red/green/blue with the magnitude in "
             "grey, the two arrows at the sensor on the timeline, and when it was in contact."),
    sample=lambda: sim2real.wrench("real"),
    views=lambda name: [*wrench_views(), _notes_view("", name)],
    page=_wrench_page,
    tags=("3d", "plot", "stream"),
))  # fmt: skip


# ── image stream: the frames on the timeline ─────────────────────────────────────────────────────────────────────────

_CHANNEL_DATATYPES = {"u1": "U8", "u2": "U16", "f4": "F32"}
_COLOR_MODELS = {"rgb8": "RGB", "bgr8": "BGR", "rgba8": "RGBA", "bgra8": "BGRA", "mono8": "L", "mono16": "L"}
_DEPTH = ("16UC1", "32FC1")


def log_frames(frames: np.ndarray, seconds: np.ndarray, path: str, recording: Any, *, encoding: str) -> None:
    """Every frame at `path` on the ``t`` timeline: one static format, then one buffer per frame (Rerun's columnar
    image logging, so a long stream is one send). Depth encodings as depth images (16UC1 in mm)."""
    data = np.ascontiguousarray(frames)
    height, width = data.shape[1:3]
    datatype = _CHANNEL_DATATYPES[data.dtype.str.lstrip("<>|=")]
    buffers = data.reshape(len(seconds), -1).view(np.uint8)
    if encoding in _DEPTH:
        fmt = rr.components.ImageFormat(width=width, height=height, channel_datatype=datatype)
        recording.log(path, rr.DepthImage.from_fields(format=fmt, meter=1000.0 if encoding == "16UC1" else 1.0,
                                                      colormap="turbo"), static=True)  # fmt: skip
        columns = rr.DepthImage.columns(buffer=buffers)
    else:
        model = _COLOR_MODELS.get(encoding, "RGB" if data.ndim == 4 and data.shape[3] == 3 else "L")
        fmt = rr.components.ImageFormat(width=width, height=height, color_model=model, channel_datatype=datatype)
        recording.log(path, rr.Image.from_fields(format=fmt), static=True)
        columns = rr.Image.columns(buffer=buffers)
    recording.send_columns(path, indexes=[rr.TimeColumn(TIMELINE, duration=seconds)], columns=columns)


def log_images(traj: ImageTrajectory, recording: Any, *, root: str = "", name: str | None = None) -> list[str]:
    name = name or "ImageTrajectory"
    leaf = part(name)
    seconds = _seconds(traj.t)
    log_frames(np.asarray(traj.frames), seconds, f"{root}image/{leaf}", recording, encoding=traj.encoding)
    channels = _log_channels(traj.channels, seconds, recording, root, leaf, name)
    rows = [f"**{name}** · {_span(traj.t)} · {traj.width} × {traj.height} `{traj.encoding}` · frame "
            f"`{traj.header.frame_id or '—'}`", "",
            "Each frame is held until the next (a camera frame is never blended)."]  # fmt: skip
    if channels:
        rows += ["", "| channel | range |", "|---|---|"]
        rows += [f"| {key} | {_range(np.asarray(traj.channels[key], dtype=np.float64))} |" for key in channels]
    _note(recording, f"{root}notes/{leaf}", "\n".join(rows))
    return channels


def image_views(root: str, name: str, traj: ImageTrajectory | None = None) -> list[Any]:
    views: list[Any] = [rrb.Spatial2DView(origin=f"/{root}image/{part(name)}", name=name)]
    if traj is not None and traj.channels:
        views.append(_plot(f"/{root}channels", "per-frame channels"))
    return views


@log.register
def _(traj: ImageTrajectory, name: str | None = None, *, recording: Any) -> None:
    log_images(traj, recording, name=name)


def _image_page(traj: ImageTrajectory, name: str, **_: Any) -> rrb.Blueprint:
    picture, *rest = image_views("", name, traj)
    side = rrb.Vertical(*rest, _notes_view("", name), row_shares=[1] * len(rest) + [1]) if rest else _notes_view("", name)
    return _panels(side_by_side(picture, side))


register(TypeView(
    type=ImageTrajectory,
    summary=("A camera stream: the frames on the timeline (scrub or play it), beside the per-frame channels "
             "(exposure, a score) and the stream's size, rate and encoding."),
    sample=lambda: sim2real.camera("real"),
    views=lambda name: [*image_views("", name), _notes_view("", name)],
    page=_image_page,
    tags=("2d", "stream"),
))  # fmt: skip


# ── the points: one sample ───────────────────────────────────────────────────────────────────────────────────────────


def _servo_point_table(point: Any, fields: Sequence[tuple[str, str, str, str, float]], name: str) -> str:
    columns = []
    for attribute, _, word, unit, scale in fields:
        values = getattr(point, attribute, None)
        if values is not None:
            columns.append((f"{word}{f' ({unit})' if unit else ''}", np.asarray(values, dtype=np.float64).reshape(-1) * scale))
    count = max((len(values) for _, values in columns), default=0)
    rows = [_point_heading(point) + f" · {count} servos", "",
            "| servo | " + " | ".join(title for title, _ in columns) + " |", "|---|" + "---:|" * len(columns)]  # fmt: skip
    for servo in range(count):
        rows.append(f"| {servo} | " + " | ".join(_num(values[servo], 4) for _, values in columns) + " |")
    rows.append("\n*A point carries no servo names: the row is the servo's index in its stream. Angles in degrees.*")
    return "\n".join(rows)


@log.register
def _(point: ServoStateTrajectoryPoint, name: str | None = None, *, recording: Any) -> None:
    name = name or "ServoStateTrajectoryPoint"
    _note(recording, f"notes/{part(name)}", _servo_point_table(point, SERVO_STATE_FIELDS, name))


@log.register
def _(point: ServoCommandTrajectoryPoint, name: str | None = None, *, recording: Any) -> None:
    name = name or "ServoCommandTrajectoryPoint"
    _note(recording, f"notes/{part(name)}", _servo_point_table(point, SERVO_COMMAND_FIELDS, name))


@log.register
def _(point: ImuTrajectoryPoint, name: str | None = None, *, recording: Any) -> None:
    log(point.imu(), name or "ImuTrajectoryPoint", recording=recording)


@log.register
def _(point: WrenchTrajectoryPoint, name: str | None = None, *, recording: Any) -> None:
    log(point.wrench, name or "WrenchTrajectoryPoint", recording=recording)


@log.register
def _(point: ImageTrajectoryPoint, name: str | None = None, *, recording: Any) -> None:
    log(point.image, name or "ImageTrajectoryPoint", recording=recording)


def _views_of(single: type) -> Any:
    """The single message's own views (Imu, Wrench, Image), looked up when called (its module registers it)."""
    return lambda name: list(view_for(single).views(name))  # type: ignore[union-attr]


def _peak_sample(traj: Any, values: np.ndarray) -> Any:
    return traj[int(np.argmax(np.linalg.norm(values.reshape(len(values), -1), axis=1)))]


register(TypeView(
    type=ServoStateTrajectoryPoint,
    summary="One sample of a drive log as a table: each servo's position (deg), velocity, effort, temperature, "
    "voltage and fault code.",
    sample=lambda: sim2real.servo_states("real")[60],
    views=lambda name: [_notes_view("", name)],
    tags=("text",),
))  # fmt: skip

register(TypeView(
    type=ServoCommandTrajectoryPoint,
    summary="One command as a table: each servo's position and velocity target (deg, deg/s), feed-forward, gains "
    "and effort cap.",
    sample=lambda: sim2real.servo_commands("real")[60],
    views=lambda name: [_notes_view("", name)],
    tags=("text",),
))  # fmt: skip

register(TypeView(
    type=ImuTrajectoryPoint,
    summary="One IMU sample, drawn as an Imu: the board turned by its orientation, its rates and acceleration.",
    sample=lambda: sim2real.imu("real")[60],
    views=_views_of(Imu),
    tags=("3d", "plot"),
))  # fmt: skip

register(TypeView(
    type=WrenchTrajectoryPoint,
    summary="One F/T sample, drawn as a Wrench: the force and torque arrows at the sensor, with their values.",
    sample=lambda: _peak_sample(sim2real.wrench("real"), np.asarray(sim2real.wrench("real").forces)),
    views=_views_of(Wrench),
    tags=("3d", "plot"),
))  # fmt: skip

register(TypeView(
    type=ImageTrajectoryPoint,
    summary="One frame of a camera stream, drawn as an Image.",
    sample=lambda: sim2real.camera("real")[8],
    views=_views_of(Image),
    tags=("2d",),
))  # fmt: skip
