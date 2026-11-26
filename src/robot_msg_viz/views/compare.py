"""Stock views of robot_msgs.compare types (docs/CONTRACT.md: one TypeView per type, registered on import).

Comparing trajectories is traj_compare's job ("never write a bespoke trajectory comparison viewer"), so this module
gives a stock view only to the messages that read clearly on their own:

- Comparison: the error signal (reference − measured) over time, one line per joint in degrees (a pose: x/y/z in mm),
  beside its metrics table in the same units. Every stream robot_msgs compares has its own layout (the page, when the
  Comparison is shown alone; runs.py reuses it for each comparison of a run):
  ServoState: a tab per field (position deg, velocity deg/s, effort N·m, temperature °C, voltage V), one line per
  joint, over a lane counting the servos whose fault codes differ. ServoCommand: a tab per target, gain and cap, and
  the latency error (ms). Imu: rotation (deg, its rotation vector), angular velocity (deg/s), acceleration (m/s²).
  Wrench: force (N) and torque (N·m). Image: the error frames as a dark-centred diverging heat map (blue: measured
  brighter, red: reference brighter) with its scale, beside mae / rmse (px) and psnr (dB) per frame. PointCloud: the
  measured cloud coloured by its distance to the reference (viridis, mm) with a colour bar. A joint or pose
  comparison opens exactly as before.
- Outcome: the verdict card: the result code in plain words, each violation, the room left (margins), in degrees.
- Violation: one broken tolerance as a sentence, in degrees.
- ErrorStats: the summary numbers of one error signal as a table, in the signal's own unit (the message carries none).

Left unsupported (listed greyed in the gallery), and why:
- Aggregate: the N-run accuracy vs repeatability analysis (ISO 9283) is a report over several runs, traj_compare's.
- GoalStats, TimeOffsetEstimate: a few scalars whose unit (rad or m) the message does not say; on their own a view
  could only restate the repr. traj_compare shows them beside the runs they came from.
- Tolerances, JointTolerance, PoseTolerance: pure configuration (what a run must stay inside); they show up as the
  bands on a comparison, which traj_compare draws.

WHY a Comparison override over robot_msgs' stock logger: the stock plots the error in radians with colours that follow
the entity path; here the joint error is in degrees (Zach's rule, and traj_compare's own /error/positions/<joint>
layout is in degrees too, so the two overlay) in each joint's colour from the JointTrajectory view.
"""

from __future__ import annotations

# WHY empty (2026-11-25): this module registers its TypeViews on import; its functions are sample data and helpers,
# not API. The public API is robot_msg_viz's own __all__ (show/save/stream/blueprint), and robot-sdk's scanner reads
# __all__ to decide what a package publishes.
__all__: list[str] = []

from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from robot_msgs.compare import (
    Comparison,
    ErrorStats,
    JointTolerance,
    Outcome,
    Tolerances,
    Violation,
    check,
    compare,
    error_stats,
)
from robot_msgs.sensor import PointCloud
from robot_msgs.trajectory import (
    ImageTrajectory,
    ImuTrajectory,
    JointTrajectory,
    PoseTrajectory,
    ServoCommandTrajectory,
    ServoStateTrajectory,
    Trajectory,
    WrenchTrajectory,
    sample_fields,
)

from robot_msg_viz._engine import log, part
from robot_msg_viz.layout import arrange
from robot_msg_viz.registry import TypeView, register
from robot_msg_viz.views._common import DEG, rng
from robot_msg_viz.views.sensor import CLOUD_POINT_RADIUS, INK, _ramp
from robot_msg_viz.views.streams import _panels, _tabs, log_frames, series_group, side_by_side
from robot_msg_viz.views.trajectory import (
    AXIS_COLORS,
    COLOR_NAMES,
    EFFORT_UNIT,
    JOINT_COLORS,
    _notes,
    _notes_view,
    _num,
    _plot,
    _seconds,
    _series,
    _stock,
    joint_color,
    pick_move,
    tool_path,
)

# ── units: a metric's name says what it measures (robot_msgs.compare's names) ────────────────────────────────────────

_JOINT_FIELDS = {"positions": (DEG, "deg"), "velocities": (DEG, "deg/s"), "accelerations": (DEG, "deg/s²"),
                 "effort": (1.0, EFFORT_UNIT)}  # fmt: skip
_POSE_METRICS = {
    "position": (1000.0, "mm"), "x": (1000.0, "mm"), "y": (1000.0, "mm"), "z": (1000.0, "mm"),
    "lag": (1000.0, "mm"), "cross_track": (1000.0, "mm"), "contour": (1000.0, "mm"),
    "rotation": (DEG, "deg"), "rx": (DEG, "deg"), "ry": (DEG, "deg"), "rz": (DEG, "deg"),
    "linear_velocity": (1000.0, "mm/s"), "angular_velocity": (DEG, "deg/s"), "lag_time": (1000.0, "ms"),
}  # fmt: skip


# robot_msgs DESIGN rule 18's stream metrics: ``<field>/<joint>`` of a servo stream, ``<field>/x|y|z`` of an IMU or a
# wrench, and the flat names of the IMU, wrench, image and cloud comparisons
_SERVO_FIELDS = {
    "temperature": (1.0, "°C"), "voltage": (1.0, "V"), "feedforward_effort": (1.0, EFFORT_UNIT),
    "stiffness": (1.0, "N·m/rad"), "damping": (1.0, "N·m·s/rad"), "max_effort": (1.0, EFFORT_UNIT),
    "fault_mismatch": (1.0, "0/1"),
    "angular_velocities": (DEG, "deg/s"), "linear_accelerations": (1.0, "m/s²"),
    "forces": (1.0, "N"), "torques": (1.0, "N·m"),
}  # fmt: skip
_STREAM_METRICS = {
    "linear_acceleration": (1.0, "m/s²"), "force": (1.0, "N"), "torque": (1.0, "N·m"),
    "fault_mismatch": (1.0, "0/1"), "channels/latency": (1000.0, "ms"),
    "pixels": (1.0, "px"), "mae": (1.0, "px"), "rmse": (1.0, "px"), "psnr": (1.0, "dB"),
    "distance": (1000.0, "mm"), "distance_reverse": (1000.0, "mm"), "chamfer": (1000.0, "mm"),
}  # fmt: skip


def metric_unit(metric: str) -> tuple[float, str]:
    """(scale from SI, unit on screen) for a compare metric name; anything unknown stays in its SI unit, said so.

    Joint metrics read every joint as revolute (degrees): robot_msgs carries no joint type. Image metrics are in the
    frames' own pixel units (``px``: a 0-255 scale for 8-bit frames, metres for depth)."""
    field = metric.split("/")[0]
    if field in _JOINT_FIELDS and "/" in metric:
        return _JOINT_FIELDS[field]
    if field in _SERVO_FIELDS and "/" in metric:
        return _SERVO_FIELDS[field]
    if metric in _STREAM_METRICS:
        return _STREAM_METRICS[metric]
    return _POSE_METRICS.get(metric, (1.0, "SI"))


def _shown(value: float, metric: str, *, magnitude: bool = False) -> str:
    scale, unit = metric_unit(metric)
    value = abs(value) if magnitude else value
    return f"{_num(value * scale)} {unit}"


# ── Comparison: the error over time and its numbers ──────────────────────────────────────────────────────────────────


def _joint_line(metric: str, joints: list[str]) -> str:
    """The colour name of a ``<field>/<joint>`` metric's line (the JointTrajectory view's per-joint colours)."""
    field, _, joint = metric.partition("/")
    if not joint or joint not in joints:
        return ""
    index = joints.index(joint)
    return COLOR_NAMES[index] if index < len(COLOR_NAMES) else "Rerun's"


_AXIS_LINE = {"x": "red", "y": "green", "z": "blue", "rx": "red", "ry": "green", "rz": "blue"}
_REVOLUTE = "*Joint angles in degrees, every joint read as revolute (robot_msgs carries no joint type).*"
_FOOTERS = {
    ServoStateTrajectory: _REVOLUTE + " *fault_mismatch: 1 where the two fault codes differ (its mean is the share "
    "of samples).*",
    ServoCommandTrajectory: _REVOLUTE,
    ImuTrajectory: "*rotation: the angle of the rotation from measured to reference (rx ry rz its rotation vector), "
    "in degrees.*",
    WrenchTrajectory: "*In the sensor's frame.*",
    ImageTrajectory: "*In the frames' pixel units (a 0-255 scale for 8-bit; metres for depth). pixels: every pixel of "
    "every frame; mae / rmse / psnr: per frame, over t.*",
    PointCloud: "*distance: each measured point to its nearest reference point; distance_reverse the other way; "
    "chamfer both (its max is the Hausdorff distance).*",
}


def _line(metric: str, error: Any, joints: list[str]) -> str:
    if isinstance(error, JointTrajectory):
        return _joint_line(metric, joints) if metric.startswith("positions/") else ""
    if isinstance(error, PoseTrajectory):
        return _AXIS_LINE[metric] if metric in ("x", "y", "z") else ""
    if joints:
        return _joint_line(metric, joints)
    axis = metric.rpartition("/")[2]
    return _AXIS_LINE.get(axis, "") if "/" in metric or metric in ("rx", "ry", "rz") else ""


def _comparison_table(comparison: Comparison) -> str:
    error = comparison.error
    joints = [str(n) for n in getattr(error, "joint_names", ())]
    timed = isinstance(error, Trajectory)
    head = f"**{len(comparison.metrics)} metrics** · reference − measured"
    if timed:
        head += (f" · measured covers {comparison.coverage:.0%} of the reference · time offset "
                 f"{_num(comparison.time_offset_s)} s · {comparison.dropped} sample{'s' if comparison.dropped != 1 else ''}"
                 " dropped")  # fmt: skip
    else:
        head += f" · {len(error)} measured points against the reference cloud (no clock)"
    lines = [head, "", "| metric | line | unit | rms | p95 \\|e\\| | max \\|e\\| | final |", "|---|---|---|---|---|---|---|"]
    for metric, stats in comparison.metrics.items():
        scale, unit = metric_unit(metric)
        cells = [_num(getattr(stats, key) * scale, 3) for key in ("rms", "p95_abs", "max_abs", "final")]
        lines.append(f"| {metric} | {_line(metric, error, joints)} | {unit} | " + " | ".join(cells) + " |")
    mismatch = comparison.metrics.get("fault_mismatch")
    if mismatch is not None:
        worst = [(m.split("/", 1)[1], s.mean) for m, s in comparison.metrics.items() if m.startswith("fault_mismatch/")]
        worst = [(joint, share) for joint, share in worst if share > 0]
        lines[1:1] = ["", f"**Fault codes differ on {mismatch.mean:.1%} of samples**"
                  + (": " + ", ".join(f"{joint} {share:.1%}" for joint, share in worst) if worst else "")]  # fmt: skip
    lines += ["", _FOOTERS.get(type(error), _REVOLUTE)]
    return "\n".join(lines)


# ── Comparison: every error kind ─────────────────────────────────────────────────────────────────────────────────────

# (field on the error stream, entity-path field, title, unit, scale from SI)
_STATE_ERRORS = (
    ("positions", "positions", "position error", "deg", DEG),
    ("velocities", "velocities", "velocity error", "deg/s", DEG),
    ("effort", "effort", "effort error", EFFORT_UNIT, 1.0),
    ("temperature", "temperature", "temperature error", "°C", 1.0),
    ("voltage", "voltage", "voltage error", "V", 1.0),
)
_COMMAND_ERRORS = (
    ("positions", "positions", "position target error", "deg", DEG),
    ("velocities", "velocities", "velocity target error", "deg/s", DEG),
    ("feedforward_effort", "feedforward_effort", "feed-forward error", EFFORT_UNIT, 1.0),
    ("stiffness", "stiffness", "stiffness error", "N·m/rad", 1.0),
    ("damping", "damping", "damping error", "N·m·s/rad", 1.0),
    ("max_effort", "max_effort", "effort cap error", EFFORT_UNIT, 1.0),
)
_COUNT = (200, 200, 205)  # the "how many servos disagree" line: neutral, not a joint's colour


def _joint_errors(error: Any, fields: Any, recording: Any, root: str, name: str) -> list[tuple[str, str, str]]:
    seconds = _seconds(error)
    logged = []
    for attribute, field, title, unit, scale in fields:
        values = getattr(error, attribute, None)
        if values is None:
            continue
        _joint_group(recording, root, field, name, seconds, np.asarray(values, dtype=np.float64) * scale,
                     error.joint_names, unit)  # fmt: skip
        logged.append((field, title, unit))
    return logged


def _joint_group(recording: Any, root: str, field: str, name: str, seconds: np.ndarray, shown: np.ndarray,
                 joints: Any, unit: str) -> None:  # fmt: skip
    """One line per joint in its colour (`series_group`: per-joint paths at the top level, one entity when nested)."""
    names = [str(joint) for joint in joints]
    series_group(recording, root, f"error/{field}", part(name), seconds, shown, names,
                 [f"{joint} error ({unit}) · {name}" for joint in names],
                 [joint_color(index) for index in range(len(names))], 2.0)  # fmt: skip


def _xyz_errors(recording: Any, root: str, name: str, seconds: np.ndarray, groups: Any) -> list[tuple[str, str, str]]:
    logged = []
    for field, title, columns, values, unit in groups:
        series_group(recording, root, f"error/{field}", part(name), seconds, values, columns,
                     [f"{column} error ({unit}) · {name}" for column in columns], list(AXIS_COLORS), 2.0)  # fmt: skip
        logged.append((field, title, unit))
    return logged


def _diverging(values: np.ndarray, limit: float) -> np.ndarray:
    """Signed values on a dark-centred blue → near-black → red scale, ±`limit` at the ends (dark canvas)."""
    unit = np.clip(np.nan_to_num(values / limit), -1.0, 1.0)[..., None]
    middle, low, high = np.array(_MIDDLE, float), np.array(_LOW, float), np.array(_HIGH, float)
    out = np.where(unit < 0, middle + (low - middle) * -unit, middle + (high - middle) * unit)
    return np.round(out).astype(np.uint8)


_LOW, _MIDDLE, _HIGH = (80, 150, 255), (28, 30, 34), (255, 105, 70)


def _colour_bar(recording: Any, path: str, colours: np.ndarray, labels: tuple[str, str, str]) -> None:
    """A 256-step colour bar image at `path` with three labels under it (low, middle, high)."""
    bar = np.repeat(colours[None, :, :], 18, axis=0)
    recording.log(f"{path}/bar", rr.Image(bar, color_model="RGB"), static=True)
    recording.log(f"{path}/labels", rr.Points2D([[2.0, 30.0], [128.0, 30.0], [253.0, 30.0]], radii=0.5,
                                                colors=[INK], labels=list(labels), show_labels=True), static=True)  # fmt: skip


def _bar_view(path: str, title: str) -> rrb.View:
    return rrb.Spatial2DView(origin=f"/{path}", name=title,
                             visual_bounds=rrb.VisualBounds2D(x_range=[-60.0, 316.0], y_range=[-8.0, 44.0]))  # fmt: skip


def _log_image_error(error: ImageTrajectory, recording: Any, root: str, name: str) -> None:
    seconds = _seconds(error)
    frames = np.asarray(error.frames, dtype=np.float64)
    signed = frames.mean(axis=-1) if frames.ndim == 4 else frames  # one signed value per pixel
    finite = np.abs(signed[np.isfinite(signed)])
    limit = float(np.percentile(finite, 99.5)) if finite.size and finite.max() > 0 else 1.0
    picture = np.stack([_diverging(frame, limit) for frame in signed])
    log_frames(picture, seconds, f"{root}image/{part(name)}/error", recording, encoding="rgb8")
    ramp = _diverging(np.linspace(-limit, limit, 256), limit)
    _colour_bar(recording, f"{root}image/{part(name)}_scale", ramp,
                (f"−{_num(limit, 3)}: measured brighter", "0", f"+{_num(limit, 3)}: reference brighter"))  # fmt: skip
    for key, field, unit in (("mae", "pixel/mae", "px"), ("rmse", "pixel/rmse", "px"), ("psnr", "psnr", "dB")):
        values = (error.channels or {}).get(key)
        if values is not None:
            _series(recording, f"{root}error/{field}/{part(name)}", seconds, np.asarray(values, dtype=np.float64),
                    label=f"{key} ({unit}) · {name}", color={"mae": JOINT_COLORS[0], "rmse": JOINT_COLORS[1],
                                                             "psnr": JOINT_COLORS[2]}[key], width=2.0)  # fmt: skip


def _log_cloud_error(error: PointCloud, recording: Any, root: str, name: str) -> None:
    distance = np.asarray(error.channels["distance"], dtype=np.float64) * 1000.0
    finite = distance[np.isfinite(distance)]
    top = float(np.percentile(finite, 99.0)) if finite.size else 1.0
    top = top if top > 0 else 1.0
    colours = _ramp(np.concatenate([np.clip(distance, 0.0, top), [0.0, top]]))[:-2]
    recording.log(f"{root}world/{part(name)}/measured",
                  rr.Points3D(np.asarray(error.points), colors=colours, radii=CLOUD_POINT_RADIUS), static=True)  # fmt: skip
    _colour_bar(recording, f"{root}image/{part(name)}_distance", _ramp(np.linspace(0.0, top, 256)),
                ("0 mm: on the reference", f"{_num(top / 2, 3)} mm", f"{_num(top, 3)} mm and beyond"))  # fmt: skip


def _log_generic_error(error: Trajectory, recording: Any, root: str, name: str) -> None:
    """Any other trajectory error: every numeric sample field and channel, SI, by column."""
    seconds = _seconds(error)
    arrays = {f.name: getattr(error, f.name) for f in sample_fields(type(error))}
    arrays.update({f"channels/{key}": value for key, value in (error.channels or {}).items()})
    for field, raw in arrays.items():
        if raw is None:
            continue
        values = np.asarray(raw)
        if values.dtype.kind not in "fiub" or values.shape[:1] != seconds.shape:
            continue
        flat = values.reshape(len(seconds), -1).astype(np.float64)
        for column in range(flat.shape[1]):
            _series(recording, f"{root}error/{field}/{column}/{part(name)}", seconds, flat[:, column],
                    label=f"{field}[{column}] error · {name}")  # fmt: skip


def log_comparison(comparison: Comparison, recording: Any, *, root: str = "", name: str = "Comparison") -> None:
    """Log one Comparison under `root` (runs.py nests a run's under ``/runs/<run>/<pair>/<stream>/``)."""
    error = comparison.error
    seconds = _seconds(error) if isinstance(error, Trajectory) else None
    if isinstance(error, JointTrajectory):
        for field, (scale, unit) in _JOINT_FIELDS.items():
            values = getattr(error, field)
            if values is None:
                continue
            _joint_group(recording, root, field, name, seconds, np.asarray(values, dtype=np.float64) * scale,
                         error.joint_names, unit)  # fmt: skip
    elif isinstance(error, PoseTrajectory):
        positions = np.asarray(error.positions, dtype=np.float64) * 1000.0
        series_group(recording, root, "error/positions", part(name), seconds, positions, ("x", "y", "z"),
                     [f"{label} error (mm) · {name}" for label in "xyz"], list(AXIS_COLORS), 2.0)  # fmt: skip
    elif isinstance(error, ServoStateTrajectory):
        _joint_errors(error, _STATE_ERRORS, recording, root, name)
        mismatch = (error.channels or {}).get("fault_mismatch")
        if mismatch is not None:
            mismatch = np.asarray(mismatch, dtype=np.float64).reshape(len(seconds), -1)
            _series(recording, f"{root}error/fault_mismatch/servos/{part(name)}", seconds, mismatch.sum(axis=1),
                    label=f"servos whose fault codes differ · {name}", color=_COUNT, width=1.5)  # fmt: skip
            for index, joint in enumerate(error.joint_names):
                if mismatch[:, index].any():
                    _series(recording, f"{root}error/fault_mismatch/{part(joint)}/{part(name)}", seconds,
                            mismatch[:, index], label=f"{joint} differs (0/1) · {name}", color=joint_color(index),
                            width=2.0)  # fmt: skip
    elif isinstance(error, ServoCommandTrajectory):
        _joint_errors(error, _COMMAND_ERRORS, recording, root, name)
        latency = (error.channels or {}).get("latency")
        if latency is not None:
            _series(recording, f"{root}error/latency/{part(name)}", seconds,
                    np.asarray(latency, dtype=np.float64) * 1000.0, label=f"latency error (ms) · {name}",
                    color=_COUNT, width=1.5)  # fmt: skip
    elif isinstance(error, ImuTrajectory):
        rotation = (error.channels or {}).get("rotation_vector")
        groups = []
        if rotation is not None:
            groups.append(("rotation", "rotation error (deg)", ("rx", "ry", "rz"),
                           np.asarray(rotation, dtype=np.float64) * DEG, "deg"))  # fmt: skip
        groups += [
            ("angular_velocity", "angular velocity error (deg/s)", ("x", "y", "z"),
             np.asarray(error.angular_velocities, dtype=np.float64) * DEG, "deg/s"),
            ("linear_acceleration", "acceleration error (m/s²)", ("x", "y", "z"),
             np.asarray(error.linear_accelerations, dtype=np.float64), "m/s²"),
        ]  # fmt: skip
        _xyz_errors(recording, root, name, seconds, groups)
        angle = (error.channels or {}).get("rotation_angle")
        if angle is not None:
            _series(recording, f"{root}error/rotation/angle/{part(name)}", seconds,
                    np.asarray(angle, dtype=np.float64) * DEG, label=f"angle (deg) · {name}", color=_COUNT,
                    width=1.5)  # fmt: skip
    elif isinstance(error, WrenchTrajectory):
        _xyz_errors(recording, root, name, seconds, [
            ("force", "force error (N)", ("x", "y", "z"), np.asarray(error.forces, dtype=np.float64), "N"),
            ("torque", "torque error (N·m)", ("x", "y", "z"), np.asarray(error.torques, dtype=np.float64), "N·m"),
        ])  # fmt: skip
    elif isinstance(error, ImageTrajectory):
        _log_image_error(error, recording, root, name)
    elif isinstance(error, PointCloud):
        _log_cloud_error(error, recording, root, name)
    elif isinstance(error, Trajectory):
        _log_generic_error(error, recording, root, name)
    recording.log(f"{root}notes/{part(name)}", rr.TextDocument(_comparison_table(comparison),
                                                               media_type="text/markdown"), static=True)  # fmt: skip


@log.register
def _(comparison: Comparison, name: str | None = None, *, recording: Any, **options: Any) -> None:
    error = comparison.error
    known = (JointTrajectory, PoseTrajectory, ServoStateTrajectory, ServoCommandTrajectory, ImuTrajectory,
             WrenchTrajectory, ImageTrajectory, PointCloud)  # fmt: skip
    if options or not isinstance(error, known):
        _stock(comparison, name, recording, **options)
        return
    log_comparison(comparison, recording, name=name or "Comparison")


def _error_plot(root: str, field: str, title: str, *, legend: bool) -> rrb.View:
    return _plot(f"/{root}error/{field}", title, legend=legend)


def comparison_layout(comparison: Comparison, root: str = "", name: str = "Comparison", title: str | None = None) -> Any:
    """The container that shows one Comparison logged by `log_comparison` under `root`: its error plots (or error
    frames, or coloured cloud) beside its metrics table."""
    error = comparison.error
    notes = rrb.TextDocumentView(origin=f"/{root}notes/{part(name)}", name="metrics")
    if isinstance(error, JointTrajectory):
        words = {"positions": "position", "velocities": "velocity", "accelerations": "acceleration", "effort": "effort"}
        plots = [_error_plot(root, field, f"{words[field]} error ({unit}), reference − measured", legend=False)
                 for field, (_, unit) in _JOINT_FIELDS.items() if getattr(error, field) is not None]  # fmt: skip
        return side_by_side(_tabs(plots), notes, name=title)
    if isinstance(error, PoseTrajectory):
        plot = _error_plot(root, "positions", "position error (mm), reference − measured", legend=True)
        return side_by_side(plot, notes, name=title)
    if isinstance(error, (ServoStateTrajectory, ServoCommandTrajectory)):
        fields = _STATE_ERRORS if isinstance(error, ServoStateTrajectory) else _COMMAND_ERRORS
        plots = [_error_plot(root, field, f"{title} ({unit})", legend=False)
                 for attribute, field, title, unit, _ in fields if getattr(error, attribute) is not None]  # fmt: skip
        lane = None
        if isinstance(error, ServoStateTrajectory) and (error.channels or {}).get("fault_mismatch") is not None:
            lane = _error_plot(root, "fault_mismatch", "fault codes differ (servos)", legend=True)
        elif isinstance(error, ServoCommandTrajectory) and (error.channels or {}).get("latency") is not None:
            plots.append(_error_plot(root, "latency", "latency error (ms)", legend=False))
        main = _tabs(plots) if lane is None else rrb.Vertical(_tabs(plots), lane, row_shares=[3, 1])
        return side_by_side(main, notes, name=title)
    if isinstance(error, ImuTrajectory):
        plots = [_error_plot(root, "rotation", "rotation error (deg)", legend=True),
                 _error_plot(root, "angular_velocity", "angular velocity error (deg/s)", legend=True),
                 _error_plot(root, "linear_acceleration", "acceleration error (m/s²)", legend=True)]  # fmt: skip
        return side_by_side(rrb.Vertical(*plots), notes, name=title)
    if isinstance(error, WrenchTrajectory):
        plots = [_error_plot(root, "force", "force error (N)", legend=True),
                 _error_plot(root, "torque", "torque error (N·m)", legend=True)]  # fmt: skip
        return side_by_side(rrb.Vertical(*plots), notes, name=title)
    if isinstance(error, ImageTrajectory):
        picture = rrb.Vertical(
            rrb.Spatial2DView(origin=f"/{root}image/{part(name)}/error", name="error frame, reference − measured"),
            _bar_view(f"{root}image/{part(name)}_scale", "scale (mean over channels)"),
            row_shares=[6, 1],
        )
        plots = rrb.Vertical(_error_plot(root, "pixel", "mae / rmse per frame (px)", legend=True),
                             _error_plot(root, "psnr", "psnr per frame (dB)", legend=True))  # fmt: skip
        return side_by_side(picture, rrb.Vertical(plots, notes, row_shares=[1, 1]), name=title)
    if isinstance(error, PointCloud):
        points = np.asarray(error.points, dtype=np.float64)
        centre = (points.min(axis=0) + points.max(axis=0)) / 2 if len(points) else np.zeros(3)
        size = float(np.ptp(points, axis=0).max()) if len(points) else 1.0
        eye = rrb.EyeControls3D(position=tuple(centre + size * np.array([-0.15, -0.85, 0.75])),
                                look_target=tuple(centre), eye_up=(0.0, 0.0, 1.0))  # fmt: skip
        cloud = rrb.Spatial3DView(origin=f"/{root}world/{part(name)}", name="measured, coloured by distance",
                                  eye_controls=eye)  # fmt: skip
        side = rrb.Vertical(_bar_view(f"{root}image/{part(name)}_distance", "distance to the reference (mm)"), notes,
                            row_shares=[1, 4])  # fmt: skip
        return side_by_side(cloud, side, name=title)
    return side_by_side(_plot(f"/{root}error", "error (SI)"), notes, name=title)


def _comparison_page(comparison: Comparison, name: str, **options: Any) -> rrb.Blueprint:
    if isinstance(comparison.error, (JointTrajectory, PoseTrajectory)) or options:
        # WHY arrange, as before this page existed: a joint or pose comparison opens exactly as it always has
        return arrange(_comparison_views(name), timed=False)
    return _panels(comparison_layout(comparison, "", name))


def _comparison_views(name: str) -> list[Any]:
    return [_plot("/error/positions", "position error, reference − measured", legend=False),
            _notes_view(name, f"{name}: metrics")]  # fmt: skip


def tracked_run() -> JointTrajectory:
    """What the arm actually did on the pick: each joint lags its command (a 12 ms first-order servo), the shoulder
    lift and elbow sag a little under gravity, and the encoders read with 0.01° of noise; 100 Hz positions only."""
    target = pick_move()
    dt = np.diff(target.t)
    command = np.asarray(target.positions)
    measured = np.empty_like(command)
    measured[0] = command[0]
    for k, step in enumerate(dt):
        measured[k + 1] = measured[k] + min(step / 0.012, 1.0) * (command[k] - measured[k])
    sag = np.radians([0.0, -0.15, 0.08, 0.0, 0.0, 0.0])
    measured = measured + sag + np.radians(rng(11).normal(0.0, 0.01, measured.shape))
    return JointTrajectory(t=target.t, joint_names=target.joint_names, positions=measured, header=target.header)


def pick_comparison() -> Comparison:
    return compare(pick_move(), tracked_run())


register(TypeView(
    type=Comparison,
    summary=("The error (reference − measured) over time, one line per joint in degrees (a pose: x/y/z in mm; a "
             "servo, IMU or F/T stream: each field in its unit; a camera: the error frames as a heat map; a point "
             "cloud: the points coloured by distance in mm), beside its metrics: rms, p95, max and final."),
    sample=pick_comparison,
    views=_comparison_views,
    page=_comparison_page,
    tags=("plot", "text"),
))  # fmt: skip


# ── Outcome and Violation: the verdict ───────────────────────────────────────────────────────────────────────────────

_VERDICT = {"SUCCESSFUL": "Succeeded: every tolerance held.",
            "PATH_TOLERANCE_VIOLATED": "Aborted: a path tolerance was broken while the reference ran.",
            "GOAL_TOLERANCE_VIOLATED": "Failed: the goal tolerance was not met after the reference ended."}  # fmt: skip


def violation_sentence(violation: Violation) -> str:
    """``path: positions/j1 error 0.62 deg, past its 0.57 deg limit, at t = 0.3 s`` (a joint's error keeps its sign)."""
    magnitude = "/" not in violation.metric  # a pose's position / rotation error is a magnitude
    return (f"{violation.phase}: {violation.metric} error {_shown(violation.error, violation.metric, magnitude=magnitude)}"
            f", past its {_shown(violation.limit, violation.metric)} limit, at t = {_num(violation.t)} s")  # fmt: skip


@log.register
def _(outcome: Outcome, name: str | None = None, *, recording: Any) -> None:
    lines = [f"### {outcome.code_name} ({outcome.error_code})", "",
             _VERDICT.get(outcome.code_name, "A result code outside FollowJointTrajectory's three.")]  # fmt: skip
    if outcome.violations:
        lines += ["", "**First violation per limit**", ""]
        lines += [f"- {violation_sentence(v)}" for v in outcome.violations]
    if outcome.margins:
        lines += ["", "**Room left** (limit − worst |error|; negative = past it by that much)", "",
                  "| phase | metric | margin |", "|---|---|---|"]  # fmt: skip
        for key, margin in outcome.margins.items():
            phase, _, metric = key.partition("/")
            lines.append(f"| {phase} | {metric} | {_shown(margin, metric)} |")
    if np.isfinite(outcome.goal_time):
        goal = f"Goal met {_num(outcome.goal_time)} s after the reference ended."
    elif outcome.code_name == "PATH_TOLERANCE_VIOLATED":
        goal = "Goal never judged: the path breach ended the check."
    else:
        goal = "Goal not met (no goal time)."
    lines += ["", goal]
    if outcome.unchecked:
        lines.append(f"Not checked (the data could not say): {', '.join(outcome.unchecked)}.")
    if outcome.error_string:
        lines += ["", f"As reported (SI): `{outcome.error_string}`"]
    lines += ["", "*Joint angles in degrees, every joint read as revolute (robot_msgs carries no joint type).*"]
    _notes(recording, name or "Outcome", "\n".join(lines))


def pick_tolerances() -> Tolerances:
    """A cell's tolerances for the pick: 1° on every joint while moving, 0.25° once arrived, within 0.3 s."""
    joints = pick_move().joint_names
    return Tolerances(
        path=tuple(JointTolerance(name=joint, position=np.radians(1.0)) for joint in joints),
        goal=tuple(JointTolerance(name=joint, position=np.radians(0.25)) for joint in joints),
        goal_time_tolerance=0.3,
    )


def pick_outcome() -> Outcome:
    return check(pick_move(), tracked_run(), pick_tolerances())


register(TypeView(
    type=Outcome,
    summary=("The verdict: the result code in plain words, the first violation of each limit, and the room left "
             "under every tolerance, in degrees."),
    sample=pick_outcome,
    views=lambda name: [_notes_view(name)],
    tags=("text",),
))  # fmt: skip


@log.register
def _(violation: Violation, name: str | None = None, *, recording: Any) -> None:
    lines = [f"### {violation_sentence(violation)}", "",
             f"- the {violation.phase} tolerance on {violation.metric}; the error is reference − measured",
             "", "*Joint angles in degrees, every joint read as revolute (robot_msgs carries no joint type).*"]  # fmt: skip
    _notes(recording, name or "Violation", "\n".join(lines))


register(TypeView(
    type=Violation,
    summary="One broken tolerance as a sentence: which limit, the error and the limit in degrees, and when.",
    sample=lambda: pick_outcome().violations[0],
    views=lambda name: [_notes_view(name)],
    tags=("text",),
))  # fmt: skip


# ── ErrorStats: the numbers of one error signal ──────────────────────────────────────────────────────────────────────

_STATS_ROWS = (
    ("n", "samples", ""),
    ("rms", "RMS", "u"),
    ("mean", "mean (signed)", "u"),
    ("median", "median (signed)", "u"),
    ("std", "standard deviation", "u"),
    ("min", "min (signed)", "u"),
    ("max", "max (signed)", "u"),
    ("max_abs", "max |e|", "u"),
    ("p95_abs", "95th percentile |e|", "u"),
    ("final", "final (signed)", "u"),
    ("iae", "IAE  ∫|e| dt", "u·s"),
    ("ise", "ISE  ∫e² dt", "u²·s"),
    ("itae", "ITAE  ∫t·|e| dt", "u·s²"),
)


@log.register
def _(stats: ErrorStats, name: str | None = None, *, recording: Any) -> None:
    lines = ["**One error signal, summarised** · u = the signal's own unit (an ErrorStats does not say which)", "",
             "| | value | unit |", "|---|---|---|"]  # fmt: skip
    for key, label, unit in _STATS_ROWS:
        value = getattr(stats, key)
        # WHY escaped: a bare | inside a cell (|e|) splits the markdown table
        cell = label.replace("|", "\\|")
        lines.append(f"| {cell} | {value if key == 'n' else _num(value, 4)} | {unit} |")
    lines += ["", "— = not available (no time axis was given, or no finite samples)."]
    _notes(recording, name or "ErrorStats", "\n".join(lines))


def tool_error_stats() -> ErrorStats:
    """The tool's position error on the pick, in metres: how far the measured tool point was from the planned one."""
    planned = tool_path()
    sag = np.array([0.0, 0.0, -0.0006])  # the tool sits 0.6 mm low, plus 0.2 mm of noise per axis
    measured = np.asarray(planned.positions) + sag + rng(5).normal(0.0, 0.0002, (len(planned), 3))
    error = np.linalg.norm(np.asarray(planned.positions) - measured, axis=1)
    return error_stats(error, planned.t)


register(TypeView(
    type=ErrorStats,
    summary="The numbers of one error signal as a table: RMS, mean, median, spread, worst, p95, final and the integrals.",
    sample=tool_error_stats,
    views=lambda name: [_notes_view(name)],
    tags=("text",),
))  # fmt: skip
