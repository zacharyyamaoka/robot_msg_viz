"""Stock views of robot_msgs.compare types (docs/CONTRACT.md: one TypeView per type, registered on import).

Comparing trajectories is traj_compare's job ("never write a bespoke trajectory comparison viewer"), so this module
gives a stock view only to the messages that read clearly on their own:

- Comparison: the error signal (reference − measured) over time, one line per joint in degrees (a pose: x/y/z in mm),
  beside its metrics table in the same units.
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

from typing import Any

import numpy as np
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
from robot_msgs.trajectory import JointTrajectory, PoseTrajectory

from robot_msg_viz._engine import log, part
from robot_msg_viz.registry import TypeView, register
from robot_msg_viz.views._common import DEG, rng
from robot_msg_viz.views.trajectory import (
    AXIS_COLORS,
    COLOR_NAMES,
    EFFORT_UNIT,
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


def metric_unit(metric: str) -> tuple[float, str]:
    """(scale from SI, unit on screen) for a compare metric name; anything unknown stays in its SI unit, said so.

    Joint metrics read every joint as revolute (degrees): robot_msgs carries no joint type."""
    field = metric.split("/")[0]
    if field in _JOINT_FIELDS and "/" in metric:
        return _JOINT_FIELDS[field]
    return _POSE_METRICS.get(metric, (1.0, "SI"))


def _shown(value: float, metric: str, *, magnitude: bool = False) -> str:
    scale, unit = metric_unit(metric)
    value = abs(value) if magnitude else value
    return f"{_num(value * scale)} {unit}"


# ── Comparison: the error over time and its numbers ──────────────────────────────────────────────────────────────────


def _comparison_table(comparison: Comparison) -> str:
    error = comparison.error
    joints = [str(n) for n in error.joint_names] if isinstance(error, JointTrajectory) else []
    lines = [
        f"**{len(comparison.metrics)} metrics** · reference − measured · measured covers "
        f"{comparison.coverage:.0%} of the reference · time offset {_num(comparison.time_offset_s)} s · "
        f"{comparison.dropped} sample{'s' if comparison.dropped != 1 else ''} dropped",
        "",
        "| metric | line | unit | rms | p95 \\|e\\| | max \\|e\\| | final |",
        "|---|---|---|---|---|---|---|",
    ]
    for metric, stats in comparison.metrics.items():
        scale, unit = metric_unit(metric)
        line = ""
        if metric.startswith("positions/") and metric.split("/", 1)[1] in joints:
            index = joints.index(metric.split("/", 1)[1])
            line = COLOR_NAMES[index] if index < len(COLOR_NAMES) else "Rerun's"
        elif isinstance(error, PoseTrajectory) and metric in ("x", "y", "z"):
            line = ("red", "green", "blue")["xyz".index(metric)]
        cells = [_num(getattr(stats, key) * scale, 3) for key in ("rms", "p95_abs", "max_abs", "final")]
        lines.append(f"| {metric} | {line} | {unit} | " + " | ".join(cells) + " |")
    lines += ["", "*Joint angles in degrees, every joint read as revolute (robot_msgs carries no joint type).*"]
    return "\n".join(lines)


@log.register
def _(comparison: Comparison, name: str | None = None, *, recording: Any, **options: Any) -> None:
    error = comparison.error
    if options or not isinstance(error, (JointTrajectory, PoseTrajectory)):
        _stock(comparison, name, recording, **options)
        return
    name = name or "Comparison"
    seconds = _seconds(error)
    if isinstance(error, JointTrajectory):
        for field, (scale, unit) in _JOINT_FIELDS.items():
            values = getattr(error, field)
            if values is None:
                continue
            shown = np.asarray(values, dtype=np.float64) * scale
            for index, joint in enumerate(error.joint_names):
                _series(recording, f"error/{field}/{part(joint)}/{part(name)}", seconds, shown[:, index],
                        label=f"{joint} error ({unit}) · {name}", color=joint_color(index), width=2.0)  # fmt: skip
    else:
        positions = np.asarray(error.positions, dtype=np.float64) * 1000.0
        for axis, label in enumerate("xyz"):
            _series(recording, f"error/positions/{label}/{part(name)}", seconds, positions[:, axis],
                    label=f"{label} error (mm) · {name}", color=AXIS_COLORS[axis], width=2.0)  # fmt: skip
    _notes(recording, name, _comparison_table(comparison))


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
    summary=("The error (reference − measured) over time, one line per joint in degrees (a pose: x/y/z in mm), "
             "beside its metrics: rms, p95, max and final error per joint."),
    sample=pick_comparison,
    views=lambda name: [_plot("/error/positions", "position error, reference − measured", legend=False),
                        _notes_view(name, f"{name}: metrics")],  # fmt: skip
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
