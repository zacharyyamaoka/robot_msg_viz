"""Stock views of robot_msgs.control types (docs/CONTRACT.md: one TypeView per type, registered on import).

    ServoState            per-servo position, velocity and effort plots, and a table of everything at the playhead
    ServoCommand          the same plots for the targets (overlay a ServoState under another name), and its table
    JointCommandSequence  the routine as lanes over its steps: one lane per joint, coloured by what it does,
                          a lane saying when each step ends, and the sequence's speed guard
    JointCommandStep      one step, drawn as a one-step sequence

Not supported, and why: JointCommand and the done-when conditions (Elapsed, Settled, Reached, Stall,
EffortStall) are a step's parameters, a few numbers with nothing of their own to draw; each one reads in
plain words inside its step's lanes above, which is where it means something.

robot_msgs' stock logger draws none of these types (each falls to its repr), so each gets a logger here.
Entity paths (docs/CONTRACT.md's `/<field>/<column>/<name>`, robot_msgs' trajectory field names, so a
ServoState overlays a JointTrajectory or a ServoCommand logged under another name in the same plot):

    /positions/<servo>/<name>     deg (rotary) or m          /velocities/<servo>/<name>   deg/s or m/s
    /effort/<servo>/<name>        N·m or N (a command's: its feed-forward)
    /temperature/<servo>/<name>   °C    /voltage/<servo>/<name>   V    /fault/<servo>/<name>   the board's code
    /latency/<name>               ms from the state a command was computed from to sending it
    /notes/<name>                 the table at the playhead
    /steps/<name>[/joints/<joint>|/step|/until]   a sequence's lanes, on the `step` timeline (1, 2, ... in order)

WHY degrees by default: Zach's rule, angles on screen in degrees. robot_msgs carries no servo or joint type,
so every servo is taken as rotary; pass ``degrees=`` the rotary names (or False) to `log` when some are linear.
"""

from __future__ import annotations

# WHY empty (2026-11-25): this module registers its TypeViews on import; its functions are sample data and helpers,
# not API. The public API is robot_msg_viz's own __all__ (show/save/stream/blueprint), and robot-sdk's scanner reads
# __all__ to decide what a package publishes.
__all__: list[str] = []

import math
from collections.abc import Collection, Iterable
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from robot_msgs.control import (
    EffortStall,
    Elapsed,
    JointCommand,
    JointCommandSequence,
    JointCommandStep,
    Reached,
    ServoCommand,
    ServoState,
    Settled,
    Stall,
)
from robot_msgs.std import Header

from robot_msg_viz._engine import log, part
from robot_msg_viz.registry import TypeView, register

DEG = 180.0 / math.pi
STEP_TIMELINE = "step"  # a sequence's lanes: step 1, 2, ... (its order, not its timing, which a run measures)

# One colour per servo, the same in every plot, so a servo is recognisable across position, velocity and effort.
# WHY not Rerun's automatic colours: those hash the entity path, which differs per field.
_PALETTE = (
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
)  # fmt: skip
_EPOCH = 1_795_521_600.0  # 2026-11-24 12:00:00 UTC: sample stamps read like a real log


def _rotary(names: Iterable[str], degrees: bool | Collection[str]) -> set[str]:
    """The names shown in degrees: all of them (True), none (False) or the ones given."""
    names = [str(n) for n in names]
    if degrees is True:
        return set(names)
    if not degrees:
        return set()
    return {str(n) for n in degrees}


def _label(servo: str, name: str, default: str) -> str:
    """A legend entry: the servo alone under the type's own name, ``servo · name`` when the caller named it."""
    return servo if name == default else f"{servo} · {name}"


def _number(value: float) -> str:
    """A table cell: 4 significant digits, NaN spelled out, and never a "-0"."""
    return "NaN" if math.isnan(value) else f"{value + 0.0:.4g}"


def _series(
    recording: rr.RecordingStream, path: str, value: float, legend: str, colour: tuple[int, ...], width: float
) -> None:
    recording.log(path, rr.Scalars([value]), rr.SeriesLines(names=legend, colors=[colour], widths=[width]), static=True)


def _kinds(names: tuple[str, ...], rotary: set[str]) -> dict[str, dict[str, tuple[str, float]]]:
    """Per servo: the unit and scale of each quantity (rotary: deg, deg/s, N·m; linear: m, m/s, N)."""
    rotary_units = {"position": ("deg", DEG), "velocity": ("deg/s", DEG), "effort": ("N·m", 1.0),
                    "stiffness": ("N·m/rad", 1.0), "damping": ("N·m·s/rad", 1.0)}  # fmt: skip
    linear_units = {"position": ("m", 1.0), "velocity": ("m/s", 1.0), "effort": ("N", 1.0),
                    "stiffness": ("N/m", 1.0), "damping": ("N·s/m", 1.0)}  # fmt: skip
    return {servo: rotary_units if servo in rotary else linear_units for servo in names}


def _table(names: tuple[str, ...], rows: list[tuple[str, str | None, list[str]]]) -> list[str]:
    """Markdown, one column per servo (so a narrow panel still fits a 6-axis arm), one row per quantity.

    ``rows`` holds (quantity, unit or None, cells); a unit shared by every servo goes in the row's name,
    otherwise each cell carries its own (a rotary and a linear servo side by side).
    """
    lines = ["| | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for quantity, unit, cells in rows:
        lines.append(f"| {quantity}{f' ({unit})' if unit else ''} | " + " | ".join(cells) + " |")
    return lines


def _quantity(
    kinds: dict[str, dict[str, tuple[str, float]]], kind: str, values: Any, names: tuple[str, ...]
) -> tuple[str | None, list[str], list[float]]:
    """The shared unit (or None), the cells and the scaled values of one per-servo array."""
    units = {kinds[servo][kind][0] for servo in names}
    shared = units.pop() if len(units) == 1 else None
    scaled = [float(v) * kinds[servo][kind][1] for servo, v in zip(names, values, strict=True)]
    cells = [_number(v) + ("" if shared else f" {kinds[servo][kind][0]}") for servo, v in zip(names, scaled, strict=True)]
    return shared, cells, scaled


def _plots() -> list[rrb.View]:
    """The three plots a ServoState and a ServoCommand share (one root each, so they overlay)."""
    return [
        rrb.TimeSeriesView(origin="/positions", name="position"),
        rrb.TimeSeriesView(origin="/velocities", name="velocity"),
        rrb.TimeSeriesView(origin="/effort", name="effort"),
    ]


# ── ServoState ──────────────────────────────────────────────────────────────────


@log.register
def _(
    state: ServoState,
    name: str | None = None,
    *,
    recording: rr.RecordingStream,
    degrees: bool | Collection[str] = True,
) -> None:
    name = name or "ServoState"
    entity = part(name)
    names = state.name
    kinds = _kinds(names, _rotary(names, degrees))
    rows: list[tuple[str, str | None, list[str]]] = []
    missing = []
    for field, quantity, kind, values in (
        ("positions", "position", "position", state.position),
        ("velocities", "velocity", "velocity", state.velocity),
        ("effort", "effort", "effort", state.effort),
    ):
        if values is None:
            missing.append(quantity)
            continue
        shared, cells, scaled = _quantity(kinds, kind, values, names)
        rows.append((quantity, shared, cells))
        for index, (servo, value) in enumerate(zip(names, scaled, strict=True)):
            unit = kinds[servo][kind][0]
            _series(recording, f"{field}/{part(servo)}/{entity}", value, f"{_label(servo, name, 'ServoState')} ({unit})",
                    _PALETTE[index % len(_PALETTE)], 2.0)  # fmt: skip
    for field, unit, values in (("temperature", "°C", state.temperature), ("voltage", "V", state.voltage)):
        if values is None:
            missing.append(field)
            continue
        rows.append((field, unit, [_number(float(v)) for v in values]))
        for index, (servo, value) in enumerate(zip(names, values.tolist(), strict=True)):
            _series(recording, f"{field}/{part(servo)}/{entity}", value, f"{_label(servo, name, 'ServoState')} ({unit})",
                    _PALETTE[index % len(_PALETTE)], 2.0)  # fmt: skip
    if state.fault is None:
        missing.append("fault")
        health = "fault code not reported"
    else:
        codes = [int(code) for code in state.fault]
        rows.append(("fault", None, [f"**{code}**" if code else "0" for code in codes]))
        for index, (servo, code) in enumerate(zip(names, codes, strict=True)):
            _series(recording, f"fault/{part(servo)}/{entity}", float(code), f"{_label(servo, name, 'ServoState')} fault code",
                    _PALETTE[index % len(_PALETTE)], 2.0)  # fmt: skip
        faulted = [f"{servo} ({code})" for servo, code in zip(names, codes, strict=True) if code]
        health = f"**faulted: {', '.join(faulted)}**" if faulted else "no faults"
    footer = "fault 0 = no fault" + (f" · not measured: {', '.join(missing)}" if missing else "")
    text = "\n".join([f"**seq {state.seq}** · {health}", "", *_table(names, rows), "", footer])
    recording.log(f"notes/{entity}", rr.TextDocument(text, media_type="text/markdown"), static=True)


def _motion() -> dict[str, np.ndarray]:
    """A 3-servo arm (base, shoulder, elbow) making one 2 s minimum-jerk move between two holds, at 50 Hz."""
    t = np.round(np.arange(0.0, 3.0 + 1e-9, 0.02), 6)
    start, end = np.radians([0.0, -40.0, 70.0]), np.radians([60.0, -20.0, 40.0])
    s = np.clip((t - 0.5) / 2.0, 0.0, 1.0)[:, None]
    span = end - start
    q = start + span * (10 * s**3 - 15 * s**4 + 6 * s**5)
    qd = span * (30 * s**2 - 60 * s**3 + 30 * s**4) / 2.0
    qdd = span * (60 * s - 180 * s**2 + 120 * s**3) / 4.0
    gravity = np.stack([np.zeros_like(t), 9.0 * np.cos(q[:, 1]), 2.5 * np.cos(q[:, 1] + q[:, 2])], axis=1)
    return {"t": t, "q": q, "qd": qd, "qdd": qdd, "gravity": gravity}


def _servo_states() -> list[tuple[float, ServoState]]:
    rng = np.random.default_rng(7)
    m = _motion()
    t, n = m["t"], len(m["t"])
    lag = 1  # one 20 ms period behind the setpoint
    q = np.vstack([m["q"][:1].repeat(lag, 0), m["q"][:-lag]]) + rng.normal(0, 0.0004, (n, 3))
    qd = np.vstack([m["qd"][:1].repeat(lag, 0), m["qd"][:-lag]]) + rng.normal(0, 0.004, (n, 3))
    inertia = np.array([0.08, 0.25, 0.04])
    effort = m["gravity"] + inertia * m["qdd"] + 0.15 * np.tanh(qd / 0.05) + rng.normal(0, 0.04, (n, 3))
    heat = np.cumsum(effort**2, axis=0) * 0.02 * 0.004
    temperature = np.round(np.array([31.0, 33.5, 29.0]) + heat, 1)
    voltage = np.round(24.1 - 0.012 * np.abs(effort).sum(axis=1, keepdims=True) + rng.normal(0, 0.008, (n, 1)), 2)
    out = []
    for k in range(n):
        out.append((float(t[k]), ServoState(
            name=("j1", "j2", "j3"),
            position=q[k], velocity=qd[k], effort=effort[k],
            temperature=temperature[k], voltage=np.repeat(voltage[k], 3), fault=[0, 0, 0],
            seq=4000 + k, header=Header(stamp=_EPOCH + float(t[k]), frame_id="base_link"),
        )))  # fmt: skip
    return out


register(TypeView(
    type=ServoState,
    summary="Per-servo position (deg), velocity (deg/s) and effort (N·m) over time, and a table of every field "
    "(temperature, voltage, fault too) at the playhead.",
    sample=lambda: _servo_states()[75][1],
    series=_servo_states,
    views=lambda name: [*_plots(), rrb.TextDocumentView(origin=f"/notes/{part(name)}", name=name)],
    tags=("plot", "stream"),
))  # fmt: skip


# ── ServoCommand ────────────────────────────────────────────────────────────────


@log.register
def _(
    command: ServoCommand,
    name: str | None = None,
    *,
    recording: rr.RecordingStream,
    degrees: bool | Collection[str] = True,
) -> None:
    name = name or "ServoCommand"
    entity = part(name)
    names = command.name
    kinds = _kinds(names, _rotary(names, degrees))
    rows: list[tuple[str, str | None, list[str]]] = []
    not_sent, boards = [], []

    def legend(servo: str, field: str) -> str:
        # "j1 target" alone, "j1 · cmd" when the caller named it (the name already says what it is); a feed-forward
        # always says so, since it shares the effort plot with a measured effort and is not the same quantity
        who = servo if name == "ServoCommand" else f"{servo} · {name}"
        if field == "effort":
            return f"{who} feed-forward"
        return f"{who} target" if name == "ServoCommand" else who

    for field, quantity, kind, values in (
        ("positions", "position target", "position", command.position),
        ("velocities", "velocity target", "velocity", command.velocity),
        ("effort", "feed-forward", "effort", command.feedforward_effort),
    ):
        if values is None:
            not_sent.append(quantity)
            continue
        shared, cells, scaled = _quantity(kinds, kind, values, names)
        if field == "positions":  # robot_msgs: NaN = "no position target" (velocity or effort only)
            cells = ["none" if math.isnan(v) else c for v, c in zip(scaled, cells, strict=True)]
        rows.append((quantity, shared, cells))
        for index, (servo, value) in enumerate(zip(names, scaled, strict=True)):
            # WHY lighter and thinner than the state's line: overlaid on a ServoState, one servo's target and its
            # measurement share a hue, and the measurement stays the line that reads first.
            colour = tuple(int(c + (255 - c) * 0.45) for c in _PALETTE[index % len(_PALETTE)])
            unit = kinds[servo][kind][0]
            # a NaN target is logged as NaN, so the plot shows a gap rather than a made-up value
            _series(recording, f"{field}/{part(servo)}/{entity}", value, f"{legend(servo, field)} ({unit})", colour, 1.5)
    for quantity, kind, values in (
        ("stiffness", "stiffness", command.stiffness),
        ("damping", "damping", command.damping),
        ("max effort", "effort", command.max_effort),
    ):
        if values is None:
            boards.append(quantity)
            continue
        shared, cells, _ = _quantity(kinds, kind, values, names)
        rows.append((quantity, shared, cells))
    if command.latency is not None:
        latency_ms = command.latency * 1000.0
        recording.log(f"latency/{entity}", rr.Scalars([latency_ms]), rr.SeriesLines(names=f"{name} latency (ms)"), static=True)
        seq = "" if command.state_seq is None else f" #{command.state_seq}"
        source = f"sent {latency_ms:.3g} ms after its state{seq}"
    else:
        source = "open loop: computed from no state"
    footer = []
    if boards:
        footer.append(f"the board's own: {', '.join(boards)}")
    if not_sent:
        footer.append(f"not sent: {', '.join(not_sent)}")
    text = "\n".join([f"**{source}**", "", *_table(names, rows), "", " · ".join(footer)])
    recording.log(f"notes/{entity}", rr.TextDocument(text, media_type="text/markdown"), static=True)


def _servo_commands() -> list[tuple[float, ServoCommand]]:
    rng = np.random.default_rng(11)
    m = _motion()
    t = m["t"]
    latency = 0.0018 + np.abs(rng.normal(0, 0.0003, len(t)))
    out = []
    for k in range(len(t)):
        sent = float(t[k] + latency[k])
        out.append((sent, ServoCommand(
            name=("j1", "j2", "j3"),
            position=m["q"][k], velocity=m["qd"][k], feedforward_effort=m["gravity"][k],
            stiffness=[40.0, 60.0, 15.0], damping=[1.2, 2.0, 0.4], max_effort=[8.0, 12.0, 4.0],
            state_stamp=_EPOCH + float(t[k]), state_seq=4000 + k,
            header=Header(stamp=_EPOCH + sent, frame_id="base_link"),
        )))  # fmt: skip
    return out


register(TypeView(
    type=ServoCommand,
    summary="Per-servo position and velocity targets (deg, deg/s) and feed-forward effort (N·m) over time, and a "
    "table of targets, gains, caps and latency at the playhead.",
    sample=lambda: _servo_commands()[75][1],
    series=_servo_commands,
    views=lambda name: [*_plots(), rrb.TextDocumentView(origin=f"/notes/{part(name)}", name=name)],
    tags=("plot", "stream"),
))  # fmt: skip


# ── JointCommandSequence / JointCommandStep: lanes over the steps ────────────────

# Colour by what the joint does: limp grey (no torque), hold blue (steady), move by orange (seeking, e.g. a
# stop), move to green (going to a known place). The step and guard lanes stay a neutral grey.
_MODE_COLOURS = {"limp": 0xB4B8BEFF, "hold": 0x5B84C4FF, "move_by": 0xE0913AFF, "position": 0x4FA46AFF}
_NEUTRAL = 0xDCDFE4FF


def _amount(value: float, rotary: bool, *, rate: bool = False, signed: bool = False, digits: int = 4) -> str:
    """A distance or speed as shown: degrees for a rotary joint, metres for a linear one (SI stays in the data)."""
    unit = ("°/s" if rate else "°") if rotary else (" m/s" if rate else " m")
    shown = (value * DEG if rotary else value) + 0.0
    return f"{shown:+.{digits}g}{unit}" if signed else f"{shown:.{digits}g}{unit}"


def _command_text(command: JointCommand, rotary: bool) -> str:
    effort = "N·m" if rotary else "N"
    if command.mode == "limp":
        return "limp"
    speed = "" if command.speed is None else f" at {_amount(command.speed, rotary, rate=True, digits=3)}"
    if command.mode == "hold":
        text = "hold"
    elif command.mode == "move_by":
        text = f"move by {_amount(command.distance, rotary, signed=True)}{speed}"
    else:
        text = f"to {_amount(command.position, rotary)}{speed}"
    text += f", ≤ {command.max_effort:.4g} {effort}"
    if command.feedforward_effort:
        text += f", {command.feedforward_effort:+.4g} {effort} feed-forward"
    return text


def _condition_text(condition: Any, rotary: set[str]) -> str:
    """A done-when condition in plain words, every parameter kept: ``j1 stalls (lead > 3.44°, < 0.573°/s, 0.2 s)``."""

    def slow(joints: Iterable[str]) -> str:
        return f"< {_amount(condition.speed, all(j in rotary for j in joints), rate=True, digits=3)}, {condition.dwell:.3g} s"

    if isinstance(condition, Elapsed):
        return f"{condition.duration:.4g} s elapsed"
    if isinstance(condition, Reached):
        within = _amount(condition.tolerance, all(j in rotary for j in condition.joints), digits=3)
        return f"{', '.join(condition.joints)} reached (± {within}, {slow(condition.joints)})"
    if isinstance(condition, Settled):
        return f"{', '.join(condition.joints)} settled ({slow(condition.joints)})"
    if isinstance(condition, Stall):
        lead = _amount(condition.lead, condition.joint in rotary, digits=3)
        return f"{condition.joint} stalls (lead > {lead}, {slow([condition.joint])})"
    if isinstance(condition, EffortStall):
        effort = "N·m" if condition.joint in rotary else "N"
        return f"{condition.joint} presses (> {condition.effort:.4g} {effort}, {slow([condition.joint])})"
    return repr(condition)  # a condition type added to robot_msgs later shows as itself


def _lane(recording: rr.RecordingStream, path: str, cells: list[tuple[str, int]], end: int | None = None) -> None:
    """One lane on the step timeline: cell i spans step i+1 to i+2 (the last one to ``end``); a null state closes it."""
    values = [str(i + 1) for i in range(len(cells))]
    starts = list(range(1, len(cells) + 1))
    recording.send_columns(
        path,
        indexes=[rr.TimeColumn(STEP_TIMELINE, sequence=[1])],
        columns=rr.StateConfiguration.columns(
            values=values,
            labels=[label for label, _ in cells],
            colors=np.array([colour for _, colour in cells], dtype=np.uint32),
        ).partition([len(values)]),
    )
    recording.send_columns(
        path,
        indexes=[rr.TimeColumn(STEP_TIMELINE, sequence=[*starts, end or len(cells) + 1])],
        columns=rr.StateChange.columns(state=[*values, None]),
    )


def _lanes(
    steps: tuple[JointCommandStep, ...],
    joints: tuple[str, ...],
    name: str,
    recording: rr.RecordingStream,
    degrees: bool | Collection[str],
    max_speed: float | None = None,
    title: str = "",
) -> None:
    """Lanes on the `step` timeline, in the order Rerun 0.40 draws them (children by path, then the parent's own),
    so the joints come first whatever they are called:

        /steps/<name>/joints/<joint>  what that joint does in each step, coloured by mode
        /steps/<name>/step            the step's name, its timeout and the positions it defines when it ends
        /steps/<name>/until           when the step is done (any one of its conditions)
        /steps/<name>                 the sequence's name and speed guard, across every step (a sequence only)

    WHY over the step index, not seconds: a step ends when its condition holds (a stall, a settle), which only
    a run measures; the message gives the order, so the axis is the order (step 1, 2, ...), each step one cell.
    WHY send_columns: it writes the step index explicitly and leaves the caller's own time cursor alone.
    """
    rotary = _rotary(joints, degrees)
    root = f"steps/{part(name)}"
    for joint in joints:
        cells = []
        for step in steps:
            named = [c for c in step.commands if c.joint == joint]
            if named:
                cells.append((_command_text(named[0], joint in rotary), _MODE_COLOURS[named[0].mode]))
            else:  # robot_msgs: "Joints the step does not name go limp"
                cells.append(("limp (not in this step)", _MODE_COLOURS["limp"]))
        _lane(recording, f"{root}/joints/{part(joint)}", cells)
    step_cells = []
    for step in steps:
        text = f"{step.name} · timeout {step.timeout:.4g} s"
        defined = [f"{joint} := {_amount(value, joint in rotary)}" for joint, value in step.define_position.items()]
        if defined:
            text += " · then " + ", ".join(defined)
        step_cells.append((text, _NEUTRAL))
    _lane(recording, f"{root}/step", step_cells)
    until = [" or ".join(_condition_text(condition, rotary) for condition in step.done_when) for step in steps]
    _lane(recording, f"{root}/until", [(text, _NEUTRAL) for text in until])
    if max_speed is not None:
        guard = _amount(max_speed, all(j in rotary for j in joints), rate=True)
        title = f"{title}: " if title else ""
        text = f"{title}the whole sequence aborts if a powered joint is driven faster than {guard}"
        _lane(recording, root, [(text, _NEUTRAL)], end=len(steps) + 1)


@log.register
def _(
    sequence: JointCommandSequence,
    name: str | None = None,
    *,
    recording: rr.RecordingStream,
    degrees: bool | Collection[str] = True,
) -> None:
    _lanes(
        sequence.steps,
        sequence.joint_names,
        name or sequence.name or "JointCommandSequence",
        recording,
        degrees,
        max_speed=sequence.max_speed,
        title=sequence.name,
    )


@log.register
def _(
    step: JointCommandStep,
    name: str | None = None,
    *,
    recording: rr.RecordingStream,
    degrees: bool | Collection[str] = True,
) -> None:
    joints = tuple(command.joint for command in step.commands)
    _lanes((step,), joints, name or step.name or "JointCommandStep", recording, degrees)


def _homing() -> JointCommandSequence:
    """A two-joint blind homing (the five-bar's shape): settle limp, seek the stop, back off, go home."""
    return JointCommandSequence(
        name="home",
        joint_names=("j1", "j2"),
        max_speed=3.0,
        steps=(
            JointCommandStep(
                name="settle",
                commands=(JointCommand.limp("j1"), JointCommand.limp("j2")),
                done_when=(Elapsed(duration=3.0), Settled(joints=("j1", "j2"), speed=0.01, dwell=0.5)),
                timeout=10.0,
            ),
            JointCommandStep(
                name="seek stop",
                commands=(
                    JointCommand.move_by("j1", 2.0, speed=0.15, max_effort=8.0),
                    JointCommand.hold("j2", max_effort=6.0),
                ),
                done_when=(Stall(joint="j1"),),
                timeout=30.0,
                define_position={"j1": 1.9967},
            ),
            JointCommandStep(
                name="back off",
                commands=(
                    JointCommand.move_by("j1", -0.1, speed=0.3, max_effort=4.0),
                    JointCommand.hold("j2", max_effort=6.0),
                ),
                done_when=(Elapsed(duration=1.0),),
                timeout=5.0,
            ),
            JointCommandStep(
                name="home",
                commands=(
                    JointCommand.to("j1", 1.6, speed=0.3, max_effort=4.0, feedforward_effort=0.5),
                    JointCommand.hold("j2", max_effort=6.0),
                ),
                done_when=(Reached(joints=("j1",), tolerance=0.01),),
                timeout=20.0,
            ),
        ),
    )


register(TypeView(
    type=JointCommandSequence,
    summary="The routine as lanes over its steps, in order: what each joint does (coloured limp, hold, move by, "
    "move to), when each step ends and its timeout, and the speed guard over all of it.",
    sample=_homing,
    views=lambda name: [rrb.StateTimelineView(origin=f"/steps/{part(name)}", name=name)],
    tags=("text",),
))  # fmt: skip

register(TypeView(
    type=JointCommandStep,
    summary="One step as lanes: what each joint it names does (coloured by mode) and when the step ends.",
    sample=lambda: JointCommandStep(
        name="seek stop",
        commands=(
            JointCommand.move_by("j1", 2.0, speed=0.15, max_effort=8.0),
            JointCommand.hold("j2", max_effort=6.0),
        ),
        done_when=(Stall(joint="j1"), Elapsed(duration=20.0)),
        timeout=30.0,
        define_position={"j1": 1.9967},
    ),
    views=lambda name: [rrb.StateTimelineView(origin=f"/steps/{part(name)}", name=name)],
    tags=("text",),
))  # fmt: skip
