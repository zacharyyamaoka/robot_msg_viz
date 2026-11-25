"""Stock views of robot_msgs.std types (docs/CONTRACT.md: one TypeView per type, registered on import).

    ColorRGBA         a swatch (opaque | at its alpha over a checker) and its channel values
    LogRecord         Rerun's text log: one row per record, coloured by level, one entity per component
    DiagnosticRecord  a health lane per component (ok / warn / error over time) and the records as a log
    Metrics           the named numbers as a table; each is also a scalar series for your own plots

Header has no view of its own (it stays listed, unsupported): it is the stamp and frame of the message that
carries it. A stream already puts the stamp on the ``t`` timeline and the caller's name on the entity, so a
Header alone has nothing left to draw but its repr, which is exactly the fallback.

robot_msgs' stock logger draws none of these types (each falls to its repr), so each gets a logger here.
Entity paths (docs/CONTRACT.md):

    /image/<name>, /image/<name>/labels|outline   the ColorRGBA swatch, its two captions and its edge
    /notes/<name>                                 ColorRGBA's values, Metrics' table (one document per name)
    /notes/<name>/<component>                     LogRecord and DiagnosticRecord lines (Rerun text logs)
    /health/<component>/<name>                    DiagnosticRecord level, a state lane
    /uptime/<component>/<name>                    DiagnosticRecord uptime (s), a scalar series
    /metrics/<name>/<component>/<metric>          each Metrics value, a scalar series
"""

from __future__ import annotations

import math

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from robot_msgs.std import ColorRGBA, DiagnosticRecord, Header, LogRecord, Metrics

from robot_msg_viz._engine import log, part
from robot_msg_viz.registry import TypeView, register

# A wall clock for the samples: 2026-11-24 12:00:00 UTC, so stamps read like a real log.
_EPOCH = 1_795_521_600.0


def _number(value: float) -> str:
    """A value as written: 6 significant digits, and NaN spelled out (never a blank cell)."""
    return "NaN (not measured)" if math.isnan(value) else f"{value:.6g}"


# ── ColorRGBA: a swatch ─────────────────────────────────────────────────────────

_SWATCH_H, _SWATCH_W, _CHECK = 160, 320, 20


def _swatch(color: ColorRGBA) -> np.ndarray:
    """Left half the colour opaque; right half as it draws at its alpha over a light checker.

    WHY two halves (the stock matplotlib swatch shows only the composite): a translucent colour over a
    checker hides its hue, and an opaque patch alone hides the alpha, so each half answers one of the two.
    """
    rows, cols = np.mgrid[0:_SWATCH_H, 0:_SWATCH_W]
    checker = np.where(((rows // _CHECK) + (cols // _CHECK)) % 2, 204.0, 255.0)[..., None]
    rgb = color.as_array()[:3] * 255.0
    pixels = np.empty((_SWATCH_H, _SWATCH_W, 3))
    pixels[:] = color.a * rgb + (1.0 - color.a) * checker
    pixels[:, : _SWATCH_W // 2] = rgb
    return np.round(pixels).astype(np.uint8)


@log.register
def _(color: ColorRGBA, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    name = part(name or "ColorRGBA")
    recording.log(f"image/{name}", rr.Image(_swatch(color)), static=True)
    # a thin grey outline, so a white or nearly transparent colour still has an edge against the viewer
    recording.log(
        f"image/{name}/outline",
        rr.Boxes2D(mins=[[0, 0]], sizes=[[_SWATCH_W, _SWATCH_H]], colors=[[150, 150, 150]]),
        static=True,
    )
    luminance = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b
    ink = [20, 20, 20] if luminance > 0.55 else [245, 245, 245]
    recording.log(
        f"image/{name}/labels",
        rr.Points2D(
            [[_SWATCH_W / 4, _SWATCH_H / 2], [3 * _SWATCH_W / 4, _SWATCH_H / 2]],
            radii=0.0,
            labels=["opaque", f"alpha {color.a:.6g}"],
            colors=[ink, [30, 30, 30]],
            show_labels=True,
        ),
        static=True,
    )
    eight_bit = [round(channel * 255) for channel in (color.r, color.g, color.b, color.a)]
    text = "\n".join([
        f"## {color.to_hex(alpha=True)}",
        "",
        "| | r | g | b | a |",
        "|---|---|---|---|---|",
        "| 0 to 1 | " + " | ".join(f"{c:.6g}" for c in color.as_array()) + " |",
        "| 8-bit | " + " | ".join(str(v) for v in eight_bit) + " |",
    ])  # fmt: skip
    recording.log(f"notes/{name}", rr.TextDocument(text, media_type="text/markdown"), static=True)


register(TypeView(
    type=ColorRGBA,
    summary="A swatch of the colour, opaque beside how it draws at its alpha over a checker, with its hex and channel values.",
    sample=lambda: ColorRGBA.from_hex("#2a7de1b3"),
    views=lambda name: [
        rrb.Spatial2DView(origin=f"/image/{part(name)}", name=name),
        rrb.TextDocumentView(origin=f"/notes/{part(name)}", name=f"{name} values"),
    ],
    tags=("2d", "text"),
))  # fmt: skip


# ── LogRecord: Rerun's text log ───────────────────────────────────────────────────

# The stdlib numbers Rerun colours, spelled as Rerun spells them; any other level keeps logging's own name
# ("Level 25"), so a custom level shows as itself rather than being bucketed into a neighbour.
_TEXT_LOG_LEVELS = {10: "DEBUG", 20: "INFO", 30: "WARN", 40: "ERROR", 50: "CRITICAL"}


@log.register
def _(record: LogRecord, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    level = _TEXT_LOG_LEVELS.get(record.level, record.level_name)
    recording.log(
        f"notes/{part(name or 'LogRecord')}/{part(record.path or 'component')}",
        rr.TextLog(f"[tick {record.tick}] {record.message}", level=level),
        static=True,
    )


def _log_records() -> list[tuple[float, LogRecord]]:
    """A pick cell's first five seconds: three components, every level but CRITICAL."""
    lines = [
        (0.00, "pick_cell.arm.jtc", 0, 20, "started: 6 joints at 1000 Hz"),
        (0.05, "pick_cell.camera", 0, 20, "streaming 640x480 at 30 Hz"),
        (0.42, "pick_cell.arm.jtc", 420, 10, "trajectory 'approach' loaded: 1.8 s, 91 samples"),
        (1.20, "pick_cell.gripper", 36, 20, "homed: width 0.085 m"),
        (2.31, "pick_cell.arm.jtc", 2310, 30, "input 'state' is 12.0 ms old (limit 5.0 ms)"),
        (2.33, "pick_cell.arm.jtc", 2330, 20, "input 'state' fresh again (1.1 ms old)"),
        (3.90, "pick_cell.camera", 117, 40, "frame 117 dropped: USB bandwidth"),
        (4.50, "pick_cell.arm.jtc", 4500, 20, "goal reached: 0.3 mm from target"),
    ]
    return [
        (t, LogRecord(header=Header(stamp=_EPOCH + t), path=path, tick=tick, level=level, message=message))
        for t, path, tick, level, message in lines
    ]


register(TypeView(
    type=LogRecord,
    summary="Rerun's text log: one row per record (time, component, level, tick and message), warnings and errors coloured.",
    sample=lambda: _log_records()[4][1],
    series=_log_records,
    views=lambda name: [rrb.TextLogView(origin=f"/notes/{part(name)}", name=name)],
    tags=("text", "stream"),
))  # fmt: skip


# ── DiagnosticRecord: a health lane per component, and the records ───────────────

# WHY these colours: amber and red only for problems (Zach's calm-UI rule); ok is a quiet green and unknown
# a grey, so a healthy cell reads as one calm band and a fault is the only thing that stands out.
_HEALTH = rr.StateConfiguration(
    values=["unknown", "ok", "warn", "error"],
    colors=np.array([0x9AA0A6FF, 0x7DB88AFF, 0xE6A23CFF, 0xD9534FFF], dtype=np.uint32),
)


def _diagnostic_text(record: DiagnosticRecord) -> str:
    said = f"{record.code}: {record.message}" if record.code else (record.message or "(no message)")
    parts = [record.path, said, record.mode, f"up {record.uptime_s:.6g} s"]
    if record.stale_inputs:
        parts.append("stale inputs: " + ", ".join(record.stale_inputs))
    return " · ".join(parts)


@log.register
def _(record: DiagnosticRecord, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    name = part(name or "DiagnosticRecord")
    component = part(record.path or "component")
    recording.log(f"health/{component}/{name}", _HEALTH, rr.StateChange(state=record.level), static=True)
    recording.log(
        f"uptime/{component}/{name}",
        rr.Scalars([record.uptime_s]),
        rr.SeriesLines(names=f"{record.path} uptime (s)"),
        static=True,
    )
    recording.log(
        f"notes/{name}/{component}",
        rr.TextLog(_diagnostic_text(record), level=record.level.upper()),
        static=True,
    )


def _diagnostics() -> list[tuple[float, DiagnosticRecord]]:
    """Twelve seconds of a pick cell at 1 Hz: the arm goes stale for three seconds; the gripper's drive
    faults, restarts (its uptime drops to zero) and comes back."""
    out: list[tuple[float, DiagnosticRecord]] = []
    for second in range(12):
        t = float(second)
        if 4 <= second < 7:
            arm = {"level": "warn", "code": "InputStale", "message": "input 'state' is 12.0 ms old (limit 5.0 ms)",
                   "stale_inputs": ("state",)}  # fmt: skip
        else:
            arm = {"level": "ok", "message": "running"}
        out.append((t, DiagnosticRecord(
            header=Header(stamp=_EPOCH + t), path="pick_cell.arm.jtc", mode="operational", uptime_s=93.0 + t, **arm,
        )))  # fmt: skip
        if second < 7:
            gripper = {"level": "ok", "message": "running", "mode": "operational", "uptime_s": 41.0 + t}
        elif second < 9:
            gripper = {"level": "error", "code": "MoteusFault.33", "message": "drive fault 33: motor driver fault",
                       "mode": "operational", "uptime_s": 41.0 + t}  # fmt: skip
        elif second == 9:
            gripper = {"level": "unknown", "code": "Restarting", "message": "drive restarting",
                       "mode": "initialization", "uptime_s": 0.0}  # fmt: skip
        else:
            gripper = {"level": "ok", "message": "running", "mode": "operational", "uptime_s": t - 9.0}
        out.append((t + 0.5, DiagnosticRecord(header=Header(stamp=_EPOCH + t + 0.5), path="pick_cell.gripper", **gripper)))
    return out


register(TypeView(
    type=DiagnosticRecord,
    summary="One health lane per component (ok, warn, error, unknown over time) above the records themselves as a log.",
    sample=lambda: _diagnostics()[8][1],
    series=_diagnostics,
    views=lambda name: [
        rrb.StateTimelineView(origin="/health", name="health"),
        # WHY the entity-path column hidden: each line already starts with its component, and the path
        # (/notes/<name>/<component>) took half of a side-by-side panel from the message
        rrb.TextLogView(
            origin=f"/notes/{part(name)}",
            name=f"{name} records",
            columns=rrb.archetypes.TextLogColumns(text_log_columns=[
                rrb.encodings.TextLogColumn("LogLevel"),
                rrb.encodings.TextLogColumn("Body"),
                rrb.encodings.TextLogColumn("EntityPath", visible=False),
            ]),
        ),
    ],
    tags=("text", "stream"),
))  # fmt: skip


# ── Metrics: a table of named numbers ───────────────────────────────────────────────


@log.register
def _(metrics: Metrics, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    """The values as a table at /notes/<name>, and each as a scalar series at /metrics/<name>/<component>/<metric>.

    WHY a table, not one plot: a component's metrics mix scales and units (1000 steps beside a 0.000183 s
    percentile), so one shared axis flattens all but the largest, and a plot per metric needs the names,
    which a view (built from the entity name alone) does not have. The series are there for your own plots.
    """
    name = part(name or "Metrics")
    component = part(metrics.path or "component")
    for metric, value in zip(metrics.names, metrics.values.tolist(), strict=True):
        recording.log(
            f"metrics/{name}/{component}/{part(metric)}",
            rr.Scalars([value]),
            rr.SeriesLines(names=metric),
            static=True,
        )
    rows = [f"**{metrics.path}**", "", "| metric | value |", "|---|---:|"]
    pairs = zip(metrics.names, metrics.values.tolist(), strict=True)
    rows += [f"| {metric} | {_number(value)} |" for metric, value in pairs]
    if not metrics.names:
        rows = [f"**{metrics.path}**", "", "(no metrics in this record)"]
    recording.log(f"notes/{name}", rr.TextDocument("\n".join(rows), media_type="text/markdown"), static=True)


register(TypeView(
    type=Metrics,
    summary="The component's named numbers as a table (NaN spelled out as not measured); each is also a scalar series.",
    sample=lambda: Metrics(
        header=Header(stamp=_EPOCH + 60.0),
        path="pick_cell.arm.jtc",
        names=("steps", "step_mean_s", "step_p99_s", "overruns", "input_age_s.state", "input_age_s.goal"),
        values=[1000.0, 0.000142, 0.000183, 0.0, 0.0011, float("nan")],
    ),
    views=lambda name: [rrb.TextDocumentView(origin=f"/notes/{part(name)}", name=name)],
    tags=("text",),
))  # fmt: skip
