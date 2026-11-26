"""Stock views of robot_msgs.compare's dataset nouns (docs/CONTRACT.md: one TypeView per type, registered on import).

Sim-to-real (Zach, 2026-11-25): record every sensor and actuator stream in sim and on the real robot, compare each
stream, then "a bunch of runs … analysis across those runs … drill down and open each run". So:

    DatasetComparison  first the answer across runs: the worst comparison of each stream, then per stream a table of
                       runs × pairs ranked worst first by its headline number, with the runs' tags; then one tab per
                       run, which opens that run's comparisons (each stream's error plots beside its metrics)
    RunComparison      one run: every pair × stream compared, ranked in a summary, each comparison one tab; what was
                       skipped and why
    Dataset            the runs and their tags in a table, then one tab per run with its recordings
    Run                one run: a tab per stream with every source overlaid (target, sim, real: each joint's hue, a
                       lighter then darker shade per source), beside the run's tags, pairs and what each source recorded
    Recording          one source's streams, a tab each

Entity layout, so each run (and each comparison) is its own subtree to open or hide in the viewer:

    /notes/<name>                                      the summary (the answer first)
    /streams/<stream>/<field>/<column>/<source>        a Run's or a Recording's streams, sources overlaid
    /runs/<run>/streams/<stream>/…                     a Dataset's runs, the same layout per run
    /runs/<run>/<a>-<b>/<stream>/error/…, …/notes/…    each Comparison of a run (compare.py's layout, under its root)
    /runs/<run>/notes/<run>                            a run's own summary inside a DatasetComparison

The headline of a comparison (what the tables rank by), per stream type: the worst joint's position rms in degrees
(JointTrajectory, ServoState, ServoCommand), the position rms in mm (a pose), the rotation rms in degrees (an IMU), the
force rms in N (a wrench), the pixel rms (a camera) and the chamfer mean in mm (a cloud). WHY ranked within a stream
and never across: degrees, newtons and pixels do not sort against each other.

WHY overrides of robot_msgs' stock logger: the stock logs every stream at the flat ``/<field>/<column>/<run.source.
stream>`` paths, so two runs (or two streams of one type) land on the same plots, and lays them out automatically.
"""

from __future__ import annotations

# WHY empty: this module registers its TypeViews on import; its functions are helpers, not API (see compare.py).
__all__: list[str] = []

import functools
from collections.abc import Iterable
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from robot_msgs.compare import (
    Comparison,
    Dataset,
    DatasetComparison,
    Recording,
    Run,
    RunComparison,
    compare_dataset,
    compare_run,
)
from robot_msgs.compare.run_comparison import split_key
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
)

from robot_msg_viz._engine import log, part
from robot_msg_viz.api import views_for
from robot_msg_viz.registry import TypeView, register
from robot_msg_viz.views import _sim2real as sim2real
from robot_msg_viz.views._common import DEG
from robot_msg_viz.views.compare import comparison_layout, log_comparison
from robot_msg_viz.views.sensor import CLOUD_POINT_RADIUS, _cloud_colours
from robot_msg_viz.views.streams import (
    _note,
    _panels,
    _range,
    _span,
    _tabs,
    image_views,
    imu_views,
    joint_hue,
    log_images,
    log_imu,
    log_servo_commands,
    log_servo_states,
    log_wrench,
    servo_command_views,
    series_group,
    servo_state_views,
    shade,
    side_by_side,
    source_index,
    wrench_views,
)
from robot_msg_viz.views.trajectory import _num, _plot

# ── one stream of several sources, under a root ──────────────────────────────────────────────────────────────────────


def _log_joints(traj: JointTrajectory, recording: Any, root: str, source: str) -> None:
    leaf = part(source)
    index = source_index(recording, leaf, root)
    seconds = np.asarray(traj.t, dtype=np.float64)
    seconds = seconds - seconds[0] if seconds.size else seconds
    for field, unit in (("positions", "deg"), ("velocities", "deg/s")):
        values = getattr(traj, field)
        if values is None:
            continue
        names = [str(name) for name in traj.joint_names]
        series_group(recording, root, field, leaf, seconds, np.asarray(values, dtype=np.float64) * DEG, names,
                     [f"{name} · {source} ({unit})" for name in names],
                     [shade(joint_hue(joint), index) for joint in range(len(names))], 2.0 if index == 0 else 1.5)  # fmt: skip
    positions = np.asarray(traj.positions, dtype=np.float64) * DEG
    rows = [f"**{source}** · {_span(traj.t)}", "", "| joint | position (deg) |", "|---|---:|"]
    rows += [f"| {name} | {_range(positions[:, j])} |" for j, name in enumerate(traj.joint_names)]
    _note(recording, f"{root}notes/{leaf}", "\n".join(rows))


def _log_cloud(cloud: PointCloud, recording: Any, root: str, source: str) -> None:
    recording.log(f"{root}world/{part(source)}", rr.Points3D(np.asarray(cloud.points), colors=_cloud_colours(cloud),
                                                             radii=CLOUD_POINT_RADIUS), static=True)  # fmt: skip
    _note(recording, f"{root}notes/{part(source)}", f"**{source}** · {len(cloud)} points · frame "
          f"`{cloud.header.frame_id or '—'}`")  # fmt: skip


_ROOTED = {
    ServoStateTrajectory: log_servo_states,
    ServoCommandTrajectory: log_servo_commands,
    ImuTrajectory: log_imu,
    WrenchTrajectory: log_wrench,
    ImageTrajectory: log_images,
}


def log_stream(msg: Any, recording: Any, root: str, source: str, stream: str) -> None:
    """One source's `stream` under `root`, overlaying the other sources of the same stream (the source is the leaf)."""
    for kind, logger in _ROOTED.items():
        if isinstance(msg, kind):
            logger(msg, recording, root=root, name=source)
            return
    if isinstance(msg, JointTrajectory):
        _log_joints(msg, recording, root, source)
    elif isinstance(msg, PointCloud):
        _log_cloud(msg, recording, root, source)
    else:  # any other message: its own view at the usual paths, named <stream>.<source>
        log(msg, f"{stream}.{source}", recording=recording)


def stream_layout(messages: dict[str, Any], root: str, stream: str) -> Any:
    """The views of one stream recorded by several sources (`messages` by source), logged by `log_stream`."""
    first = next(iter(messages.values()))
    notes = [rrb.TextDocumentView(origin=f"/{root}notes/{part(s)}", name=s) for s in messages]
    side = notes[0] if len(notes) == 1 else rrb.Vertical(*notes)
    if isinstance(first, ServoStateTrajectory):
        return side_by_side(rrb.Tabs(*servo_state_views(root, first)), side, name=stream)
    if isinstance(first, ServoCommandTrajectory):
        return side_by_side(rrb.Tabs(*servo_command_views(root, first)), side, name=stream)
    if isinstance(first, ImuTrajectory):  # nested: the plots only (streams.py `_wrench_arrows` says why)
        return side_by_side(rrb.Vertical(*imu_views(root)), side, name=stream)
    if isinstance(first, WrenchTrajectory):
        return side_by_side(rrb.Vertical(*wrench_views(root)), side, name=stream)
    if isinstance(first, ImageTrajectory):
        pictures = [image_views(root, source)[0] for source in messages]
        extra = image_views(root, "", first)[1:]
        return side_by_side(rrb.Grid(*pictures), rrb.Vertical(*extra, side) if extra else side, name=stream)
    if isinstance(first, JointTrajectory):
        plots = [_plot(f"/{root}positions", "position (deg)", legend=False),
                 _plot(f"/{root}velocities", "velocity (deg/s)", legend=False)]  # fmt: skip
        return side_by_side(rrb.Vertical(*plots), side, name=stream)
    if isinstance(first, PointCloud):
        return side_by_side(rrb.Spatial3DView(origin=f"/{root}world", name="clouds"), side, name=stream)
    views: list[Any] = []
    for source, msg in messages.items():
        views += views_for(msg, f"{stream}.{source}")
    return _tabs(views, name=stream)


def _streams(recordings: Iterable[Recording]) -> dict[str, None]:
    """Every stream name, in the order the recordings first name them."""
    return dict.fromkeys(name for rec in recordings for name in rec.streams)


# ── Recording, Run, Dataset: what was recorded ───────────────────────────────────────────────────────────────────────


def _matrix(recordings: dict[str, Recording]) -> list[str]:
    """Stream × source: which source recorded which stream, and how long."""
    sources = list(recordings)
    rows = ["| stream | type | " + " | ".join(sources) + " |", "|---|---|" + "---|" * len(sources)]
    for stream in _streams(recordings.values()):
        kind = next(type(rec.streams[stream]).__name__ for rec in recordings.values() if stream in rec.streams)
        cells = [_short_span(rec.streams[stream]) if stream in rec.streams else "—" for rec in recordings.values()]
        rows.append(f"| {stream} | {kind} | " + " | ".join(cells) + " |")
    return rows


def _short_span(msg: Any) -> str:
    if isinstance(msg, Trajectory):
        return f"{len(msg)} × {_num(msg.duration, 3)} s"
    if isinstance(msg, PointCloud):
        return f"{len(msg)} points"
    return "1 message"


def _meta(recording: Recording) -> str:
    meta = recording.header.meta or {}
    return ", ".join(f"{key} {value}" for key, value in meta.items())


def log_recordings(recordings: dict[str, Recording], recording: Any, base: str) -> None:
    for stream in _streams(recordings.values()):
        root = f"{base}streams/{part(stream)}/"
        for source, rec in recordings.items():
            if stream in rec.streams:
                log_stream(rec.streams[stream], recording, root, source, stream)


def recordings_layout(recordings: dict[str, Recording], base: str) -> list[Any]:
    out = []
    for stream in _streams(recordings.values()):
        messages = {source: rec.streams[stream] for source, rec in recordings.items() if stream in rec.streams}
        out.append(stream_layout(messages, f"{base}streams/{part(stream)}/", stream))
    return out


@log.register
def _(rec: Recording, name: str | None = None, *, recording: Any) -> None:
    name = name or "Recording"
    log_recordings({name: rec}, recording, "")
    rows = [f"**{name}** · {len(rec)} streams" + (f" · {_meta(rec)}" if _meta(rec) else ""), "", *_matrix({name: rec})]
    _note(recording, f"notes/{part(name)}", "\n".join(rows))


def _recording_page(rec: Recording, name: str, **_: Any) -> rrb.Blueprint:
    summary = rrb.TextDocumentView(origin=f"/notes/{part(name)}", name=name)
    return _panels(rrb.Tabs(*recordings_layout({name: rec}, ""), summary))


def _run_summary(run: Run, name: str) -> str:
    title = run.title or run.name
    rows = [f"**{title}** · run `{run.name}` · {len(run.recordings)} sources · "
            f"{len(_streams(run.recordings.values()))} streams", ""]  # fmt: skip
    if run.tags:
        rows += ["| tag | value |", "|---|---|", *(f"| {key} | {value} |" for key, value in run.tags.items()), ""]
    rows += ["Compared: " + ", ".join(f"{a} → {b}" for a, b in run.resolved_pairs()), ""]
    rows += _matrix(dict(run.recordings))
    metas = [f"- {source}: {_meta(rec)}" for source, rec in run.recordings.items() if _meta(rec)]
    if metas:
        rows += ["", *metas]
    offsets = [f"{source} {_num(offset, 3)} s" for source, offset in run.time_offset_s.items() if offset]
    if offsets:
        rows += ["", "Clock offsets: " + ", ".join(offsets)]
    rows += ["", "*In each stream's plots the first source keeps each joint's colour, the next a lighter shade, the "
             "next a darker one.*"]  # fmt: skip
    return "\n".join(rows)


def log_run(run: Run, recording: Any, *, base: str = "", name: str | None = None) -> None:
    name = name or run.name
    log_recordings(dict(run.recordings), recording, base)
    _note(recording, f"{base}notes/{part(name)}", _run_summary(run, name))


def run_layout(run: Run, base: str, name: str) -> Any:
    summary = rrb.TextDocumentView(origin=f"/{base}notes/{part(name)}", name="run")
    return side_by_side(rrb.Tabs(*recordings_layout(dict(run.recordings), base)), summary, name=name)


@log.register
def _(run: Run, name: str | None = None, *, recording: Any) -> None:
    log_run(run, recording, name=name or "Run")


register(TypeView(
    type=Recording,
    summary="One source's recorded streams, a tab each in its own view (servo, IMU, F/T, camera, joints), and a "
    "table of what it recorded and for how long.",
    sample=lambda: sim2real.full_run().recordings["real"],
    views=lambda name: [rrb.TextDocumentView(origin=f"/notes/{part(name)}", name=name)],
    page=_recording_page,
    tags=("plot", "text"),
))  # fmt: skip

register(TypeView(
    type=Run,
    summary="One run: a tab per stream with every source overlaid (target, sim, real), beside its tags, the pairs "
    "it compares and what each source recorded.",
    sample=sim2real.full_run,
    views=lambda name: [rrb.TextDocumentView(origin=f"/notes/{part(name)}", name=name)],
    page=lambda run, name, **_: _panels(run_layout(run, "", name)),
    tags=("plot", "text"),
))  # fmt: skip


def _dataset_summary(dataset: Dataset, name: str) -> str:
    keys = list(dict.fromkeys(key for run in dataset.runs for key in run.tags))
    rows = [f"**{dataset.title or name}** · {len(dataset)} runs · open a run's tab to see its streams", "",
            "| run | " + " | ".join(keys) + " | sources | streams | duration |",
            "|---|" + "---|" * len(keys) + "---|---|---|"]  # fmt: skip
    for run in dataset.runs:
        streams = _streams(run.recordings.values())
        spans = [msg.duration for rec in run.recordings.values() for msg in rec.streams.values() if isinstance(msg, Trajectory)]
        rows.append(f"| {run.name} | " + " | ".join(str(run.tags.get(key, "—")) for key in keys)
                    + f" | {', '.join(run.recordings)} | {len(streams)} | {_num(max(spans, default=float('nan')), 3)} s |")  # fmt: skip
    return "\n".join(rows)


@log.register
def _(dataset: Dataset, name: str | None = None, *, recording: Any) -> None:
    name = name or "Dataset"
    for run in dataset.runs:
        log_run(run, recording, base=f"runs/{part(run.name)}/", name=run.name)
    _note(recording, f"notes/{part(name)}", _dataset_summary(dataset, name))


def _dataset_page(dataset: Dataset, name: str, **_: Any) -> rrb.Blueprint:
    overview = rrb.TextDocumentView(origin=f"/notes/{part(name)}", name="runs")
    return _panels(rrb.Tabs(overview, *(run_layout(run, f"runs/{part(run.name)}/", run.name) for run in dataset.runs)))


register(TypeView(
    type=Dataset,
    summary="Many runs: a table of the runs and their tags first, then a tab per run that opens its streams, every "
    "source overlaid.",
    sample=sim2real.sweep,
    views=lambda name: [rrb.TextDocumentView(origin=f"/notes/{part(name)}", name=name)],
    page=_dataset_page,
    tags=("plot", "text"),
))  # fmt: skip


# ── the headline of one comparison, and the summaries that rank by it ────────────────────────────────────────────────


def headline(comparison: Comparison) -> tuple[str, float, str, str]:
    """(what, value on screen, unit, where): the one number a comparison is ranked by (see the module docstring)."""
    error, metrics = comparison.error, comparison.metrics
    if isinstance(error, (JointTrajectory, ServoStateTrajectory, ServoCommandTrajectory)):
        joints = [(m.split("/", 1)[1], s.rms) for m, s in metrics.items() if m.startswith("positions/")]
        finite = [(joint, value) for joint, value in joints if np.isfinite(value)]
        if finite:
            joint, value = max(finite, key=lambda pair: abs(pair[1]))
            return "position rms, worst joint", abs(value) * DEG, "deg", joint
    picks = (
        (PoseTrajectory, "position", "rms", 1000.0, "mm", "position rms"),
        (ImuTrajectory, "rotation", "rms", DEG, "deg", "rotation rms"),
        (WrenchTrajectory, "force", "rms", 1.0, "N", "force rms"),
        (ImageTrajectory, "pixels", "rms", 1.0, "px", "pixel rms"),
        (PointCloud, "chamfer", "mean", 1000.0, "mm", "chamfer mean"),
    )
    for kind, metric, stat, scale, unit, what in picks:
        if isinstance(error, kind) and metric in metrics:
            return what, abs(float(getattr(metrics[metric], stat))) * scale, unit, ""
    if metrics:
        metric, stats = next(iter(metrics.items()))
        return f"{metric} rms", abs(float(stats.rms)), "SI", ""
    return "no metrics", float("nan"), "", ""


def _cell(value: float, unit: str) -> str:
    return f"{_num(value, 3)} {unit}" if np.isfinite(value) else "—"


def _comparison_name(a: str, b: str) -> str:
    return f"{a}→{b}"


def _comparison_root(run: str, a: str, b: str, stream: str) -> str:
    return f"runs/{part(run)}/{part(a)}-{part(b)}/{part(stream)}/"


def _skipped(result: RunComparison) -> list[tuple[str, str, str, str]]:
    return [(*split_key(key), why) for key, why in result.skipped.items()]


def _skipped_lines(skipped: list[tuple[str, str, str, str]], *, runs: int) -> list[str]:
    """Skipped streams (a, b, stream, why), one row per pair and reason with its streams and how many runs skipped them.

    WHY grouped: a target that records only the joints skips every sensor stream of every pair it is in; listed one by
    one that buried the summary (12 rows for one run). "only 'sim' recorded 'imu'" folds to "only sim recorded them"."""
    if not skipped:
        return []
    groups: dict[tuple[str, str], dict[str, int]] = {}
    for a, b, stream, reason in skipped:
        side = reason.removeprefix("only ").split(" recorded ")[0] if reason.startswith("only ") else None
        why = f"only {side.strip(chr(39))} recorded them" if side else reason
        streams = groups.setdefault((f"{a} → {b}", why), {})
        streams[stream] = streams.get(stream, 0) + 1
    lines = ["", f"**Not compared ({len(skipped)})**, and why", "", "| pair | streams | why | runs |", "|---|---|---|---|"]
    for (pair, why), streams in groups.items():
        counts = sorted(set(streams.values()))
        lines.append(f"| {pair} | {', '.join(streams)} | {why} | {'/'.join(map(str, counts))} of {runs} |")
    return lines


def log_run_comparison(result: RunComparison, recording: Any, *, summary: str) -> None:
    """Every comparison of the run under ``/runs/<run>/<a>-<b>/<stream>/``, and the run's summary at `summary`."""
    rows = []
    for key, comparison in result.comparisons.items():
        a, b, stream = split_key(key)
        log_comparison(comparison, recording, root=_comparison_root(result.run, a, b, stream),
                       name=_comparison_name(a, b))  # fmt: skip
        what, value, unit, where = headline(comparison)
        rows.append((stream, a, b, what, value, unit, where))
    lines = [f"**Run `{result.run}`** · {len(result.comparisons)} comparisons · {len(result.skipped)} not compared · "
             "each comparison is a tab", ""]  # fmt: skip
    if result.tags:
        lines += [" · ".join(f"{key} = {value}" for key, value in result.tags.items()), ""]
    lines += ["| stream | pair | headline | value | where |", "|---|---|---|---:|---|"]
    for stream, a, b, what, value, unit, where in rows:
        lines.append(f"| {stream} | {a} → {b} | {what} | {_cell(value, unit)} | {where} |")
    lines += _skipped_lines(_skipped(result), runs=1)
    _note(recording, summary, "\n".join(lines))


def run_comparison_tabs(result: RunComparison) -> list[Any]:
    tabs = []
    for key, comparison in result.comparisons.items():
        a, b, stream = split_key(key)
        tabs.append(comparison_layout(comparison, _comparison_root(result.run, a, b, stream), _comparison_name(a, b),
                                      title=f"{stream} · {a}→{b}"))  # fmt: skip
    return tabs


@log.register
def _(result: RunComparison, name: str | None = None, *, recording: Any) -> None:
    log_run_comparison(result, recording, summary=f"notes/{part(name or 'RunComparison')}")


def _run_comparison_page(result: RunComparison, name: str, **_: Any) -> rrb.Blueprint:
    summary = rrb.TextDocumentView(origin=f"/notes/{part(name)}", name=f"run {result.run}")
    return _panels(rrb.Horizontal(rrb.Tabs(*run_comparison_tabs(result)), summary, column_shares=[3, 2]))


register(TypeView(
    type=RunComparison,
    summary="One run compared: a summary of every pair × stream with its headline error (worst joint in degrees, "
    "force in N, …) and what was not compared and why, beside a tab per comparison with its error plots.",
    sample=lambda: _full_run_comparison(),
    views=lambda name: [rrb.TextDocumentView(origin=f"/notes/{part(name)}", name=name)],
    page=_run_comparison_page,
    tags=("plot", "text"),
))  # fmt: skip


@functools.cache
def _full_run_comparison() -> RunComparison:
    return compare_run(sim2real.full_run())


def _dataset_comparison_summary(result: DatasetComparison, name: str) -> str:
    rows: dict[str, list[tuple[float, str, str, str, str, str]]] = {}
    keys: list[str] = []
    for run in result.runs.values():
        keys += [key for key in run.tags if key not in keys]
        for key, comparison in run.comparisons.items():
            a, b, stream = split_key(key)
            what, value, unit, where = headline(comparison)
            rows.setdefault(stream, []).append((value, run.run, f"{a} → {b}", what, unit, where))
    compared = sum(len(run.comparisons) for run in result.runs.values())
    skipped = [entry for run in result.runs.values() for entry in _skipped(run)]
    lines = [f"**{result.title or name}** · {len(result.runs)} runs · {compared} comparisons · {len(skipped)} not "
             "compared · open a run's tab to drill into it", "", "**Worst of each stream**", ""]  # fmt: skip
    ranked = {stream: sorted(found, key=lambda row: -row[0] if np.isfinite(row[0]) else np.inf)
              for stream, found in rows.items()}  # fmt: skip
    for stream, found in ranked.items():
        value, run, pair, what, unit, where = found[0]
        lines.append(f"- **{stream}**: {_cell(value, unit)} {what.split(',')[0]}" + (f" ({where})" if where else "")
                     + f" · run `{run}`, {pair}")  # fmt: skip
    for stream, found in ranked.items():
        what, unit = found[0][3], found[0][4]
        lines += ["", f"**{stream}** · {what} ({unit}), worst first", "",
                  "| # | run | pair | value | where | " + " | ".join(keys) + " |",
                  "|---|---|---|---:|---|" + "---|" * len(keys)]  # fmt: skip
        for rank, (value, run, pair, _, unit, where) in enumerate(found, 1):
            tags = result.runs[run].tags
            lines.append(f"| {rank} | {run} | {pair} | {_cell(value, unit)} | {where} | "
                         + " | ".join(str(tags.get(key, "—")) for key in keys) + " |")  # fmt: skip
    lines += _skipped_lines(skipped, runs=len(result.runs))
    return "\n".join(lines)


@log.register
def _(result: DatasetComparison, name: str | None = None, *, recording: Any) -> None:
    name = name or "DatasetComparison"
    for run_name, run in result.runs.items():
        log_run_comparison(run, recording, summary=f"runs/{part(run_name)}/notes/{part(run_name)}")
    _note(recording, f"notes/{part(name)}", _dataset_comparison_summary(result, name))


def _dataset_comparison_page(result: DatasetComparison, name: str, **_: Any) -> rrb.Blueprint:
    overview = rrb.TextDocumentView(origin=f"/notes/{part(name)}", name="across runs")
    runs = []
    for run_name, run in result.runs.items():
        summary = rrb.TextDocumentView(origin=f"/runs/{part(run_name)}/notes/{part(run_name)}", name=f"run {run_name}")
        runs.append(rrb.Horizontal(rrb.Tabs(*run_comparison_tabs(run)), summary, column_shares=[3, 2], name=run_name))
    return _panels(rrb.Tabs(overview, *runs))


@functools.cache
def _sweep_comparison() -> DatasetComparison:
    return compare_dataset(sim2real.sweep())


register(TypeView(
    type=DatasetComparison,
    summary="Many runs compared: the worst of each stream first, then per stream a table of runs × pairs ranked "
    "worst first with the runs' tags; a tab per run opens its comparisons.",
    sample=_sweep_comparison,
    views=lambda name: [rrb.TextDocumentView(origin=f"/notes/{part(name)}", name=name)],
    page=_dataset_comparison_page,
    tags=("plot", "text"),
))  # fmt: skip
