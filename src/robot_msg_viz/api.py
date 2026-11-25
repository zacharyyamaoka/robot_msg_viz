"""show / save / stream: the three ways to look at robot_msgs messages in Rerun.

    import robot_msg_viz as viz

    viz.show(pose)                          # opens the viewer with Pose's stock view
    viz.show(target=planned, real=measured) # keyword names become the entity names (overlay)
    viz.save("pose.rrd", pose)              # blueprint embedded; open later with `rerun pose.rrd`
    viz.show(traj, robot="ur5e.urdf")       # a JointTrajectory as its URDF arm (robot_msgs' stock viewer)

    with viz.stream("arm") as live:         # a live viewer; the layout grows as new types arrive
        for state in states:
            live.log(state)                 # time from header.stamp, else seconds since the stream began
"""

from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Any

import rerun as rr
import rerun.blueprint as rrb

from robot_msg_viz._engine import TIMELINE, Timed, log, part
from robot_msg_viz.layout import arrange
from robot_msg_viz.registry import view_for

APPLICATION_ID = "robot_msg_viz"


def views_for(msg: Any, name: str) -> list[Any]:
    """The views one message needs: its type's stock views, else a text panel of its repr."""
    view = view_for(type(msg))
    if view is not None:
        return list(view.views(name))
    return [rrb.TextDocumentView(origin=f"/notes/{part(name)}", name=name)]


def _named(messages: tuple[Any, ...], named: dict[str, Any]) -> list[tuple[str, Any]]:
    return [(type(m).__name__, m) for m in messages] + list(named.items())


def _split(
    messages: tuple[Any, ...], named: dict[str, Any], robot: Any, ghosts: bool
) -> tuple[list[tuple[str, Any]], dict[str, Any]]:
    """(name, message) pairs and the options for the type's page and logger.

    WHY robot_msgs' own rule: ``robot=`` (a URDF path) draws ONE JointTrajectory as its arm, ``tool=`` is then its tool's
    PoseTrajectory and ``ghosts`` its start / end arms; without ``robot``, ``tool`` is a message name like any other
    (``show(target=a, tool=path)``). One API with ``robot_msgs.viz.rerun.show(robot=...)``."""
    if robot is None:
        return _named(messages, named), {}
    named = dict(named)
    tool = named.pop("tool", None)
    pairs = _named(messages, named)
    if len(pairs) != 1:
        raise ValueError(f"robot= draws one message (its arm), got {[type(m).__name__ for _, m in pairs]}")
    options = {"robot": robot, "ghosts": ghosts}
    if tool is not None:
        options["tool"] = tool
    return pairs, options


def _page(pairs: list[tuple[str, Any]], options: dict[str, Any], timed: bool) -> rrb.Blueprint:
    """One message whose type has a page: that page; else every message's stock views, arranged."""
    if len(pairs) == 1:
        name, msg = pairs[0]
        view = view_for(type(msg))
        if view is not None and view.page is not None:
            return view.page(msg, name, **options)
    if options:
        kinds = [type(m).__name__ for _, m in pairs]
        raise TypeError(f"{', '.join(sorted(options))} draw a message on its own page; {kinds} has none")
    views: list[Any] = []
    for name, msg in pairs:
        views += views_for(msg, name)
    return arrange(views, timed=timed)


def blueprint(
    *messages: Any, timed: bool = False, robot: str | Path | None = None, ghosts: bool = True, **named: Any
) -> rrb.Blueprint:
    """The stock blueprint for these messages (what show/save embed), to pass to your own recording.

    A lone message whose type has a page opens on it (a JointTrajectory: robot_msgs' stock trajectory viewer, its tabs
    at full width, or with ``robot=`` its URDF arm beside them); log it with the same options (``viz.log(traj, name,
    recording=rec, robot=urdf)``)."""
    pairs, options = _split(messages, named, robot, ghosts)
    return _page(pairs, options, timed)


def recording(
    *messages: Any, application_id: str = APPLICATION_ID, robot: str | Path | None = None, ghosts: bool = True,
    **named: Any,
) -> tuple[rr.RecordingStream, rrb.Blueprint]:  # fmt: skip
    """A fresh in-memory recording holding the messages, and the blueprint that lays them out."""
    rec = rr.RecordingStream(application_id)
    pairs, options = _split(messages, named, robot, ghosts)
    bp = _page(pairs, options, timed=False)
    for name, msg in pairs:
        log(msg, name, recording=rec, **options)
    return rec, bp


def show(
    *messages: Any, web: bool = False, application_id: str = APPLICATION_ID, robot: str | Path | None = None,
    ghosts: bool = True, **named: Any,
) -> rr.RecordingStream:  # fmt: skip
    """Open the Rerun viewer on the messages, each in its stock view.

    Desktop viewer by default (it ships inside rerun-sdk); ``web=True`` serves the web viewer and
    opens your browser instead; a notebook cell shows it inline. ``robot=`` (a URDF path) draws a
    JointTrajectory as its arm, ``tool=`` its tool's PoseTrajectory, ``ghosts=False`` without the
    faint start / end arms.
    """
    rec, bp = recording(*messages, application_id=application_id, robot=robot, ghosts=ghosts, **named)
    return _open(rec, bp, web=web)


def save(
    path: str | Path, *messages: Any, application_id: str = APPLICATION_ID, robot: str | Path | None = None,
    ghosts: bool = True, **named: Any,
) -> Path:  # fmt: skip
    """Write the messages to an .rrd with their blueprint inside; open it with ``rerun <path>``.

    ``robot=`` / ``tool=`` / ``ghosts=`` as in `show`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = rr.RecordingStream(application_id)
    pairs, options = _split(messages, named, robot, ghosts)
    bp = _page(pairs, options, timed=False)
    rec.save(str(path))
    # WHY send_blueprint, not save(default_blueprint=): that one is make_active=false, so a viewer that remembers a
    # layout for this app id keeps the old one (prior-art sweep 2026-11-24, claim 57); doing both stores it twice.
    rec.send_blueprint(bp, make_active=True, make_default=True)
    for name, msg in pairs:
        log(msg, name, recording=rec, **options)
    rec.flush()
    rec.disconnect()
    return path


def save_series(
    path: str | Path,
    series: list[tuple[float, Any]],
    *,
    name: str | None = None,
    application_id: str = APPLICATION_ID,
) -> Path:
    """Write ``[(t, msg), ...]`` as a timed recording (each message at its time on ``t``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    first = series[0][1]
    label = name or type(first).__name__
    rec = rr.RecordingStream(application_id)
    bp = arrange(views_for(first, label), timed=True)
    rec.save(str(path))
    rec.send_blueprint(bp, make_active=True, make_default=True)  # WHY: as in save()
    timed = Timed(rec)
    for t, msg in series:
        rec.set_time(TIMELINE, duration=float(t))
        log(msg, label, recording=timed)
    rec.flush()
    rec.disconnect()
    return path


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"


def _open(rec: rr.RecordingStream, bp: rrb.Blueprint, *, web: bool) -> rr.RecordingStream:
    if _in_notebook():
        rec.notebook_show(blueprint=bp)
    elif web:
        url = rec.serve_grpc(default_blueprint=bp)
        rr.serve_web_viewer(connect_to=url)
    else:
        rec.spawn(default_blueprint=bp)
    return rec


class Stream:
    """A live recording: ``log`` messages as they happen; the layout grows as new names appear.

    Time on the ``t`` timeline: an explicit ``t=`` (seconds), else the message's ``header.stamp``,
    else seconds since the stream started. Use as a context manager, or call ``close()``.
    ``spawn=True`` opens the desktop viewer, ``web=True`` the web viewer, ``save=`` also writes an .rrd;
    ``blueprint=`` fixes the layout (a dashboard) instead of growing it from the stock views.
    """

    def __init__(
        self,
        name: str = APPLICATION_ID,
        *,
        spawn: bool = True,
        web: bool = False,
        save: str | Path | None = None,
        blueprint: rrb.Blueprint | None = None,
        recording: rr.RecordingStream | None = None,
    ) -> None:
        self.rec = recording or rr.RecordingStream(name)
        self._timed = Timed(self.rec)
        self._fixed = blueprint
        self._views: list[Any] = []
        self._seen: set[tuple[str, type]] = set()
        self._t0_wall = _time.monotonic()
        self._t0_stamp: float | None = None
        if save is not None:
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            self.rec.save(str(save))
        if spawn or web:
            _open(self.rec, blueprint or arrange([], timed=True), web=web)
        if blueprint is not None:
            self.rec.send_blueprint(blueprint)

    def _time_of(self, msg: Any, t: float | None) -> float:
        if t is not None:
            return float(t)
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp:
            if self._t0_stamp is None:
                self._t0_stamp = float(stamp)
            return float(stamp) - self._t0_stamp
        return _time.monotonic() - self._t0_wall

    def log(self, *messages: Any, t: float | None = None, **named: Any) -> None:
        """Log one or more messages at one time (positional ones under their type name)."""
        pairs = _named(messages, named)
        if not pairs:
            return
        self.rec.set_time(TIMELINE, duration=self._time_of(pairs[0][1], t))
        grew = False
        for name, msg in pairs:
            log(msg, name, recording=self._timed)
            key = (name, type(msg))
            if key not in self._seen:
                self._seen.add(key)
                self._views += views_for(msg, name)
                grew = True
        if grew and self._fixed is None:
            self.rec.send_blueprint(arrange(self._views, timed=True))

    def close(self) -> None:
        self.rec.flush()
        self.rec.disconnect()

    def __enter__(self) -> Stream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def stream(name: str = APPLICATION_ID, **options: Any) -> Stream:
    """A live `Stream` (see its docstring); ``with viz.stream("arm") as live: live.log(state)``."""
    return Stream(name, **options)
