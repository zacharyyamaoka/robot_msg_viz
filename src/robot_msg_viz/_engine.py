"""The one dispatch table: robot_msg_viz's loggers first, robot_msgs' stock logger underneath.

`log(msg, name, recording=rec)` is a `functools.singledispatch`. A view module overrides a type with
``@log.register`` only when the stock logger in ``robot_msgs.viz.rerun`` is missing or not enough
(for example, it logs a JointState as a static table, so a stream of them would have no history).
Everything else falls through to the stock logger, so the two never disagree on a type both cover.

WHY not edit robot_msgs.viz.rerun directly (2026-11-24): robot_msgs ships `.show()` from PyPI and a
peer round is growing its JointTrajectory view in that module today. Overrides live here until the
two are merged; `robot_msgs.viewers` can then point "rerun" at this package.
"""

from __future__ import annotations

import re
from functools import singledispatch
from typing import Any

import rerun as rr
from robot_msgs.viz import rerun as stock

TIMELINE = stock.TIMELINE  # "t": seconds, shared with robot_msgs and traj_compare so recordings overlay
RERUN_SDK_VERSION = stock.RERUN_SDK_VERSION


def part(name: object) -> str:
    """One entity-path segment (Rerun splits paths on '/'); the same rule as robot_msgs."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)) or "_"


@singledispatch
def log(msg: Any, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    """Log one message under `name` (default: its type name) into `recording`.

    Falls through to robot_msgs' stock logger; a view module registers an override per type.
    """
    stock.log(msg, name, recording=recording)
    # WHY: the stock logger remembers every path per stream id for its own auto-layout and only
    # forgets them in its own show()/save(); we lay out from the views instead, so drop them here
    # or a long stream grows that list forever.
    getattr(stock, "_PATHS", {}).pop(id(recording), None)
    getattr(stock, "_PATHS", {}).pop(id(getattr(recording, "_inner", None)), None)


def has_dedicated_logger(message_type: type) -> bool:
    """True when some logger (ours or the stock one) draws this type, not the repr-as-text fallback."""
    if log.dispatch(message_type) is not log.dispatch(object):
        return True
    return stock.log.dispatch(message_type) is not stock.log.dispatch(object)


class Timed:
    """A recording that puts every log on the timeline, even ones a stock logger marks static.

    WHY: robot_msgs' loggers log one message as a static snapshot (right for `show(msg)`); a stream
    needs each message at its time, or the viewer keeps only the last one and nothing can be scrubbed.
    """

    def __init__(self, inner: rr.RecordingStream) -> None:
        self._inner = inner

    def log(self, entity_path: Any, *archetypes: Any, static: bool = False, **options: Any) -> None:
        self._inner.log(entity_path, *archetypes, static=False, **options)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
