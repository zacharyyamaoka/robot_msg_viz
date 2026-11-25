"""Place a set of views into one blueprint: the 3D scene left, everything else beside it.

WHY a fixed arrangement, not Rerun's automatic one: the automatic layout gives every entity its own
panel (robot_msgs found it split an overlay into nine plots, 2026-11-11). Here the views come from the
types that were logged, so the same message always opens the same way.
"""

from __future__ import annotations

from typing import Any

import rerun.blueprint as rrb


def _key(view: Any) -> tuple[str, str]:
    origin = getattr(view, "origin", "")
    return type(view).__name__, str(origin)


def unique(views: list[Any]) -> list[Any]:
    """Drop repeats of the same view class on the same origin, first one wins."""
    seen: set[tuple[str, str]] = set()
    kept = []
    for view in views:
        key = _key(view)
        if key in seen:
            continue
        seen.add(key)
        kept.append(view)
    return kept


def arrange(views: list[Any], *, timed: bool = False, time_panel: Any | None = None) -> rrb.Blueprint:
    """One blueprint: 3D views stacked on the left, the rest as a grid (≤4) or tabs on the right.

    The blueprint and selection side panels start collapsed (the view is the point); the time panel
    is open only when the data moves (``timed``), so a single message opens as a clean picture.
    """
    views = unique(views)
    spatial = [v for v in views if isinstance(v, rrb.Spatial3DView)]
    rest = [v for v in views if not isinstance(v, rrb.Spatial3DView)]
    parts: list[Any] = []
    if spatial:
        parts.append(spatial[0] if len(spatial) == 1 else rrb.Vertical(*spatial))
    if rest:
        if len(rest) == 1:
            parts.append(rest[0])
        elif len(rest) <= 4:
            parts.append(rrb.Vertical(*rest) if spatial else rrb.Grid(*rest))
        else:
            parts.append(rrb.Tabs(*rest))
    if not parts:
        parts.append(rrb.TextDocumentView(origin="/", name="empty"))
    root = parts[0] if len(parts) == 1 else rrb.Horizontal(*parts, column_shares=[3, 2])
    if time_panel is None:
        # WHY collapsed even when timed: collapsed still shows the play bar and scrubber; expanded adds the
        # streams tree, which took a quarter of every still (wave 1 site lane, 2026-11-24).
        time_panel = rrb.TimePanel(state="collapsed")
    return rrb.Blueprint(
        root,
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        time_panel,
    )
