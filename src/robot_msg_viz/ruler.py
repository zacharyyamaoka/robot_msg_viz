"""Read an .rrd back with Rerun's own CLI (`rerun rrd print`), independent of the code that wrote it.

WHY an outside oracle: the writer's own bookkeeping can claim a path was logged when nothing reached
the file; `rerun rrd print` reports what the viewer will actually load.
Part of the frozen ruler (robot_msg_viz's and robot_dashboards' contract tests and robot_viz_agent's
evaluation all import it): lanes never edit this file.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

_CHUNK = re.compile(r"^Chunk\(\S+\) with (\d+) rows .* - (\S+) - data columns: \[(.*)\]")
# WHY by path, not by the StoreInfo block above a chunk: a stream that re-sends its blueprint interleaves
# the two stores' chunks, and `rrd print` does not repeat the store per chunk. Blueprint entities live only
# at these paths (views and containers by uuid, the viewport and the panels).
_BLUEPRINT = re.compile(r"^/(?:(?:view|container)/[0-9a-f-]{36}(?:/.*)?|viewport|\w+_panel)$")


@dataclass
class RrdContents:
    data_paths: set[str] = field(default_factory=set)
    blueprint_paths: set[str] = field(default_factory=set)
    data_columns: dict[str, set[str]] = field(default_factory=dict)
    default_blueprint: bool = False
    rows: int = 0

    @property
    def has_viewport(self) -> bool:
        return "/viewport" in self.blueprint_paths


def rerun_cli() -> list[str]:
    found = Path(sys.executable).with_name("rerun")
    if found.exists():
        return [str(found)]
    which = shutil.which("rerun")
    return [which] if which else [sys.executable, "-m", "rerun"]


def read(path: str | Path) -> RrdContents:
    out = subprocess.run(
        [*rerun_cli(), "rrd", "print", str(path)], capture_output=True, text=True, timeout=120, check=True
    ).stdout
    contents = RrdContents()
    for line in out.splitlines():
        if line.startswith("BlueprintActivationCommand") and "make_default: true" in line:
            contents.default_blueprint = True
            continue
        match = _CHUNK.match(line)
        if not match:
            continue
        rows, entity, columns = int(match.group(1)), match.group(2), match.group(3)
        if _BLUEPRINT.match(entity):
            contents.blueprint_paths.add(entity)
        else:
            if entity.startswith("/__properties"):
                continue
            contents.data_paths.add(entity)
            contents.rows += rows
            contents.data_columns.setdefault(entity, set()).update(columns.split())
    return contents


def covers(contents: RrdContents, origin: str) -> bool:
    """True when some logged entity lives at or under ``origin`` (a view would show something)."""
    origin = "/" + str(origin).strip("/")
    if origin == "/":
        return bool(contents.data_paths)
    return any(p == origin or p.startswith(origin + "/") for p in contents.data_paths)


def views_in(blueprint: object) -> list[object]:
    """Every view in a blueprint (walks its containers), for checking each one points at data."""
    import rerun.blueprint as rrb

    found: list[object] = []

    def walk(part: object) -> None:
        if isinstance(part, rrb.View):
            found.append(part)
            return
        for child in getattr(part, "contents", None) or []:
            walk(child)

    walk(getattr(blueprint, "root_container", blueprint))
    return found
