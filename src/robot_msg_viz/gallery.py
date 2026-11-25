"""The gallery: every robot_msgs type (supported or not) plus every use-case dashboard a plugin adds.

The landing page, the screenshots and the ruler all read `entries()`, so a new view or dashboard shows
up everywhere with no site edit. A package adds use cases through the entry-point group
``robot_msg_viz.gallery``: each entry point names a function returning a list of `GalleryEntry`
(robot_dashboards does this), so this package never imports the packages built on it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path

from robot_msg_viz.api import save, save_series
from robot_msg_viz.registry import catalog

ENTRY_POINT_GROUP = "robot_msg_viz.gallery"


@dataclass(frozen=True)
class GalleryEntry:
    """One card on the landing page.

    - ``kind``: ``"type"`` (one robot_msgs type) or ``"use_case"`` (a dashboard).
    - ``key``: ``"geometry/Pose"`` or ``"object_detection"``; also the file stem of its media.
    - ``group``: the robot_msgs domain, or the use case's category.
    - ``supported``: a type without a stock view yet is listed, greyed, never hidden.
    - ``write``: writes the sample recording (blueprint inside) to the path it is given.
    - ``consumes``: the robot_msgs type keys it draws (links a use case to its types).
    - ``source``: the package that provides it (for the page and the ruler).
    - ``viewers`` / ``adapters``: where it can be drawn and what it takes in (the use-case page lists both).
    """

    kind: str
    key: str
    title: str
    group: str
    summary: str
    supported: bool
    code: str = ""
    write: Callable[[Path], Path] | None = None
    consumes: tuple[str, ...] = ()
    source: str = "robot_msg_viz"
    timed: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)
    # (viewer title, "supported" | "roadmap"): where the entry can be drawn; a type here is drawn in Rerun.
    viewers: tuple[tuple[str, str], ...] = (("Rerun", "supported"),)
    # the input adapters it takes (a use case: robot_msgs plus whatever its package registers)
    adapters: tuple[str, ...] = ("robot_msgs",)

    @property
    def slug(self) -> str:
        return self.key.replace("/", "-")


def _var(name: str) -> str:
    return name[0].lower() + name[1:]


def type_entries() -> list[GalleryEntry]:
    """One entry per robot_msgs type, in the catalogue's order."""
    out = []
    for item in catalog():
        view = item.view
        if view is None:
            out.append(GalleryEntry(
                kind="type", key=item.key, title=item.name, group=item.domain,
                summary="No stock view yet: show() draws its repr as text.", supported=False,
                code=f"from robot_msgs.{item.domain} import {item.name}",
            ))  # fmt: skip
            continue
        var = _var(item.name)
        code = view.code or (
            f"import robot_msg_viz as viz\nfrom robot_msgs.{item.domain} import {item.name}\n\n"
            f"viz.show({var})                 # the stock view, in the desktop viewer\n"
            f'viz.save("{var}.rrd", {var})     # or a file: rerun {var}.rrd'
        )
        app_id = f"robot_msg_viz_{item.name}"  # WHY no dot: Rerun 0.40 migrates dotted app ids to a hashed name

        def write(path: Path, view=view, name=item.name, app_id=app_id) -> Path:
            if view.series is not None:
                return save_series(path, view.series(), name=name, application_id=app_id)
            return save(path, application_id=app_id, **{name: view.sample()})

        out.append(GalleryEntry(
            kind="type", key=item.key, title=item.name, group=item.domain, summary=view.summary,
            supported=True, code=code, write=write, consumes=(item.key,), timed=view.series is not None,
            tags=view.tags,
        ))  # fmt: skip
    return out


def use_case_entries() -> list[GalleryEntry]:
    """Every use case the installed plugins register (empty when none is installed)."""
    out: list[GalleryEntry] = []
    for point in sorted(entry_points(group=ENTRY_POINT_GROUP), key=lambda p: p.name):
        out += list(point.load()())
    return out


def entries() -> list[GalleryEntry]:
    return type_entries() + use_case_entries()
