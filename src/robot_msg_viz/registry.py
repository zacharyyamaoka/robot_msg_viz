"""The standard view of each robot_msgs type, and the catalogue of every type (supported or not).

A view module (``robot_msg_viz/views/<domain>.py``) registers one `TypeView` per type it supports:

    register(TypeView(
        type=Pose,
        summary="A coordinate frame: red/green/blue axes at the position, along the orientation.",
        sample=lambda: Pose.from_xyz_rpy(0.4, 0.1, 0.3, 0.0, 0.5, 0.8),
        views=lambda name: [rrb.Spatial3DView(origin="/world", name="world")],
    ))

`catalog()` lists every type `robot_msgs.message_types()` knows, in robotmsgs.com's order, each
marked supported when it has a TypeView. The contract every TypeView meets is docs/CONTRACT.md and
tests/test_contract.py.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import robot_msgs

# WHY this order: robotmsgs.com's sidebar (website/build/model.py PACKAGE_ORDER, Zach's order), then
# the domains it adds after, so the two sites list types the same way.
DOMAIN_ORDER = (
    "std", "geometry", "shape", "sensor", "vision", "trajectory", "control", "manipulation", "compare",
)  # fmt: skip


@dataclass(frozen=True)
class TypeView:
    """The stock view of one robot_msgs type.

    - ``type``: the robot_msgs class (a subclass falls back to its parent's view only if it has none).
    - ``summary``: one line, what you see in the viewer.
    - ``sample``: a showcase message, deterministic, small (it ships in the gallery's .rrd).
    - ``views``: the Rerun views a message logged under ``name`` needs; `layout.arrange` places them.
    - ``series``: optional ``[(t_seconds, message), ...]`` the gallery streams instead of ``sample``
      (a JointState or an Imu reads as a signal over time, not one value).
    - ``code``: optional snippet for the gallery page; the default is ``show(<var>)``.
    - ``page``: optional ``page(msg, name, **options) -> rrb.Blueprint``, the whole layout when the message is shown
      alone (a dashboard built from the message itself, as JointTrajectory's tabs, whose strips carry its joints and
      numbers); ``options`` are what `show` / `save` take besides messages (``robot=``, ``ghosts=``, ``tool=``) and the
      type's logger takes them too. Several messages are laid out from ``views`` instead.
    - ``options``: optional ``() -> dict``, the options the gallery shows the sample with (``{"robot": <URDF>}``).
    """

    type: type
    summary: str
    sample: Callable[[], Any]
    views: Callable[[str], list[Any]]
    series: Callable[[], list[tuple[float, Any]]] | None = None
    code: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    page: Callable[..., Any] | None = None
    options: Callable[[], dict[str, Any]] | None = None

    @property
    def key(self) -> str:
        return key_of(self.type)


VIEWS: dict[type, TypeView] = {}


def register(view: TypeView) -> TypeView:
    """Add (or replace) the stock view of a type; returns it."""
    VIEWS[view.type] = view
    return view


def key_of(message_type: type) -> str:
    """``"geometry/Pose"``: robot_msgs' own key for a type (its domain and name)."""
    for key, cls in robot_msgs.message_types().items():
        if cls is message_type:
            return key
    domain = message_type.__module__.split(".")[1] if message_type.__module__.startswith("robot_msgs.") else "other"
    return f"{domain}/{message_type.__name__}"


def view_for(message_type: type) -> TypeView | None:
    """The type's own view, else the nearest base class's (a JointTrajectory uses Trajectory's)."""
    _load_views()
    for cls in message_type.__mro__:
        if cls in VIEWS:
            return VIEWS[cls]
    return None


@dataclass(frozen=True)
class CatalogEntry:
    key: str
    domain: str
    name: str
    type: type
    view: TypeView | None

    @property
    def supported(self) -> bool:
        return self.view is not None


def catalog() -> list[CatalogEntry]:
    """Every robot_msgs type, supported or not, in the gallery's order."""
    _load_views()
    types = {k: c for k, c in robot_msgs.message_types().items() if c.__module__.startswith("robot_msgs.")}
    rank = {d: i for i, d in enumerate(DOMAIN_ORDER)}
    entries = []
    for key, cls in types.items():
        domain, name = key.split("/")
        listed = list(getattr(getattr(robot_msgs, domain, None), "__all__", []))
        order = listed.index(name) if name in listed else len(listed)
        entries.append(((rank.get(domain, len(rank)), domain, order, name), CatalogEntry(key, domain, name, cls, VIEWS.get(cls))))
    return [entry for _, entry in sorted(entries, key=lambda pair: pair[0])]


def supported_types() -> Iterable[type]:
    _load_views()
    return tuple(VIEWS)


_LOADED = False


def _load_views() -> None:
    """Import every view module once (each registers its TypeViews on import)."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    from robot_msg_viz import views  # noqa: F401  (imports every domain module)
