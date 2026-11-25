"""robot_msg_viz: a stock Rerun viewer for every robot_msgs type.

    import robot_msg_viz as viz

    viz.show(pose)                           # the stock view, in the desktop viewer
    viz.show(target=planned, real=measured)  # names become entities: an overlay
    viz.save("cloud.rrd", cloud)             # blueprint inside: `rerun cloud.rrd`
    with viz.stream("arm") as live:          # live: log as messages arrive
        live.log(joint_state)

    viz.blueprint(image, detections)         # the stock layout, for your own recording
    viz.catalog()                            # every robot_msgs type, supported or not

The logger underneath is robot_msgs' own (robot_msgs.viz.rerun); this package adds a curated view,
a sample and a screenshot per type, streaming, and the gallery (docs/CONTRACT.md).
"""

from robot_msg_viz._engine import RERUN_SDK_VERSION, TIMELINE, has_dedicated_logger, log
from robot_msg_viz.api import Stream, blueprint, recording, save, save_series, show, stream, views_for
from robot_msg_viz.layout import arrange
from robot_msg_viz.registry import TypeView, catalog, register, view_for

__version__ = "0.1.0"

# WHY at import, last: each view module registers its loggers on import. Registering lazily (on the first
# catalog() call) let a first viz.show(msg) in a fresh process log through the stock logger only (wave 1, 2026-11-24).
from robot_msg_viz import views as _views  # noqa: E402,F401

__all__ = [
    "RERUN_SDK_VERSION",
    "TIMELINE",
    "Stream",
    "TypeView",
    "arrange",
    "blueprint",
    "catalog",
    "has_dedicated_logger",
    "log",
    "recording",
    "register",
    "save",
    "save_series",
    "show",
    "stream",
    "view_for",
    "views_for",
]
