"""Every stock view, one module per robot_msgs domain; importing this registers them all."""

from robot_msg_viz.views import (  # noqa: F401  (each module registers its TypeViews on import)
    compare,
    control,
    geometry,
    manipulation,
    runs,
    sensor,
    shape,
    std,
    streams,
    trajectory,
    vision,
)
