"""The gallery site: one landing page with a Types tab and a Use cases tab, and one page per entry.

    python -m robot_msg_viz.site build OUT [--shots DIR]   # OUT/index.html, pages, .rrd, screenshots, gallery.json
    python -m robot_msg_viz.site serve OUT [--port 4493]   # serve OUT and a local Rerun web viewer

Everything is generated from `robot_msg_viz.gallery.entries()`: the types (supported or not) and the use cases
every installed plugin registers, so a new view or dashboard shows up here with no site edit. Pure stdlib HTML,
like robotmsgs.com's builder (robot_msgs/website/build), whose look it shares.
"""

from robot_msg_viz.site.build import build

__all__ = ["build"]
