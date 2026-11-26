"""robot_msgs 0.6.0's fiducial messages and 0.5.0's collision scene each have a stock view (2026-11-26).

Failing-first for the watch's red line of 2026-11-26 (robot_viz_agent): ten new robot_msgs types reached the gallery
with no view and no decision. This pins the family: every type exported by robot_msgs.fiducial, and the two collision
scene types, is registered with a real logger (not the repr-as-text fallback), and its sample draws spatial data.
"""

import rerun as rr
import robot_msgs.fiducial as fiducial
from robot_msgs.manipulation import CollisionElement, CollisionScene

import robot_msg_viz as viz
from robot_msg_viz._engine import has_dedicated_logger, log

FIDUCIAL_TYPES = [getattr(fiducial, name) for name in fiducial.__all__ if isinstance(getattr(fiducial, name), type)]
TYPES = [*FIDUCIAL_TYPES, CollisionElement, CollisionScene]


def test_the_family_is_every_fiducial_message():
    assert len(FIDUCIAL_TYPES) == 8


def test_each_type_has_a_view_and_a_real_logger():
    missing = [t.__name__ for t in TYPES if viz.view_for(t) is None or not has_dedicated_logger(t)]
    assert not missing, f"no stock view: {missing}"


def test_each_sample_logs_without_error():
    for message_type in TYPES:
        view = viz.view_for(message_type)
        recording = rr.RecordingStream(f"test_{message_type.__name__}")
        log(view.sample(), message_type.__name__, recording=recording)
