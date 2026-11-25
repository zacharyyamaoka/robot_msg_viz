"""The ruler's own red tests: each gate must fail on an input built to fail it. Frozen with the ruler."""

from __future__ import annotations

import rerun as rr

import robot_msg_viz as viz

from robot_msg_viz.ruler import covers, read


def test_a_view_pointing_at_nothing_is_caught(tmp_path):
    from robot_msgs.geometry import Pose

    path = viz.save(tmp_path / "pose.rrd", Pose=Pose.identity() if hasattr(Pose, "identity") else Pose())
    contents = read(path)
    assert covers(contents, "/world")
    assert not covers(contents, "/nowhere")
    assert not covers(contents, "/worl")


def test_a_recording_without_a_blueprint_is_caught(tmp_path):
    path = tmp_path / "bare.rrd"
    rec = rr.RecordingStream("bare")
    rec.save(str(path))
    rec.log("world/p", rr.Points3D([[0, 0, 0]]), static=True)
    rec.flush()
    rec.disconnect()
    contents = read(path)
    assert contents.data_paths == {"/world/p"}
    assert not contents.default_blueprint and not contents.has_viewport


def test_the_text_fallback_does_not_count_as_a_logger():
    class NotAMessage:
        pass

    assert not viz.has_dedicated_logger(NotAMessage)


def test_an_empty_recording_has_no_data(tmp_path):
    path = tmp_path / "empty.rrd"
    rec = rr.RecordingStream("empty")
    rec.save(str(path))
    rec.flush()
    rec.disconnect()
    assert not read(path).data_paths
