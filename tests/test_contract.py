"""The contract every stock view meets (docs/CONTRACT.md). Frozen: lanes never edit this file.

A view module adds a type to the gallery by registering a TypeView; these tests then hold it to:
1. its sample is that robot_msgs type (a stream's samples too), deterministic;
2. a real logger draws it (ours or robot_msgs'), not the repr-as-text fallback;
3. the gallery .rrd it writes loads (`rerun rrd verify`), carries a default blueprint with a
   viewport, and every view in that blueprint points at data that is actually in the file;
4. streaming the sample through `Stream` puts it on the `t` timeline.
"""

from __future__ import annotations

import subprocess

import pytest
import rerun as rr
import rerun.blueprint as rrb
import robot_msgs

import robot_msg_viz as viz
from robot_msg_viz.gallery import entries, type_entries
from robot_msg_viz.registry import VIEWS, catalog

from robot_msg_viz.ruler import covers, read, rerun_cli

SUPPORTED = sorted(viz.registry.supported_types(), key=lambda c: c.__name__)
IDS = [c.__name__ for c in SUPPORTED]


def all_robot_msgs_keys() -> set[str]:
    return {k for k, c in robot_msgs.message_types().items() if c.__module__.startswith("robot_msgs.")}


def test_catalog_lists_every_robot_msgs_type_once():
    keys = [e.key for e in catalog()]
    assert len(keys) == len(set(keys))
    assert set(keys) == all_robot_msgs_keys()


def test_gallery_lists_every_type_supported_or_not():
    gallery = {e.key: e for e in type_entries()}
    assert set(gallery) == all_robot_msgs_keys()
    for entry in gallery.values():
        assert entry.supported == (entry.write is not None)


def test_gallery_entries_have_unique_keys():
    keys = [e.key for e in entries()]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("message_type", SUPPORTED, ids=IDS)
def test_sample_is_the_type_and_deterministic(message_type):
    view = VIEWS[message_type]
    assert view.summary.strip(), "a one-line summary of what the viewer shows"
    if view.series is not None:
        series = view.series()
        assert len(series) >= 2
        times = [t for t, _ in series]
        assert times == sorted(times)
        assert all(isinstance(m, message_type) for _, m in series)
        assert view.series() == series
    sample = view.sample()
    assert isinstance(sample, message_type)
    assert view.sample() == sample


@pytest.mark.parametrize("message_type", SUPPORTED, ids=IDS)
def test_a_real_logger_draws_it(message_type):
    assert viz.has_dedicated_logger(message_type), "not the repr-as-text fallback"


@pytest.mark.parametrize("message_type", SUPPORTED, ids=IDS)
def test_gallery_rrd_loads_with_a_blueprint_whose_views_have_data(message_type, tmp_path):
    entry = next(e for e in type_entries() if e.key == VIEWS[message_type].key)
    path = entry.write(tmp_path / f"{entry.slug}.rrd")
    assert path.exists() and path.stat().st_size > 0
    subprocess.run([*rerun_cli(), "rrd", "verify", str(path)], check=True, capture_output=True, timeout=120)
    contents = read(path)
    assert contents.data_paths, "the recording holds data"
    assert contents.has_viewport and contents.default_blueprint, "a default blueprint is embedded"
    views = VIEWS[message_type].views(message_type.__name__)
    assert views and all(isinstance(v, rrb.View) for v in views)
    for view in views:
        assert covers(contents, str(view.origin)), f"{type(view).__name__} at {view.origin} shows nothing"


@pytest.mark.parametrize("message_type", SUPPORTED, ids=IDS)
def test_streaming_puts_it_on_the_timeline(message_type, tmp_path):
    view = VIEWS[message_type]
    messages = [m for _, m in view.series()] if view.series is not None else [view.sample()] * 2
    path = tmp_path / "stream.rrd"
    with viz.stream("contract", spawn=False, save=path) as live:
        for i, msg in enumerate(messages):
            live.log(msg, t=0.1 * i)
    contents = read(path)
    assert contents.data_paths
    assert contents.rows >= 2, "each streamed message is its own row on t, not one static snapshot"


def test_show_save_blueprint_take_names(tmp_path):
    msg = next(iter(VIEWS.values())).sample() if VIEWS else None
    if msg is None:
        pytest.skip("no views registered yet")
    bp = viz.blueprint(a=msg, b=msg)
    assert isinstance(bp, rrb.Blueprint)
    path = viz.save(tmp_path / "two.rrd", a=msg, b=msg)
    assert read(path).data_paths
    rec, _ = viz.recording(msg)
    assert isinstance(rec, rr.RecordingStream)
