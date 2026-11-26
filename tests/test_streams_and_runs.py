"""The stream views, every Comparison kind and the dataset views: what lands where, read back from the .rrd.

The ruler (tests/test_contract.py) already holds each new TypeView to the gates; these pin what the gates cannot see:
two streams overlay with distinct colours, each error kind has its own paths and units, a joint Comparison still
opens on its old layout, and a run's comparisons sit under their own subtree with the worst ranked first.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from robot_msgs.compare import compare, compare_run

import robot_msg_viz as viz
from robot_msg_viz._engine import log
from robot_msg_viz.registry import VIEWS
from robot_msg_viz.ruler import read, views_in
from robot_msg_viz.views import _sim2real as s
from robot_msg_viz.views.compare import metric_unit, pick_comparison
from robot_msg_viz.views.runs import _dataset_comparison_summary, _sweep_comparison, headline


class Capture:
    """A stand-in recording that keeps each series' style (name and colour) by path."""

    def __init__(self) -> None:
        self.styles: dict[str, tuple[str, int]] = {}
        self.paths: set[str] = set()

    def log(self, path: str, *archetypes: Any, static: bool = False, **_: Any) -> None:
        self.paths.add(path)
        for archetype in archetypes:
            names, colors = getattr(archetype, "names", None), getattr(archetype, "colors", None)
            if names is not None and colors is not None:
                self.styles[path] = (str(names.as_arrow_array().to_pylist()[0]), int(colors.as_arrow_array().to_pylist()[0]))

    def send_columns(self, path: str, **_: Any) -> None:
        self.paths.add(path)


def test_two_servo_streams_overlay_on_the_same_plots_with_distinct_colours_and_labels(tmp_path):
    sim, real = s.servo_states("sim"), s.servo_states("real")
    capture = Capture()
    log(sim, "sim", recording=capture)
    log(real, "real", recording=capture)
    for joint in sim.joint_names:
        a, b = capture.styles[f"positions/{joint}/sim"], capture.styles[f"positions/{joint}/real"]
        assert a[1] != b[1], f"{joint}: sim and real share a colour"
        assert "sim" in a[0] and "real" in b[0]
    # and through the public API: one recording, one position plot holding both sources
    path = viz.save(tmp_path / "overlay.rrd", sim=sim, real=real)
    contents = read(path)
    assert {f"/positions/{j}/{source}" for j in sim.joint_names for source in ("sim", "real")} <= contents.data_paths
    origins = [str(v.origin) for v in views_in(viz.blueprint(sim=sim, real=real))]
    assert origins.count("/positions") == 1


def test_a_lone_stream_keeps_the_joint_colours():
    capture = Capture()
    log(s.servo_states("real"), recording=capture)
    from robot_msg_viz.views.trajectory import JOINT_COLORS

    first = capture.styles["positions/shoulder_pan_joint/ServoStateTrajectory"]
    r, g, b = JOINT_COLORS[0]
    assert first[1] == (r << 24) | (g << 16) | (b << 8) | 0xFF


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("positions/elbow_joint", "deg"), ("temperature/elbow_joint", "°C"), ("voltage/elbow_joint", "V"),
        ("fault_mismatch", "0/1"), ("feedforward_effort/j1", "N·m"), ("stiffness/j1", "N·m/rad"),
        ("channels/latency", "ms"), ("rotation", "deg"), ("angular_velocities/x", "deg/s"),
        ("linear_accelerations/z", "m/s²"), ("force", "N"), ("torques/y", "N·m"), ("psnr", "dB"), ("rmse", "px"),
        ("chamfer", "mm"), ("x", "mm"),
    ],
)  # fmt: skip
def test_metric_units_for_every_stream(metric, expected):
    assert metric_unit(metric)[1] == expected


KINDS = {
    "servo_state": (lambda: compare(s.servo_states("sim"), s.servo_states("real")),
                    ["/error/positions/wrist_1_joint/e", "/error/temperature/elbow_joint/e",
                     "/error/fault_mismatch/servos/e"]),
    "servo_command": (lambda: compare(s.servo_commands("sim"), s.servo_commands("real")),
                      ["/error/stiffness/elbow_joint/e", "/error/latency/e"]),
    "imu": (lambda: compare(s.imu("sim"), s.imu("real")),
            ["/error/rotation/rx/e", "/error/angular_velocity/z/e", "/error/linear_acceleration/x/e"]),
    "wrench": (lambda: compare(s.wrench("sim"), s.wrench("real")), ["/error/force/z/e", "/error/torque/x/e"]),
    "camera": (lambda: compare(s.camera("sim", frames=4), s.camera("real", frames=4)),
               ["/image/e/error", "/error/pixel/rmse/e", "/error/psnr/e"]),
    "scan": (lambda: compare(s.scan("sim"), s.scan("real")), ["/world/e/measured", "/image/e_distance/bar"]),
}  # fmt: skip


@pytest.mark.parametrize("kind", list(KINDS))
def test_every_comparison_kind_draws_its_own_layout(kind, tmp_path):
    make, paths = KINDS[kind]
    comparison = make()
    path = viz.save(tmp_path / f"{kind}.rrd", e=comparison)
    contents = read(path)
    for expected in [*paths, "/notes/e"]:
        assert expected in contents.data_paths, f"{kind}: nothing at {expected}"
    for view in views_in(viz.blueprint(e=comparison)):
        origin = str(view.origin)
        assert any(p == origin or p.startswith(origin.rstrip("/") + "/") for p in contents.data_paths), origin


def test_a_joint_comparison_opens_on_its_old_layout():
    comparison = pick_comparison()
    origins = [(type(v).__name__, str(v.origin)) for v in views_in(viz.blueprint(Comparison=comparison))]
    assert origins == [("TimeSeriesView", "/error/positions"), ("TextDocumentView", "/notes/Comparison")]
    assert VIEWS[type(comparison)].views("Comparison")[0].origin == "/error/positions"


def test_a_run_comparison_nests_each_comparison_under_its_run(tmp_path):
    result = compare_run(s.run(1.0))
    path = viz.save(tmp_path / "run.rrd", result)
    contents = read(path)
    assert "/notes/RunComparison" in contents.data_paths
    assert "/runs/pick_1x/sim-real/servo_state/error/fault_mismatch/servos/sim_real" in contents.data_paths
    assert "/runs/pick_1x/target-real/joints/notes/target_real" in contents.data_paths
    tabs = [v for v in views_in(viz.blueprint(result))]
    assert len(tabs) > len(result.comparisons), "a view group per comparison, plus the summary"


def test_the_dataset_summary_ranks_each_stream_worst_first():
    result = _sweep_comparison()
    text = _dataset_comparison_summary(result, "sweep")
    joints = [(headline(c)[1], run.run) for run in result.runs.values() for key, c in run.comparisons.items()
              if key.endswith("/servo_state")]  # fmt: skip
    worst_run = max(joints)[1]
    assert f"- **servo_state**:" in text and f"run `{worst_run}`" in text.split("- **servo_state**:")[1].split("\n")[0]
    table = text.split("**servo_state** · ")[1].split("\n\n")[1]  # that stream's table, up to the next blank line
    values = [float(line.split("|")[4].split()[0].replace("−", "-")) for line in table.splitlines()
              if line.split("|")[1].strip().isdigit()]  # fmt: skip
    assert values == sorted(values, reverse=True) and len(values) == len(joints)


def test_the_real_arm_faults_only_at_full_speed():
    assert not np.asarray(s.servo_states("real", 0.75).fault).any()
    assert np.asarray(s.servo_states("real", 1.0).fault).any()
    assert not np.asarray(s.servo_states("sim", 1.0).fault).any()
