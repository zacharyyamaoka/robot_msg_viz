"""Zach's pick for the stock JointTrajectory view: V1 (2026-11-25, feedback trigger fb-0a86ed9d).

V1 is robot_msgs' stock trajectory viewer (robot_msgs.viz.rerun with ``robot=``): the URDF arm on the left half and the
tabs q · qd · qdd · qddd · q / qd / qdd · TCP on the right half, every angle in degrees. Without a URDF there is no arm,
so a lone JointTrajectory opens on V1's right-hand tabs at full width (q · qd · qdd · qddd · q / qd / qdd), never the
old one-plot-per-field view.

Behaviour, not pixels: every layout is read back from the blueprint inside the recording (Rerun's own RrdReader), the
way the contract tests read a recording back, and the data from the same file.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import rerun as rr
from rerun.chunk import RrdReader

import robot_msg_viz as viz
from robot_msg_viz.gallery import type_entries

V1_TABS = ["q", "qd", "qdd", "qddd", "q / qd / qdd"]
UR5E_URDF = Path(viz.__file__).resolve().parent / "assets" / "ur5e" / "ur5e.urdf"
# Rerun's ContainerKind enum, as the blueprint stores it
KINDS = {1: "Tabs", 2: "Horizontal", 3: "Vertical", 4: "Grid"}


def pick():
    """robot_msg_viz's JointTrajectory sample: a UR5e-class pick, minimum jerk between stops, with JointLimits."""
    from robot_msgs.trajectory import JointTrajectory

    sample = viz.view_for(JointTrajectory).sample()
    assert isinstance(sample, JointTrajectory)
    return sample


def layout(path: Path) -> dict:
    """The default blueprint in the recording as a tree: {"kind", "name", "children"} for a container, {"kind": "view",
    "class", "name", "contents"} for a view. The last blueprint store in the file wins, as in the viewer."""
    reader = RrdReader(path)
    stores = list(reader.blueprints())
    assert stores, "the recording carries a blueprint"
    latest: dict[str, dict[str, object]] = {}
    for chunk in reader.stream(store=stores[-1]):
        batch = chunk.to_record_batch()
        for name in batch.schema.names:
            if ":" not in name:
                continue
            values = [v for v in batch.column(name).to_pylist() if v is not None]
            if values:
                latest.setdefault(chunk.entity_path.lstrip("/"), {})[name] = values[-1]

    def first(value):
        return value[0] if isinstance(value, list) and value else value

    def node(path: str) -> dict:
        fields = latest[path]
        if path.startswith("view/"):
            query = latest.get(f"{path}/ViewContents", {}).get("ViewContents:query") or []
            return {"kind": "view", "class": first(fields["ViewBlueprint:class_identifier"]),
                    "name": first(fields.get("ViewBlueprint:display_name")), "contents": list(query)}  # fmt: skip
        children = fields.get("ContainerBlueprint:contents") or []
        return {"kind": KINDS[first(fields["ContainerBlueprint:container_kind"])],
                "name": first(fields.get("ContainerBlueprint:display_name")),
                "children": [node(child) for child in children]}  # fmt: skip

    root = latest["viewport"]["ViewportBlueprint:root_container"]
    while isinstance(root, list) and len(root) == 1:  # one component in the row
        root = root[0]
    if isinstance(root, list) and len(root) == 16:  # a UUID as its 16 bytes
        root = str(uuid.UUID(bytes=bytes(root)))
    return node(f"container/{first(root)}")


def views(tree: dict) -> list[dict]:
    if tree["kind"] == "view":
        return [tree]
    return [view for child in tree["children"] for view in views(child)]


def scalars(path: Path, entity: str) -> np.ndarray:
    """Every scalar logged at one entity, in time order (the series as the viewer plots it)."""
    reader = RrdReader(path)
    found: list[float] = []
    for store in reader.recordings():
        for chunk in reader.stream(store=store):
            if chunk.entity_path != entity:
                continue
            batch = chunk.to_record_batch()
            if "Scalars:scalars" in batch.schema.names:
                found += [row[0] for row in batch.column("Scalars:scalars").to_pylist() if row]
    return np.asarray(found, dtype=np.float64)


def assert_v1_with_arm(tree: dict, name: str) -> None:
    assert tree["kind"] == "Horizontal", f"the page is split in half, got {tree['kind']}"
    arm, tabs = tree["children"]
    assert arm["kind"] == "view" and arm["class"] == "3D", "the URDF arm is on the left"
    assert any(f"/world/{name}" in query for query in arm["contents"]), arm["contents"]
    assert tabs["kind"] == "Tabs", "the tabs are on the right"
    assert [tab["name"] for tab in tabs["children"]] == [*V1_TABS, "TCP"]


def assert_v1_tabs_alone(tree: dict) -> None:
    assert tree["kind"] == "Tabs", f"V1's tabs at full width, got {tree['kind']}"
    assert [tab["name"] for tab in tree["children"]] == V1_TABS
    assert not [v for v in views(tree) if v["class"] == "3D"], "no arm without a URDF"
    # not the old view: one plot per field (joint position / velocity / acceleration) beside a joint table
    names = {v["name"] for v in views(tree)}
    assert not names & {"joint position (deg)", "joint velocity (deg/s)", "joint acceleration (deg/s²)"}


# WHY (Zach, 2026-11-25): "when we are look at just a normal trajectory without comparison than I think it makes sense
# to just keep with V1. that should basically just be the stock like robot msg viz viewer for a joint trajectory"


def test_save_with_a_urdf_opens_in_v1_the_arm_left_the_tabs_right_in_degrees(tmp_path):
    traj = pick()
    path = viz.save(tmp_path / "pick.rrd", plan=traj, robot=UR5E_URDF)
    assert_v1_with_arm(layout(path), "plan")
    elbow = list(traj.joint_names).index("elbow_joint")
    np.testing.assert_allclose(scalars(path, "/positions/elbow_joint/plan"), np.degrees(traj.positions[:, elbow]))
    q_tab = next(tab for tab in layout(path)["children"][1]["children"] if tab["name"] == "q")
    assert all("[°]" in view["name"] for view in views(q_tab)), "every q strip is in degrees"


def test_the_gallery_sample_is_v1_on_a_real_ur5e(tmp_path):
    entry = next(e for e in type_entries() if e.key == "trajectory/JointTrajectory")
    path = entry.write(tmp_path / "gallery.rrd")
    assert_v1_with_arm(layout(path), "JointTrajectory")
    reader = RrdReader(path)
    arm = {chunk.entity_path for store in reader.recordings() for chunk in reader.stream(store=store)
           if chunk.entity_path.startswith("/world/JointTrajectory/robot/")}  # fmt: skip
    assert arm, "the UR5e's links are in the file"


def test_save_without_a_urdf_opens_on_v1s_tabs_not_the_old_view(tmp_path):
    traj = pick()
    path = viz.save(tmp_path / "plan.rrd", traj)
    assert_v1_tabs_alone(layout(path))
    np.testing.assert_allclose(scalars(path, "/positions/shoulder_pan_joint/JointTrajectory"),
                               np.degrees(traj.positions[:, 0]))  # fmt: skip
    assert scalars(path, "/jerks/wrist_3_joint/JointTrajectory").size, "the qddd tab has its data"


def test_log_into_your_own_recording_with_the_stock_blueprint_is_v1(tmp_path):
    traj = pick()
    for robot, check in ((None, assert_v1_tabs_alone), (UR5E_URDF, lambda tree: assert_v1_with_arm(tree, "mine"))):
        path = tmp_path / f"own-{robot is None}.rrd"
        stream = rr.RecordingStream("own")
        stream.save(str(path), default_blueprint=viz.blueprint(mine=traj, robot=robot))
        options = {} if robot is None else {"robot": robot}
        viz.log(traj, "mine", recording=stream, **options)
        stream.flush()
        stream.disconnect()
        check(layout(path))
        assert scalars(path, "/velocities/elbow_joint/mine").size
