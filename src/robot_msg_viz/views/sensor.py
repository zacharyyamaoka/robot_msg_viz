"""Stock views of robot_msgs.sensor types (docs/CONTRACT.md: one TypeView per type, registered on import).

| Type | What the viewer shows | Logger |
|---|---|---|
| JointState | position (deg), velocity (deg/s) and effort over t, one colour per joint, and a labelled bar per joint at the playhead | ours: the stock one writes a static table, so a stream had no history |
| Image | the picture; a depth image through the Turbo colour map (the viewer's default) | robot_msgs' |
| CameraInfo | the pinhole frustum in its own optical frame, its pixel grid and principal point on the image plane, and the intrinsics as text | ours: the stock one logs a bare pinhole at /camera/<name>, which no 3D view shows |
| PointCloud | the points, coloured by their rgb channel, else intensity, else height | ours: the stock one leaves a cloud without rgb one flat colour, so its shape does not read |
| Imu | acceleration (m/s²), angular velocity (deg/s) and roll/pitch/yaw (deg) over t, and the orientation as a frame | ours: robot_msgs has none |

The Image and PointCloud samples are one tabletop scene seen by the CameraInfo sample's camera, ray-cast here in
numpy (deterministic, no files, no network), so the three gallery cards show the same table.

Not supported, on purpose: CameraSettings (exposure, gain, white balance, gamma, focus, frame rate) is the
configuration a camera captured with, with nothing of its own to draw (docs/CONTRACT.md: never fake a view); it keeps
the repr-as-text fallback, and robot_viz_agent lists it under configuration in its unsupported.json.
"""  # noqa: E501

from __future__ import annotations

# WHY empty (2026-11-25): this module registers its TypeViews on import; its functions are sample data and helpers,
# not API. The public API is robot_msg_viz's own __all__ (show/save/stream/blueprint), and robot-sdk's scanner reads
# __all__ to decide what a package publishes.
__all__: list[str] = []

import functools
import math
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from robot_msgs.geometry import Quaternion, Transform, Vector3
from robot_msgs.sensor import CameraInfo, Image, Imu, JointState, PointCloud
from robot_msgs.std import Header

from robot_msg_viz._engine import Timed, log, part
from robot_msg_viz.registry import TypeView, register
from robot_msg_viz.views._common import DEG, rng

STAMP = 1_764_000_000.0  # a fixed epoch second (2025-11-24) for the samples' headers, so they are deterministic
AXES = ((220, 50, 60), (40, 160, 70), (50, 105, 230))  # x, y, z: red, green, blue (the geometry views' triad)
GREY = (130, 135, 145)
INK = (70, 75, 90)
# One colour per joint, in joint order (Tableau 10, cycling past ten joints).
JOINT_COLOURS = (
    (78, 121, 167), (242, 142, 43), (89, 161, 79), (225, 87, 89), (176, 122, 161),
    (118, 183, 178), (237, 201, 72), (255, 157, 167), (156, 117, 95), (186, 176, 172),
)  # fmt: skip
THIN = rr.Radius.ui_points(1.0)


def _styles(recording: Any) -> Any:
    """Where legend names, colours and fixed scenery go: static, once, even inside a stream.

    WHY: `_engine.Timed` lifts every log onto the timeline so a streamed message can be scrubbed; a legend or a
    scale is not the message's data, so it stays static instead of being copied once per message.
    """
    return recording._inner if isinstance(recording, Timed) else recording


def _legend(column: str, name: str, default: str) -> str:
    """The legend text: the column alone for the default name, else the column and the caller's name."""
    return column if name == default else f"{column} · {name}"


def _joint_colour(index: int) -> tuple[int, int, int]:
    return JOINT_COLOURS[index % len(JOINT_COLOURS)]


# ── JointState ───────────────────────────────────────────────────────────────

_ROW = 84.0  # scene units between joint rows in the bar view, whose x axis is degrees
_TICKS = (-180, -90, 0, 90, 180)
# WHY the label rides this high: Rerun centres a point's label on it horizontally and hangs it ~4-24 screen px
# below, with no alignment option; at the scales a gallery panel gives (0.6-1.5 px per degree) this keeps each
# row's "name  value" pill clear above its own bar, nearer its bar than the row above. Measured in the web viewer.
_LABEL_RISE = 48.0


@log.register
def _(state: JointState, name: str | None = None, *, recording: Any) -> None:
    """One series per joint and field at ``<field>/<joint>/<name>`` (robot_msgs' JointTrajectory layout, so a
    measured state overlays a planned trajectory), and the bar view at ``joints/<name>``.

    WHY every joint in degrees: robot_msgs carries no joint type, so, like the stock JointTrajectory logger's
    ``degrees=True``, every joint is read as revolute (a prismatic joint's metres would show ×57.3).
    """
    name = name or "JointState"
    label, styles = part(name), _styles(recording)
    fields = (("positions", state.position, DEG), ("velocities", state.velocity, DEG), ("effort", state.effort, 1.0))
    for field_name, values, scale in fields:
        if values is None:  # not measured: nothing drawn, never zeros
            continue
        for index, joint in enumerate(state.name):
            path = f"{field_name}/{part(joint)}/{label}"
            legend = _legend(joint, name, "JointState")
            styles.log(path, rr.SeriesLines(names=legend, colors=_joint_colour(index)), static=True)
            recording.log(path, rr.Scalars(float(values[index]) * scale))
    _joint_bars(state, f"joints/{label}", recording, styles)


def _joint_bars(state: JointState, path: str, recording: Any, styles: Any) -> None:
    """A bar per joint from 0° to its position on a ±180° scale, its name and value centred above it.

    WHY a 2D view and not Rerun's BarChart: a BarChart has no per-bar labels, so its bars would read 0..5, not
    joint names. A joint past ±180° draws past the scale, never clipped to it.
    """
    count = len(state.name)
    if count == 0:
        return
    degrees = np.asarray(state.position, dtype=np.float64) * DEG
    bars = np.arange(count) * _ROW
    colours = [_joint_colour(i) for i in range(count)]
    top, bottom = -_LABEL_RISE, bars[-1] + 0.2 * _ROW
    styles.log(
        f"{path}/scale",
        rr.LineStrips2D(
            [[[tick, top], [tick, bottom]] for tick in _TICKS], colors=[(*GREY, 90)], radii=THIN, draw_order=1.0
        ),
        static=True,
    )
    styles.log(
        f"{path}/scale_labels",
        # WHY on top: a label hangs below its anchor, so along the bottom edge the pills fell outside the
        # view's framing and were cut off in a short panel; anchored above the first row they stay inside.
        rr.Points2D([[tick, top - 40.0] for tick in _TICKS], labels=[f"{tick}°" for tick in _TICKS], colors=[GREY],
                    radii=0.01),
        static=True,
    )
    styles.log(
        f"{path}/tracks",
        rr.LineStrips2D([[[-180.0, y], [180.0, y]] for y in bars], colors=[(*GREY, 160)], radii=THIN, draw_order=2.0),
        static=True,
    )
    recording.log(
        f"{path}/bars",
        rr.LineStrips2D(
            [[[0.0, y], [value, y]] for y, value in zip(bars, degrees, strict=True)],
            colors=colours,
            radii=8.0,
            draw_order=3.0,
        ),
        static=True,
    )
    recording.log(
        f"{path}/labels",
        rr.Points2D(
            [[0.0, y - _LABEL_RISE] for y in bars],
            labels=[f"{joint}  {value:+.1f}°" for joint, value in zip(state.name, degrees, strict=True)],
            colors=colours,
            radii=0.01,
        ),
        static=True,
    )


def _joint_state_views(name: str) -> list[rrb.View]:
    return [
        rrb.Spatial2DView(
            origin=f"/joints/{part(name)}", name=_legend("joint positions now (deg)", name, "JointState")
        ),
        rrb.TimeSeriesView(origin="/positions", name="position (deg)"),
        # WHY no legend on these two: the bars and the position plot name every joint in the same colours, and
        # in a quarter-width panel a six-line legend hides the curves (hover still names a line).
        rrb.TimeSeriesView(origin="/velocities", name="velocity (deg/s)", plot_legend=rrb.PlotLegend(visible=False)),
        rrb.TimeSeriesView(origin="/effort", name="effort (N·m)", plot_legend=rrb.PlotLegend(visible=False)),
    ]


ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
_ARM_WAYPOINTS_DEG = np.array([
    [0.0, -90.0, 90.0, -90.0, -90.0, 0.0],  # home
    [38.0, -62.0, 98.0, -126.0, -90.0, 75.0],  # above the part, the gripper turned to it
    [38.0, -48.0, 104.0, -146.0, -88.0, 75.0],  # down on it
    [-30.0, -75.0, 95.0, -110.0, -92.0, 10.0],  # over the bin
])  # fmt: skip
_ARM_SEGMENT_S = 1.5
_ARM_RATE_HZ = 25.0


@functools.cache
def _arm_states() -> tuple[tuple[float, JointState], ...]:
    """A UR-style 6-joint arm on a pick: home → above the part → down → over the bin, minimum-jerk segments.

    Effort is a gravity + inertia + friction model with a little seeded sensor noise, in N·m.
    """
    segments = len(_ARM_WAYPOINTS_DEG) - 1
    t = np.round(np.arange(0.0, segments * _ARM_SEGMENT_S + 1e-9, 1.0 / _ARM_RATE_HZ), 6)
    segment = np.minimum((t // _ARM_SEGMENT_S).astype(int), segments - 1)
    tau = (t - segment * _ARM_SEGMENT_S) / _ARM_SEGMENT_S
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / _ARM_SEGMENT_S
    dds = (60 * tau - 180 * tau**2 + 120 * tau**3) / _ARM_SEGMENT_S**2
    start = np.radians(_ARM_WAYPOINTS_DEG[segment])
    step = np.radians(_ARM_WAYPOINTS_DEG[segment + 1]) - start
    q, qd, qdd = start + step * s[:, None], step * ds[:, None], step * dds[:, None]
    lift, elbow, wrist = q[:, 1], q[:, 1] + q[:, 2], q[:, 1] + q[:, 2] + q[:, 3]
    gravity = np.zeros_like(q)
    gravity[:, 1] = 38.0 * np.cos(lift) + 11.0 * np.cos(elbow)
    gravity[:, 2] = 11.0 * np.cos(elbow) + 1.2 * np.cos(wrist)
    gravity[:, 3] = 1.2 * np.cos(wrist)
    inertia = np.array([3.2, 4.5, 1.6, 0.25, 0.2, 0.1]) * qdd
    friction = np.array([1.5, 1.5, 1.0, 0.4, 0.4, 0.3]) * np.tanh(qd / 0.05)
    effort = gravity + inertia + friction + rng(7).normal(0.0, 0.15, q.shape)
    return tuple(
        (float(t[i]), JointState(
            name=ARM_JOINTS, position=q[i], velocity=qd[i], effort=effort[i],
            header=Header(stamp=STAMP + float(t[i])),
        ))
        for i in range(len(t))
    )  # fmt: skip


def _arm_series() -> list[tuple[float, JointState]]:
    return list(_arm_states())


def _arm_sample() -> JointState:
    return _arm_states()[50][1]  # t = 2.0 s: on the way down to the part, every joint moving


_STREAM_CODE = """import robot_msg_viz as viz

# live: opens the viewer; time comes from each message's header.stamp
with viz.stream("{stream}") as live:
    for {var} in {source}:  # your robot_msgs.sensor.{type} messages
        live.log({var})

# or a file, t in seconds (open it with: rerun {stream}.rrd)
viz.save_series("{stream}.rrd", [(i / {rate}, {var}) for i, {var} in enumerate({source})])"""

register(TypeView(
    type=JointState,
    summary="Each joint's position (deg), velocity (deg/s) and effort over time, one colour per joint, and a "
    "labelled bar per joint showing where it is now.",
    sample=_arm_sample,
    views=_joint_state_views,
    series=_arm_series,
    code=_STREAM_CODE.format(stream="arm", var="state", source="joint_states", type="JointState", rate=_ARM_RATE_HZ),
    tags=("plot", "stream"),
))  # fmt: skip


# ── one tabletop scene for Image, CameraInfo and PointCloud ──────────────────

# A RealSense-D435-like colour camera, rectified (plumb_bob, all zero), looking down at a table.
CAMERA = CameraInfo.from_intrinsics(
    615.3, 615.1, 321.8, 238.4, 640, 480,
    d=np.zeros(5), header=Header(stamp=STAMP, frame_id="camera_color_optical_frame"),
)  # fmt: skip
_EYE = np.array([0.0, -0.62, 0.50])  # metres, world z up, table top at z = 0
_LOOK_AT = np.array([0.0, 0.04, 0.03])
_LIGHT = np.array([-0.45, -0.35, 1.0]) / np.linalg.norm([-0.45, -0.35, 1.0])
_EPS = 1e-6
_TABLE = (-0.8, 0.8, -0.65, 0.55)  # x min, x max, y min, y max (m)
_WALL_Y = 0.55
_WORKSPACE = ((-0.42, -0.26, -0.01), (0.42, 0.42, 0.4))  # the cloud's crop box (m, world)


def _world_from_optical() -> np.ndarray:
    """4×4 pose of the camera's optical frame (x right, y down, z forward) in the world."""
    forward = (_LOOK_AT - _EYE) / np.linalg.norm(_LOOK_AT - _EYE)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    matrix = np.eye(4)
    matrix[:3, :3] = np.stack([right, down, forward], axis=1)
    matrix[:3, 3] = _EYE
    return matrix


def _srgb(colour: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(colour, dtype=np.float64) ** 2.2  # shade in linear light


# The objects on the table: (kind, parameters, sRGB albedo, specular strength).
_OBJECTS = (
    ("sphere", {"centre": (-0.17, 0.02, 0.045), "radius": 0.045}, (0.80, 0.16, 0.13), 0.35),
    ("cylinder", {"centre": (0.02, 0.21), "radius": 0.034, "height": 0.125}, (0.18, 0.58, 0.30), 0.30),
    ("box", {"centre": (0.15, 0.05, 0.035), "half": (0.07, 0.045, 0.035), "yaw": 0.45}, (0.17, 0.36, 0.72), 0.08),
    ("cylinder", {"centre": (-0.05, -0.10), "radius": 0.042, "height": 0.03}, (0.95, 0.72, 0.15), 0.20),
)


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b)


def _sphere(o: np.ndarray, d: np.ndarray, centre: Any, radius: float) -> tuple[np.ndarray, np.ndarray]:
    oc = o - np.asarray(centre)
    a, b, c = _dot(d, d), 2.0 * _dot(d, oc), _dot(oc, oc) - radius**2
    disc = b * b - 4.0 * a * c
    t = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a)
    t = np.where((disc > 0.0) & (t > _EPS), t, np.inf)
    hit = o + d * np.where(np.isfinite(t), t, 0.0)[:, None]
    return t, (hit - np.asarray(centre)) / radius


def _cylinder(
    o: np.ndarray, d: np.ndarray, centre: Any, radius: float, height: float
) -> tuple[np.ndarray, np.ndarray]:
    """A vertical cylinder standing on the table (z from 0 to height), with its top cap."""
    ox, oy = o[:, 0] - centre[0], o[:, 1] - centre[1]
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2.0 * (ox * d[:, 0] + oy * d[:, 1])
    c = ox**2 + oy**2 - radius**2
    disc = b * b - 4.0 * a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        side = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a)
        cap = (height - o[:, 2]) / d[:, 2]
    z = o[:, 2] + side * d[:, 2]
    side = np.where((disc > 0.0) & (side > _EPS) & (z >= 0.0) & (z <= height), side, np.inf)
    cx, cy = ox + cap * d[:, 0], oy + cap * d[:, 1]
    cap = np.where((cap > _EPS) & (cx**2 + cy**2 <= radius**2), cap, np.inf)
    t = np.minimum(side, cap)
    hit = o + d * np.where(np.isfinite(t), t, 0.0)[:, None]
    normal = np.stack([(hit[:, 0] - centre[0]) / radius, (hit[:, 1] - centre[1]) / radius, np.zeros(len(t))], axis=1)
    normal[cap < side] = (0.0, 0.0, 1.0)
    return t, normal


def _box(o: np.ndarray, d: np.ndarray, centre: Any, half: Any, yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """A box turned `yaw` about z (slab test in the box's own frame)."""
    c, s = math.cos(yaw), math.sin(yaw)
    to_box = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    local_o, local_d = (o - np.asarray(centre)) @ to_box.T, d @ to_box.T
    half = np.asarray(half)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-half - local_o) / local_d, (half - local_o) / local_d
    near_each = np.nan_to_num(np.minimum(t1, t2), nan=-np.inf)
    far_each = np.nan_to_num(np.maximum(t1, t2), nan=np.inf)
    near, far = near_each.max(axis=1), far_each.min(axis=1)
    t = np.where((near <= far) & (near > _EPS), near, np.inf)
    axis = near_each.argmax(axis=1)
    local_n = np.zeros_like(o)
    rows = np.arange(len(o))
    local_n[rows, axis] = -np.sign(local_d[rows, axis])
    return t, local_n @ to_box


def _bound(kind: str, params: dict[str, Any]) -> tuple[np.ndarray, float]:
    """A sphere around the object, so most rays skip its exact test."""
    if kind == "sphere":
        return np.asarray(params["centre"], dtype=float), params["radius"]
    if kind == "cylinder":
        half_height = 0.5 * params["height"]
        return np.array([*params["centre"], half_height]), math.hypot(params["radius"], half_height)
    return np.asarray(params["centre"], dtype=float), float(np.linalg.norm(params["half"]))


def _hit_object(kind: str, o: np.ndarray, d: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """(t, normal) per ray, t = inf on a miss; only rays passing the bounding sphere get the exact test."""
    centre, radius = _bound(kind, params)
    to_centre = centre - o
    along = _dot(to_centre, d)
    near = np.flatnonzero((along > 0.0) & (_dot(to_centre, to_centre) - along**2 / _dot(d, d) <= radius**2))
    t, normal = np.full(len(d), np.inf), np.zeros((len(d), 3))
    if near.size:
        exact = {"sphere": _sphere, "cylinder": _cylinder, "box": _box}[kind]
        t[near], normal[near] = exact(np.asarray(o)[near], np.asarray(d)[near], **params)
    return t, normal


def _footprint_distance(kind: str, xy: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Horizontal distance from table points to an object's footprint (for the contact shadow)."""
    if kind == "sphere":
        return np.hypot(xy[:, 0] - params["centre"][0], xy[:, 1] - params["centre"][1]) - 0.6 * params["radius"]
    if kind == "cylinder":
        return np.hypot(xy[:, 0] - params["centre"][0], xy[:, 1] - params["centre"][1]) - params["radius"]
    c, s = math.cos(params["yaw"]), math.sin(params["yaw"])
    dx, dy = xy[:, 0] - params["centre"][0], xy[:, 1] - params["centre"][1]
    local = np.abs(np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)) - np.asarray(params["half"][:2])
    return np.hypot(*np.maximum(local, 0.0).T) + np.minimum(local.max(axis=1), 0.0)


def _wood(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Oak planks 18 cm wide running away from the camera: fine grain along each plank, a slightly different tint
    per plank, a dark seam between them."""
    plank = np.floor((x + 0.09) / 0.18)
    tint = 0.93 + 0.07 * np.sin(plank * 12.9898) * np.cos(plank * 4.1414)
    grain = np.sin(2 * np.pi * (x * 38.0 + 0.35 * np.sin(y * 6.0 + plank) + 0.08 * np.sin(y * 29.0 + x * 3.0)))
    seam = np.abs(((x + 0.09) / 0.18) % 1.0 - 0.5) > 0.488
    shade = tint * (0.95 + 0.05 * grain) * np.where(seam, 0.55, 1.0)
    return _srgb((0.68, 0.50, 0.33))[None, :] * shade[:, None]


@functools.cache
def _render(camera: CameraInfo, supersample: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Ray-cast the tabletop through `camera` placed at the scene's eye: (H, W, 3) uint8 rgb and (H, W) float32
    depth in metres along the optical axis (0 where nothing is hit). `supersample` averages s×s rays per pixel."""
    sample = supersample
    rows, cols = np.mgrid[0 : camera.height * sample, 0 : camera.width * sample]
    u, v = (cols + 0.5) / sample - 0.5, (rows + 0.5) / sample - 0.5
    rays = np.stack([(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy, np.ones_like(u, dtype=float)], axis=-1)
    pose = _world_from_optical()
    d = rays.reshape(-1, 3) @ pose[:3, :3].T  # optical z = 1, so the hit distance t IS the depth
    o = np.broadcast_to(pose[:3, 3], d.shape)
    n = len(d)
    t = np.full(n, np.inf)
    normal = np.zeros((n, 3))
    albedo = np.zeros((n, 3))
    shine = np.zeros(n)
    on_table = np.zeros(n, dtype=bool)
    # the table top (z = 0) and the wall behind it (y = _WALL_Y)
    with np.errstate(divide="ignore", invalid="ignore"):
        table_t = -o[:, 2] / d[:, 2]
        wall_t = (_WALL_Y - o[:, 1]) / d[:, 1]
    tx, ty = o[:, 0] + table_t * d[:, 0], o[:, 1] + table_t * d[:, 1]
    x0, x1, y0, y1 = _TABLE
    table_t = np.where((table_t > _EPS) & (tx >= x0) & (tx <= x1) & (ty >= y0) & (ty <= y1), table_t, np.inf)
    wall_z = o[:, 2] + wall_t * d[:, 2]
    wall_t = np.where((wall_t > _EPS) & (wall_z >= 0.0), wall_t, np.inf)
    closer = table_t < t
    t[closer], normal[closer], on_table[closer] = table_t[closer], (0.0, 0.0, 1.0), True
    albedo[closer] = _wood(tx[closer], ty[closer])
    closer = wall_t < t
    t[closer], normal[closer], on_table[closer] = wall_t[closer], (0.0, -1.0, 0.0), False
    albedo[closer] = _srgb((0.80, 0.80, 0.78)) * (0.92 + 0.08 * np.clip(wall_z[closer] / 0.4, 0, 1))[:, None]
    for kind, params, colour, specular in _OBJECTS:
        hit_t, hit_n = _hit_object(kind, o, d, params)
        closer = hit_t < t
        t[closer], normal[closer], on_table[closer] = hit_t[closer], hit_n[closer], False
        albedo[closer], shine[closer] = _srgb(colour), specular
    hit = np.isfinite(t)
    point = o + d * np.where(hit, t, 0.0)[:, None]
    # hard shadows from the objects, a soft contact shadow where they stand, a sky term, Blinn-Phong highlights
    lambert = np.clip(normal @ _LIGHT, 0.0, 1.0)
    lit = np.ones(n)
    facing = np.flatnonzero(hit & (lambert > 0.0))  # WHY a subset: shadow rays only where the light can reach
    shadow_o, shadow_d = point[facing] + normal[facing] * 1e-4, np.broadcast_to(_LIGHT, (len(facing), 3))
    for kind, params, _, _ in _OBJECTS:
        lit[facing[np.isfinite(_hit_object(kind, shadow_o, shadow_d, params)[0])]] = 0.0
    contact = np.ones(n)
    table = np.flatnonzero(on_table)
    for kind, params, _, _ in _OBJECTS:
        gap = np.maximum(_footprint_distance(kind, point[table, :2], params), 0.0)
        contact[table] *= 1.0 - 0.45 * np.exp(-gap / 0.012)
    sky = 0.22 + 0.10 * (0.5 + 0.5 * normal[:, 2])
    view = -d / np.linalg.norm(d, axis=1, keepdims=True)
    halfway = (view + _LIGHT) / np.linalg.norm(view + _LIGHT, axis=1, keepdims=True)
    highlight = shine * np.clip(_dot(normal, halfway), 0.0, 1.0) ** 60 * lit
    linear = albedo * (sky * contact + 0.78 * lambert * lit)[:, None] + highlight[:, None]
    linear[~hit] = _srgb((0.25, 0.25, 0.27))
    rgb = np.clip(linear, 0.0, 1.0) ** (1 / 2.2)
    height, width = camera.height, camera.width
    rgb = rgb.reshape(height, sample, width, sample, 3).mean(axis=(1, 3))
    depth = np.where(hit, t, 0.0).reshape(height, sample, width, sample)[:, 0, :, 0]
    return np.round(rgb * 255).astype(np.uint8), depth.astype(np.float32)


def _tabletop_image() -> Image:
    rgb, _ = _render(CAMERA.scaled(0.75), 2)  # 480 × 360: crisp, and small enough for the gallery's .rrd
    return Image.from_array(rgb, header=Header(stamp=STAMP, frame_id="camera_color_optical_frame"))


@functools.cache
def _tabletop_cloud() -> PointCloud:
    """The same table as an RGB-D camera sees it: depth with ~1 mm of seeded noise, back-projected, moved into
    the world frame and cropped to the robot's workspace box on the table (the wall and table edges dropped, as a
    tabletop pipeline does), coloured by the camera's rgb."""
    camera = CAMERA.scaled(0.5)  # 320 × 240 pixels
    rgb, depth = _render(camera, 1)
    noisy = np.where(depth > 0, depth + rng(3).normal(0.0, 0.001, depth.shape), 0.0).astype(np.float32)
    cloud = PointCloud.from_depth(Image.from_array(noisy), camera, Image.from_array(rgb))
    cloud = cloud.transformed(Transform.from_matrix(_world_from_optical()))
    cloud = cloud.cropped(_WORKSPACE[0], _WORKSPACE[1])
    return cloud.replace(header=Header(stamp=STAMP, frame_id="world"))


# ── Image ────────────────────────────────────────────────────────────────────

register(TypeView(
    type=Image,
    summary="The picture as the camera took it; a depth image is coloured by distance (Turbo).",
    sample=_tabletop_image,
    views=lambda name: [rrb.Spatial2DView(origin=f"/image/{part(name)}", name=name)],
    tags=("2d",),
))  # fmt: skip


# ── CameraInfo ───────────────────────────────────────────────────────────────

_IMAGE_PLANE_M = 0.4  # where the frustum's image plane is drawn: a display choice, the intrinsics fix only its shape


@log.register
def _(info: CameraInfo, name: str | None = None, *, recording: Any) -> None:
    """The camera at the origin of its own optical frame (it carries no pose): its x/y/z triad, the pinhole
    frustum, and on the image plane the pixel grid, the u/v axes and the principal point. The numbers go to
    ``notes/<name>``."""
    name = name or "CameraInfo"
    label = part(name)
    frame, image = f"world/{label}", f"world/{label}/image"
    width, height = info.width, info.height
    recording.log(frame, rr.TransformAxes3D(0.1), static=True)
    recording.log(
        image,
        rr.Pinhole(
            image_from_camera=np.asarray(info.k),
            width=width,
            height=height,
            image_plane_distance=_IMAGE_PLANE_M,
            color=INK,
        ),
        static=True,
    )
    step = max(1, round(width / 8))
    grid = [[[x, 0.0], [x, height]] for x in range(step, width, step)]
    grid += [[[0.0, y], [width, y]] for y in range(step, height, step)]
    recording.log(f"{image}/grid", rr.LineStrips2D(grid, colors=[(200, 203, 210)], radii=THIN), static=True)
    border = [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height], [0.0, 0.0]]
    recording.log(
        f"{image}/border", rr.LineStrips2D([border], colors=[GREY], radii=rr.Radius.ui_points(1.5)), static=True
    )
    arrow = 0.18 * width
    recording.log(
        f"{image}/uv",
        rr.Arrows2D(origins=[[0.0, 0.0], [0.0, 0.0]], vectors=[[arrow, 0.0], [0.0, arrow]], colors=AXES[:2],
                   labels=["u (px)", "v (px)"], radii=rr.Radius.ui_points(2.0), draw_order=50.0),
        static=True,
    )
    recording.log(
        f"{image}/principal_point",
        rr.Points2D(
            [[info.cx, info.cy]],
            colors=[AXES[2]],
            radii=rr.Radius.ui_points(5.0),
            labels=[f"principal point ({_num(info.cx)}, {_num(info.cy)})"],
        ),
        static=True,
    )
    recording.log(f"notes/{label}", rr.TextDocument(_intrinsics(info, name), media_type="text/markdown"), static=True)


def _num(value: float) -> str:
    return f"{float(value):.6g}"


def _intrinsics(info: CameraInfo, name: str) -> str:
    """The intrinsics as a markdown table; angles in degrees, every number at full precision (6 significant)."""
    wide = math.degrees(math.atan(info.cx / info.fx) + math.atan((info.width - info.cx) / info.fx))
    high = math.degrees(math.atan(info.cy / info.fy) + math.atan((info.height - info.cy) / info.fy))
    corner_a = np.array([-info.cx / info.fx, -info.cy / info.fy, 1.0])
    corner_b = np.array([(info.width - info.cx) / info.fx, (info.height - info.cy) / info.fy, 1.0])
    diagonal = math.degrees(math.acos(corner_a @ corner_b / (np.linalg.norm(corner_a) * np.linalg.norm(corner_b))))
    coefficients = ", ".join(_num(c) for c in info.d) if len(info.d) else "none"
    frame = info.header.frame_id or "(no frame_id)"
    k = np.asarray(info.k)
    matrix = "\n".join("  [" + ", ".join(f"{_num(c):>8}" for c in row) + "]" for row in k)
    return "\n".join([
        f"### {name}  ·  `{frame}`",
        "",
        "| | |",
        "|---|---|",
        f"| image | {info.width} × {info.height} px |",
        f"| focal length | fx {_num(info.fx)} px · fy {_num(info.fy)} px |",
        f"| principal point | cx {_num(info.cx)} px · cy {_num(info.cy)} px |",
        f"| field of view | {wide:.1f}° wide · {high:.1f}° high · {diagonal:.1f}° diagonal |",
        f"| distortion | {info.distortion_model}: {coefficients} |",
        "",
        "camera matrix K",
        "```",
        matrix,
        "```",
    ])  # fmt: skip


CAMERA_EYE = rrb.EyeControls3D(position=(0.50, -0.32, -0.12), look_target=(0.0, 0.0, 0.2), eye_up=(0.0, -1.0, 0.0))


def _camera_views(name: str) -> list[rrb.View]:
    label = part(name)
    return [
        # WHY no ground grid: the camera's frame has no "up", so the grid lay edge-on through the frustum.
        rrb.Spatial3DView(
            origin="/world", name="world", eye_controls=CAMERA_EYE, line_grid=rrb.LineGrid3D(visible=False)
        ),
        rrb.Spatial2DView(origin=f"/world/{label}/image", name="image plane (px)"),
        rrb.TextDocumentView(origin=f"/notes/{label}", name="intrinsics"),
    ]


register(TypeView(
    type=CameraInfo,
    summary="The camera's view cone (pinhole frustum) with its pixel grid and principal point, and the "
    "intrinsics (size, focal length, field of view in degrees, distortion) as a table.",
    sample=lambda: CAMERA,
    views=_camera_views,
    tags=("3d", "text"),
))  # fmt: skip


# ── PointCloud ───────────────────────────────────────────────────────────────

# A perceptually even ramp (viridis' anchors), low → high.
_RAMP = np.array([(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)], dtype=np.float64)


def _ramp(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    low, high = (values[finite].min(), values[finite].max()) if finite.any() else (0.0, 1.0)
    unit = np.clip((values - low) / (high - low), 0.0, 1.0) if high > low else np.zeros_like(values)
    position = np.where(finite, unit, 0.0) * (len(_RAMP) - 1)
    index = np.minimum(position.astype(int), len(_RAMP) - 2)
    frac = (position - index)[:, None]
    colours = _RAMP[index] * (1 - frac) + _RAMP[index + 1] * frac
    colours[~finite] = GREY
    return colours.astype(np.uint8)


def _cloud_colours(cloud: PointCloud) -> np.ndarray | None:
    """rgb (or colors/color) as given; else intensity, else height (z), on a ramp."""
    channels = cloud.channels or {}
    for key in ("rgb", "colors", "color"):
        value = channels.get(key)
        if value is not None and np.asarray(value).shape[-1:] in ((3,), (4,)):
            return np.asarray(value)
    if len(cloud) == 0:
        return None
    intensity = channels.get("intensity")
    if intensity is not None and np.asarray(intensity).size == len(cloud):
        return _ramp(intensity)
    return _ramp(np.asarray(cloud.points)[:, 2])


CLOUD_POINT_RADIUS = rr.Radius.ui_points(2.0)  # a dense cloud reads as a surface at any zoom


@log.register
def _(cloud: PointCloud, name: str | None = None, *, recording: Any) -> None:
    recording.log(
        f"world/{part(name or 'PointCloud')}",
        rr.Points3D(np.asarray(cloud.points), colors=_cloud_colours(cloud), radii=CLOUD_POINT_RADIUS),
        static=True,
    )


register(TypeView(
    type=PointCloud,
    summary="The points in 3D, coloured by their rgb channel, else by intensity, else by height.",
    sample=_tabletop_cloud,
    views=lambda name: [rrb.Spatial3DView(origin="/world", name="world")],
    tags=("3d",),
))  # fmt: skip


# ── Imu ──────────────────────────────────────────────────────────────────────


_IMU_BOARD = (0.035, 0.022, 0.004)  # half sizes (m) of the board drawn for the sensor: a display choice


def _orientation_unknown(imu: Imu) -> bool:
    """ROS's convention: orientation_covariance[0] = -1 means the sensor gives no orientation."""
    covariance = imu.orientation_covariance
    return covariance is not None and float(np.asarray(covariance)[0, 0]) == -1.0


@log.register
def _(imu: Imu, name: str | None = None, *, recording: Any) -> None:
    """Series at ``linear_acceleration|angular_velocity|orientation/<axis>/<name>`` (rates and angles in degrees),
    and the sensor at ``world/<name>``: a small board turned by the orientation, with its x/y/z triad."""
    name = name or "Imu"
    label, styles = part(name), _styles(recording)
    groups = [
        ("linear_acceleration", ("x", "y", "z"), imu.linear_acceleration.as_array(), "m/s²"),
        ("angular_velocity", ("x", "y", "z"), imu.angular_velocity.as_array() * DEG, "deg/s"),
    ]
    q = None if _orientation_unknown(imu) else imu.orientation.normalized()
    if q is not None:
        groups.append(("orientation", ("roll", "pitch", "yaw"), np.array(q.to_euler()) * DEG, "deg"))
    for field_name, columns, values, unit in groups:
        for column, value, colour in zip(columns, values, AXES, strict=True):
            path = f"{field_name}/{column}/{label}"
            legend = _legend(f"{column} ({unit})", name, "Imu")
            styles.log(path, rr.SeriesLines(names=legend, colors=colour), static=True)
            recording.log(path, rr.Scalars(float(value)))
    if q is not None:
        # WHY a level outline: the frame alone shows the turn but not from what; this is the board at rest
        # (gravity-aligned, yaw 0), fixed in the world, so the tilt reads against it.
        hx, hy = _IMU_BOARD[0], _IMU_BOARD[1]
        outline = [[-hx, -hy, 0.0], [hx, -hy, 0.0], [hx, hy, 0.0], [-hx, hy, 0.0], [-hx, -hy, 0.0]]
        styles.log(f"world/{label}_level", rr.LineStrips3D([outline], colors=[GREY], radii=THIN), static=True)
        recording.log(
            f"world/{label}", rr.Transform3D(quaternion=[q.x, q.y, q.z, q.w]), rr.TransformAxes3D(0.08), static=True
        )
        styles.log(
            f"world/{label}/board",
            rr.Boxes3D(half_sizes=[_IMU_BOARD], colors=[(*INK, 230)], fill_mode="solid"),
            static=True,
        )


IMU_EYE = rrb.EyeControls3D(position=(0.12, -0.16, 0.09), look_target=(0.0, 0.0, 0.0), eye_up=(0.0, 0.0, 1.0))


def _imu_views(name: str) -> list[rrb.View]:
    return [
        rrb.Spatial3DView(origin="/world", name="world", eye_controls=IMU_EYE),
        rrb.TimeSeriesView(origin="/linear_acceleration", name="acceleration (m/s²)"),
        rrb.TimeSeriesView(origin="/angular_velocity", name="angular velocity (deg/s)"),
        rrb.TimeSeriesView(origin="/orientation", name="orientation (deg)"),
    ]


_IMU_RATE_HZ = 50.0
_IMU_SECONDS = 6.0
_G = 9.80665


@functools.cache
def _imu_messages() -> tuple[tuple[float, Imu], ...]:
    """A handheld sensor rocking gently: roll ±20°, pitch ±12°, yaw ±35°, with seeded noise and a small bias.

    The gyro reads the body rates of that motion, the accelerometer gravity's reaction in the body frame.
    """
    t = np.round(np.arange(0.0, _IMU_SECONDS + 1e-9, 1.0 / _IMU_RATE_HZ), 6)
    w = 2.0 * np.pi
    amp = np.radians([20.0, 12.0, 35.0])
    freq, phase = np.array([0.2, 0.3, 0.1]), np.array([0.0, 0.8, 0.0])
    angles = amp * np.sin(w * freq * t[:, None] + phase)
    rates = amp * w * freq * np.cos(w * freq * t[:, None] + phase)
    roll, pitch = angles[:, 0], angles[:, 1]
    droll, dpitch, dyaw = rates.T
    body_rate = np.stack([
        droll - dyaw * np.sin(pitch),
        dpitch * np.cos(roll) + dyaw * np.cos(pitch) * np.sin(roll),
        -dpitch * np.sin(roll) + dyaw * np.cos(pitch) * np.cos(roll),
    ], axis=1)  # fmt: skip
    noise = rng(11)
    gyro = body_rate + noise.normal(0.0, 0.0015, body_rate.shape) + np.array([0.001, -0.0008, 0.0005])
    messages = []
    for i, ti in enumerate(t):
        orientation = Quaternion.from_euler(*angles[i])
        specific_force = orientation.to_matrix().T @ np.array([0.0, 0.0, _G])
        accel = specific_force + noise.normal(0.0, 0.03, 3) + np.array([0.02, -0.015, 0.03])
        messages.append((float(ti), Imu(
            orientation=orientation,
            angular_velocity=Vector3(*map(float, gyro[i])),
            linear_acceleration=Vector3(*map(float, accel)),
            orientation_covariance=np.eye(3) * 1e-4,
            angular_velocity_covariance=np.eye(3) * 0.0015**2,
            linear_acceleration_covariance=np.eye(3) * 0.03**2,
            header=Header(stamp=STAMP + float(ti), frame_id="imu_link"),
        )))  # fmt: skip
    return tuple(messages)


register(TypeView(
    type=Imu,
    summary="Acceleration (m/s²), angular velocity (deg/s) and roll/pitch/yaw (deg) over time, x/y/z in "
    "red/green/blue, and the sensor's orientation as a turning frame in 3D.",
    sample=lambda: _imu_messages()[60][1],  # t = 1.2 s
    views=_imu_views,
    series=lambda: list(_imu_messages()),
    code=_STREAM_CODE.format(stream="imu", var="reading", source="imu_readings", type="Imu", rate=_IMU_RATE_HZ),
    tags=("3d", "plot", "stream"),
))  # fmt: skip
