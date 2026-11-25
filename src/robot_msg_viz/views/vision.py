"""Stock views of robot_msgs.vision types (docs/CONTRACT.md: one TypeView per type, registered on import).

What each one shows:

- 2D boxes and detections: the boxes on the image plane (`/image/<name>`), class-coloured through an
  `rr.AnnotationContext`, labelled "<class> <score>" (the tracking id when there is no class). A rotated box is
  drawn rotated.
- 3D boxes and detections: oriented wireframe boxes in the shared scene (`/world/<name>`), the same colours and labels.
- ObjectHypothesis: a score bar on a 0-1 track, coloured on a confidence ramp, labelled with the class.
- ObjectHypothesisWithPose: a labelled frame at the pose, plus its 1-sigma position ellipsoid when it has a covariance.

WHY overrides of robot_msgs' stock vision loggers (robot_msgs.viz.rerun, never edited from here):
- the stock 2D logger draws a rotated box (theta != 0) unrotated, which misplaces it; here it is a rotated outline;
- the stock loggers give every class the same colour; here a class keeps one colour per recording (Zach's rule:
  colour by meaning), and a 3D label sits above its box instead of inside it;
- ObjectHypothesis and ObjectHypothesisWithPose have no stock logger (they fall back to repr text).

The samples are one synthetic scene, a top-down camera over a waste-sorting conveyor (a bottle, a can, a carton and
a cup), drawn in code: the 2D samples sit on its picture, the 3D ones in its point cloud.
"""

from __future__ import annotations

# WHY empty (2026-11-25): this module registers its TypeViews on import; its functions are sample data and helpers,
# not API. The public API is robot_msg_viz's own __all__ (show/save/stream/blueprint), and robot-sdk's scanner reads
# __all__ to decide what a package publishes.
__all__: list[str] = []

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from robot_msgs.geometry import Pose, Pose2D, Quaternion, Vector3
from robot_msgs.std import Header
from robot_msgs.viz.rerun import AXIS_LENGTH
from robot_msgs.vision import (
    BoundingBox2D,
    BoundingBox3D,
    Detection2D,
    Detection2DArray,
    Detection3D,
    Detection3DArray,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)

from robot_msg_viz._engine import log, part
from robot_msg_viz.registry import TypeView, register
from robot_msg_viz.views._common import DEG, rng

# ── colours ──────────────────────────────────────────────────────────────────

# One colour per class. WHY mid-luminance, saturated hues (Okabe-Ito's colour-blind-safe set, darkened): Rerun draws
# a label as text in the instance colour on a grey pill that is light in the light theme and dark in the dark one,
# so a pale colour (sky blue, yellow) was unreadable on the light pill (stills of 2026-11-24).
_PALETTE = (
    (20, 130, 205),  # blue
    (225, 120, 0),  # orange
    (0, 155, 110),  # bluish green
    (200, 80, 160),  # reddish purple
    (205, 75, 30),  # vermillion
    (120, 95, 205),  # violet
    (150, 140, 0),  # olive
    (120, 120, 120),  # grey
)
_NO_CLASS = (140, 140, 140)  # a detection without results
_BOX = (20, 130, 205)  # a bare BoundingBox2D/3D: no class, one calm colour
_TRACK = (70, 74, 82)  # the empty part of a score bar
_RAMP_LOW, _RAMP_HIGH = np.array([110, 118, 132]), np.array([20, 130, 205])  # confidence 0 -> 1: slate -> strong blue
_LABEL_LIFT = 0.04  # m: a 3D label floats this far above its box, clear of the object's own points
_LINE = -1.75  # outline half-width in UI points (negative radius = screen space), so boxes stay crisp at any zoom

# A class keeps its colour within a recording: id(recording) -> {class name: class id (1, 2, ...)}.
# WHY first-appearance order, not a hash: a hash into 8 colours gives two of four classes one colour 59 % of the time.
_CLASS_TABLES: dict[int, dict[str, int]] = {}
# (id(recording), path) of optional parts already drawn, so a stream can clear them when a later message lacks them.
_DRAWN: set[tuple[int, str]] = set()


def _class_colour(class_id: int) -> tuple[int, int, int]:
    return _NO_CLASS if class_id == 0 else _PALETTE[(class_id - 1) % len(_PALETTE)]


def _class_ids(recording: Any, path: str, names: Sequence[str]) -> list[int]:
    """Class ids for `names` (0 for no class), and the AnnotationContext at `path` that colours and names them."""
    table = _CLASS_TABLES.setdefault(id(recording), {})
    ids = []
    for name in names:
        if not name:
            ids.append(0)
            continue
        ids.append(table.setdefault(name, len(table) + 1))
    infos = [rr.AnnotationInfo(id=0, label="no class", color=_NO_CLASS)]
    infos += [rr.AnnotationInfo(id=i, label=n, color=_class_colour(i)) for n, i in table.items()]
    recording.log(path, rr.AnnotationContext(infos), static=True)
    return ids


def _ramp(score: float) -> list[int]:
    share = min(max(score, 0.0), 1.0) if math.isfinite(score) else 1.0  # colour only; the bar length stays the value
    return [int(c) for c in np.round(_RAMP_LOW + (_RAMP_HIGH - _RAMP_LOW) * share)]


def _optional(recording: Any, path: str, archetypes: Sequence[Any] | None, empty: Sequence[Any]) -> None:
    """Log `archetypes` at `path`; when there are none, clear what an earlier message drew there (a stream)."""
    key = (id(recording), path)
    if archetypes is not None:
        _DRAWN.add(key)
        recording.log(path, *archetypes, static=True)
    elif key in _DRAWN:
        recording.log(path, *empty, static=True)


# ── labels ───────────────────────────────────────────────────────────────────


def _hypothesis_text(hypothesis: ObjectHypothesis) -> str:
    return f"{hypothesis.class_id} {hypothesis.score:.2f}"  # robot_msgs' stock label format


def _detection_label(detection: Detection2D | Detection3D) -> str:
    """The best guess ("bottle 0.94"); the tracking id only when there is no guess (robot_msgs' stock rule).

    WHY not both: "can 0.88 · id 13" wrapped onto two lines under a can-sized box and covered the next object.
    """
    best = detection.best
    if best is not None:
        return _hypothesis_text(best.hypothesis)
    return f"id {detection.id}" if detection.id else ""


def _size_label_2d(box: BoundingBox2D) -> str:
    text = f"{box.size_x:.0f} × {box.size_y:.0f} px"
    if box.center.theta:
        text += f" · {box.center.theta * DEG:.0f}°"  # angles on screen in degrees; the message keeps radians
    return text


def _size_label_3d(box: BoundingBox3D) -> str:
    size = box.size
    return f"{size.x * 1000:.0f} × {size.y * 1000:.0f} × {size.z * 1000:.0f} mm"


# ── drawing ──────────────────────────────────────────────────────────────────


def _corners_2d(box: BoundingBox2D) -> list[list[float]]:
    """The box outline as a closed loop of pixel corners, honouring `center.theta`."""
    c, s = math.cos(box.center.theta), math.sin(box.center.theta)
    hx, hy = box.size_x / 2.0, box.size_y / 2.0
    loop = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy), (-hx, -hy)]
    return [[box.center.x + c * u - s * v, box.center.y + s * u + c * v] for u, v in loop]


def _boxes_2d(
    recording: Any,
    path: str,
    boxes: Sequence[BoundingBox2D],
    labels: Sequence[str],
    class_ids: Sequence[int] | None,
) -> None:
    """Axis-aligned boxes as Boxes2D at `path`, rotated ones as outlines at `path/rotated` (Boxes2D cannot rotate)."""
    colour = {} if class_ids is not None else {"colors": [_BOX]}

    def pick(values: Sequence[Any] | None, keep: list[int]) -> dict[str, Any]:
        return {} if values is None else {"class_ids": [values[i] for i in keep]}

    aligned = [i for i, b in enumerate(boxes) if b.center.theta == 0.0]
    rotated = [i for i, b in enumerate(boxes) if b.center.theta != 0.0]
    recording.log(
        path,
        rr.Boxes2D(
            centers=[[boxes[i].center.x, boxes[i].center.y] for i in aligned],
            sizes=[[boxes[i].size_x, boxes[i].size_y] for i in aligned],
            labels=[labels[i] for i in aligned],
            show_labels=True,
            radii=_LINE,
            **colour,
            **pick(class_ids, aligned),
        ),
        static=True,
    )
    drawn = None
    if rotated:
        outlines = [_corners_2d(boxes[i]) for i in rotated]
        # WHY the label on its own point under the outline: a line strip's label sits on the strip's middle, i.e. on
        # top of the object (stills of 2026-11-24); Boxes2D puts its label under the box, so this matches it.
        under = [[float(np.mean([p[0] for p in o])), max(p[1] for p in o) + 12.0] for o in outlines]
        drawn = (
            rr.LineStrips2D(outlines, radii=_LINE, show_labels=False, **colour, **pick(class_ids, rotated)),
            rr.Points2D(under, radii=0.0, labels=[labels[i] for i in rotated], show_labels=True, **colour,
                        **pick(class_ids, rotated)),
        )  # fmt: skip
    _optional(recording, f"{path}/rotated", drawn, (rr.LineStrips2D([]), rr.Points2D([])))


def _boxes_3d(
    recording: Any,
    path: str,
    boxes: Sequence[BoundingBox3D],
    labels: Sequence[str],
    class_ids: Sequence[int] | None,
) -> None:
    colour: dict[str, Any] = {"class_ids": list(class_ids)} if class_ids is not None else {"colors": [_BOX]}
    recording.log(
        path,
        rr.Boxes3D(
            centers=[b.center.position.as_array() for b in boxes],
            half_sizes=[b.size.as_array() / 2.0 for b in boxes],
            quaternions=[b.center.orientation.as_array() for b in boxes],
            radii=_LINE,
            show_labels=False,  # WHY: with class ids it would also print the class name from the context inside it
            **colour,
        ),
        static=True,
    )
    # WHY labels on points just above each box, not on the box: Rerun centres a 3D box's label inside the box, where
    # the object's own points cover it (stills of 2026-11-24).
    above = [[*b.center.position.as_array()[:2], b.corners()[:, 2].max() + _LABEL_LIFT] for b in boxes]
    recording.log(
        f"{path}/labels",
        rr.Points3D(above, radii=0.0, labels=list(labels), show_labels=True, **colour),
        static=True,
    )


# ── loggers (registered on robot_msg_viz's dispatch table, over robot_msgs' stock ones) ──


@log.register
def _(box: BoundingBox2D, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    path = f"image/{part(name or 'BoundingBox2D')}"
    _backdrop(box, path, recording)
    _boxes_2d(recording, path, [box], [_size_label_2d(box)], None)


@log.register
def _(detection: Detection2D, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    path = f"image/{part(name or 'Detection2D')}"
    _backdrop(detection, path, recording)
    ids = _class_ids(recording, path, [detection.class_id])
    _boxes_2d(recording, path, [detection.bbox], [_detection_label(detection)], ids)


@log.register
def _(array: Detection2DArray, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    path = f"image/{part(name or 'Detection2DArray')}"
    _backdrop(array, path, recording)
    ids = _class_ids(recording, path, [d.class_id for d in array.detections])
    labels = [_detection_label(d) for d in array.detections]
    _boxes_2d(recording, path, [d.bbox for d in array.detections], labels, ids)


@log.register
def _(box: BoundingBox3D, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    path = f"world/{part(name or 'BoundingBox3D')}"
    _backdrop(box, path, recording)
    _boxes_3d(recording, path, [box], [_size_label_3d(box)], None)


@log.register
def _(detection: Detection3D, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    path = f"world/{part(name or 'Detection3D')}"
    _backdrop(detection, path, recording)
    ids = _class_ids(recording, path, [detection.class_id])
    _boxes_3d(recording, path, [detection.bbox], [_detection_label(detection)], ids)


@log.register
def _(array: Detection3DArray, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    path = f"world/{part(name or 'Detection3DArray')}"
    _backdrop(array, path, recording)
    ids = _class_ids(recording, path, [d.class_id for d in array.detections])
    labels = [_detection_label(d) for d in array.detections]
    _boxes_3d(recording, path, [d.bbox for d in array.detections], labels, ids)


_BAR_HALF_HEIGHT = 0.04  # score units: the bar is 0.08 of the 0-1 track tall


def _bar(low: float, high: float, colour: Any, *, draw_order: float) -> rr.LineStrips2D:
    """A filled bar from `low` to `high` on the score axis: a thick line whose round caps end exactly at the values.

    WHY inset by the radius: a line strip's caps stick out by its radius, which drew 0.91 as 0.95 (stills of
    2026-11-24); a bar shorter than its height gets a smaller radius, so its length stays the value.
    """
    radius = min(_BAR_HALF_HEIGHT, (high - low) / 2.0)
    if radius <= 0.0:
        return rr.LineStrips2D([])
    return rr.LineStrips2D(
        [[[low + radius, 0.0], [high - radius, 0.0]]], radii=radius, colors=[colour], draw_order=draw_order
    )


@log.register
def _(hypothesis: ObjectHypothesis, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    # WHY a bar on a 0-1 track and not a bar chart: Rerun's bar charts cannot name a bar, and the class is the answer.
    # The bar's length is the score as given (a logit above 1 or below 0 runs past the track, never clipped).
    base = f"scores/{part(name or 'ObjectHypothesis')}"
    score = hypothesis.score
    recording.log(f"{base}/track", _bar(0.0, 1.0, _TRACK, draw_order=1.0), static=True)
    end = score if math.isfinite(score) else 0.0
    recording.log(f"{base}/score", _bar(min(0.0, end), max(0.0, end), _ramp(score), draw_order=2.0), static=True)
    recording.log(
        f"{base}/label",
        rr.Points2D([[0.5, -0.11]], radii=0.0, labels=[_hypothesis_text(hypothesis)], show_labels=True,
                    colors=[_ramp(score)]),
        static=True,
    )  # fmt: skip
    recording.log(
        f"{base}/scale",
        rr.Points2D([[0.0, 0.1], [0.5, 0.1], [1.0, 0.1]], radii=0.0, labels=["0", "0.5", "1"], show_labels=True,
                    colors=[(150, 150, 150)]),
        static=True,
    )  # fmt: skip


@log.register
def _(result: ObjectHypothesisWithPose, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    base = f"world/{part(name or 'ObjectHypothesisWithPose')}"
    _backdrop(result, base, recording)
    pose = result.pose
    recording.log(
        base,
        rr.Transform3D(translation=pose.position.as_array(), quaternion=pose.orientation.as_array()),
        rr.TransformAxes3D(AXIS_LENGTH),
        static=True,
    )
    ids = _class_ids(recording, base, [result.hypothesis.class_id])
    # WHY the label at the tip of the frame's z axis, not at its origin: the origin is inside the object.
    recording.log(
        f"{base}/label",
        rr.Points3D([[0.0, 0.0, AXIS_LENGTH * 1.15]], radii=0.0, labels=[_hypothesis_text(result.hypothesis)],
                    show_labels=True, class_ids=ids),
        static=True,
    )  # fmt: skip
    ellipsoid = None
    if result.covariance is not None:
        half_sizes, quaternion = _position_ellipsoid(np.asarray(result.covariance), pose)
        ellipsoid = (
            rr.Ellipsoids3D(half_sizes=[half_sizes], quaternions=[quaternion], class_ids=ids, line_radii=-1.25,
                            show_labels=False),
        )
    _optional(recording, f"{base}/covariance", ellipsoid, (rr.Ellipsoids3D(half_sizes=[]),))


def _position_ellipsoid(covariance: np.ndarray, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
    """The 1-sigma ellipsoid of the position block, in the pose's own frame (half sizes, xyzw quaternion).

    WHY rotate into the pose frame: ROS orders the covariance x, y, z, roll, pitch, yaw about the PARENT frame's axes,
    and the ellipsoid is logged under the pose's transform, so its axes are turned back by the pose's rotation.
    """
    rotation = pose.orientation.to_matrix()
    local = rotation.T @ covariance[:3, :3] @ rotation
    variances, axes = np.linalg.eigh((local + local.T) / 2.0)
    if np.linalg.det(axes) < 0:
        axes[:, 0] = -axes[:, 0]
    return np.sqrt(np.clip(variances, 0.0, None)), Quaternion.from_matrix(axes).as_array()


# ── the sample scene: a top-down camera over a waste-sorting conveyor ────────

_W, _H = 640, 480  # pixels
_FX = 615.0  # the catalogue's 640 x 480 colour camera (fx = fy, principal point at the centre)
_CX, _CY = _W / 2.0, _H / 2.0
_CAMERA_HEIGHT = 0.75  # m above the belt, looking straight down
_PX_PER_M = _FX / _CAMERA_HEIGHT  # 820 px per metre on the belt surface
_BELT_X = 0.6  # the belt centre, metres in front of the robot base; the belt surface is z = 0
_BELT_HALF_WIDTH = 0.2
_RAIL = 0.035  # the steel side rails beside the belt


@dataclass(frozen=True)
class _Item:
    kind: str  # bottle (lying), can (upright), carton (lying), cup (upright)
    x: float
    y: float
    yaw_deg: float
    size: tuple[float, float, float]  # its box extent along its own x, y, z, metres
    side: tuple[int, int, int]  # the colour of its walls in the point cloud
    results: tuple[tuple[str, float], ...]  # what the detector says it is
    track: str


_ITEMS = (
    _Item("bottle", 0.47, 0.06, 25.0, (0.24, 0.068, 0.068), (130, 180, 215), (("bottle", 0.94), ("can", 0.04)), "12"),
    _Item("can", 0.71, -0.10, 0.0, (0.066, 0.066, 0.122), (180, 38, 44), (("can", 0.88), ("bottle", 0.07)), "13"),
    _Item("carton", 0.79, 0.09, -12.0, (0.19, 0.07, 0.07), (225, 228, 232), (("carton", 0.71), ("cup", 0.18)), "14"),
    _Item("cup", 0.41, -0.11, 0.0, (0.09, 0.09, 0.095), (236, 234, 228), (("cup", 0.46), ("can", 0.31)), "15"),
)  # fmt: skip


def _to_pixel(x: float, y: float) -> tuple[float, float]:
    """Belt coordinates (metres, base frame) -> image pixels: image right is +x, image down is -y."""
    return _CX + (x - _BELT_X) * _PX_PER_M, _CY - y * _PX_PER_M


def _bottle_radius(u: np.ndarray) -> np.ndarray:
    body, neck = 0.034, 0.013
    shoulder = np.clip((u - 0.04) / 0.045, 0.0, 1.0)
    radius = neck + (body - neck) * 0.5 * (1.0 + np.cos(np.pi * shoulder))
    radius = np.where(u > 0.098, 0.0155, radius)  # the cap
    return np.where(u < -0.112, body * np.sqrt(np.clip(1.0 - ((u + 0.112) / 0.008) ** 2, 0.0, 1.0)), radius)


def _item_surface(item: _Item, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """(inside mask, top height, albedo, glossy) of one item seen from above."""
    yaw = math.radians(item.yaw_deg)
    u = (x - item.x) * math.cos(yaw) + (y - item.y) * math.sin(yaw)
    v = -(x - item.x) * math.sin(yaw) + (y - item.y) * math.cos(yaw)
    albedo = np.empty((*x.shape, 3))
    if item.kind == "bottle":
        radius = _bottle_radius(u)
        inside = (np.abs(u) <= 0.12) & (np.abs(v) < radius)
        height = 0.034 + np.sqrt(np.clip(radius**2 - v**2, 0.0, None))
        albedo[:] = (150, 196, 226)
        band = (u > -0.075) & (u < 0.02)
        albedo[band] = (58, 142, 84)
        albedo[band & (np.abs(u + 0.028) < 0.006)] = (235, 235, 235)
        albedo[u > 0.098] = (34, 84, 196)
        return inside, height, albedo, True
    if item.kind == "can":
        dist = np.hypot(u, v)
        inside = dist <= 0.033
        height = np.where(dist > 0.029, 0.1245, 0.122)
        albedo[:] = (168, 171, 178)
        albedo[dist > 0.029] = (198, 201, 207)
        tab = (np.abs(u - 0.011) < 0.007) & (np.abs(v) < 0.0045)
        albedo[tab] = (214, 216, 220)
        height = np.where(tab, 0.1228, height)
        return inside, height, albedo, True
    if item.kind == "carton":
        inside = (np.abs(u) <= 0.095) & (np.abs(v) <= 0.035)
        height = np.where(u > 0.07, 0.07 - (u - 0.07) * 0.8, 0.07)
        albedo[:] = (226, 228, 232)
        albedo[(u > -0.045) & (u < 0.03)] = (46, 96, 176)
        albedo[(u > -0.03) & (u < 0.015) & (np.abs(v) < 0.012)] = (238, 238, 238)
        return inside, height, albedo, False
    dist = np.hypot(u, v)  # the cup: a white rim around its stained floor
    inside = dist <= 0.045
    height = np.where(dist > 0.040, 0.095, 0.012)
    albedo[:] = (240, 238, 232)
    albedo[dist <= 0.040] = (112, 96, 78)
    albedo[(dist <= 0.040) & (np.hypot(u - 0.008, v + 0.006) < 0.018)] = (84, 60, 40)
    return inside, height, albedo, False


@dataclass(frozen=True)
class _Scene:
    picture: np.ndarray  # (H, W, 3) uint8, what the camera sees
    points: np.ndarray  # (N, 3) metres, base frame: the depth camera's cloud of the belt and what is on it
    colors: np.ndarray  # (N, 3) uint8
    owners: np.ndarray  # (N,) which item a point lies on (its index in _ITEMS), -1 for the belt and rails
    footprints: tuple[tuple[float, float, float, float], ...]  # per item: its pixel extent x0, y0, x1, y1


@cache
def _scene() -> _Scene:
    u, v = np.meshgrid(np.arange(_W) + 0.5, np.arange(_H) + 0.5)
    x, y = _BELT_X + (u - _CX) / _PX_PER_M, -(v - _CY) / _PX_PER_M
    noise = rng(7)
    # the belt (rubber with faint cleats), the steel rails, and the floor far below
    height = np.where(np.abs(y) <= _BELT_HALF_WIDTH, 0.0, np.where(np.abs(y) <= _BELT_HALF_WIDTH + _RAIL, 0.03, -0.6))
    albedo = np.empty((_H, _W, 3))
    albedo[:] = (56, 58, 62)
    albedo *= (1.0 + 0.06 * np.sin(2 * np.pi * x / 0.028))[..., None]
    albedo += noise.normal(0.0, 2.5, (_H, _W, 1))
    rail = (np.abs(y) > _BELT_HALF_WIDTH) & (np.abs(y) <= _BELT_HALF_WIDTH + _RAIL)
    albedo[rail] = (146, 150, 156)
    albedo[height < -0.1] = (30, 32, 35)
    owner = np.full((_H, _W), -1)
    glossy = np.zeros((_H, _W), dtype=bool)
    footprints = []
    for index, item in enumerate(_ITEMS):
        inside, top, colour, shiny = _item_surface(item, x, y)
        on_top = inside & (top > height)
        height = np.where(on_top, top, height)
        albedo[on_top] = colour[on_top]
        owner[on_top] = index
        glossy |= on_top & shiny
        rows, cols = np.nonzero(inside)
        footprints.append((float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1)))
    # a soft drop shadow to the lower right (the light is upper left)
    shadow = np.roll(owner >= 0, (7, 7), axis=(0, 1)) & (owner < 0) & (height > -0.1)
    albedo[shadow] *= 0.6
    # shading from the height field's normals (Lambert + a highlight on glossy plastic and metal)
    slope_x = np.clip(np.gradient(height, axis=1) * _PX_PER_M, -3.0, 3.0)
    slope_y = np.clip(-np.gradient(height, axis=0) * _PX_PER_M, -3.0, 3.0)
    normal = np.dstack([-slope_x, -slope_y, np.ones_like(height)])
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    light = np.array([-0.5, 0.6, 1.0]) / np.linalg.norm([-0.5, 0.6, 1.0])
    half = (light + [0.0, 0.0, 1.0]) / np.linalg.norm(light + [0.0, 0.0, 1.0])
    diffuse = np.clip(normal @ light, 0.0, 1.0)
    shine = np.where(glossy, np.clip(normal @ half, 0.0, 1.0) ** 60 * 0.55, 0.0)
    shaded = albedo * (0.32 + 0.75 * diffuse)[..., None] + 255.0 * shine[..., None]
    clean = np.clip(shaded, 0, 255).astype(np.uint8)
    picture = np.clip(shaded + noise.normal(0.0, 1.5, shaded.shape), 0, 255).astype(np.uint8)
    # the cloud: every 5th pixel's surface point (top-down depth camera, range-gated to the conveyor) plus item walls
    step = 5
    grid = (slice(step // 2, None, step), slice(step // 2, None, step))
    gx, gy, gh, gowner, gcol = x[grid], y[grid], height[grid], owner[grid], clean[grid]
    keep = gh > -0.1
    points = [np.column_stack([gx[keep], gy[keep], gh[keep]])]
    colors = [gcol[keep]]
    owners = [gowner[keep]]
    for index, item in enumerate(_ITEMS):
        mine = gowner == index
        padded = np.pad(mine, 1)
        edge = mine & ~(padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
        for ex, ey, eh in zip(gx[edge], gy[edge], gh[edge], strict=True):
            levels = np.arange(0.004, eh, 0.006)
            points.append(np.column_stack([np.full_like(levels, ex), np.full_like(levels, ey), levels]))
            colors.append(np.tile(np.array(item.side, dtype=np.uint8), (len(levels), 1)))
            owners.append(np.full(len(levels), index))
    cloud = np.concatenate(points)
    cloud[:, 2] += noise.normal(0.0, 0.0006, len(cloud))  # depth noise, about 0.6 mm
    return _Scene(
        picture, cloud.astype(np.float32), np.concatenate(colors).astype(np.uint8), np.concatenate(owners),
        tuple(footprints),
    )  # fmt: skip


# The samples are drawn over the scene they were "detected" in; a user's own message never is.
# WHY: a vision_msgs detection carries no pixels (ROS 2 dropped source_img), and a box with no picture reads poorly,
# so only the sample objects built in this module carry a backdrop, matched by identity (equal user data is not them).
_BACKDROPS: dict[int, tuple[Any, Callable[[str, Any], None]]] = {}


def _backdrop(msg: Any, path: str, recording: Any) -> None:
    entry = _BACKDROPS.get(id(msg))
    if entry is not None and entry[0] is msg:
        entry[1](path, recording)


def _picture(path: str, recording: Any) -> None:
    recording.log(f"{path}/picture", rr.Image(_scene().picture, color_model="RGB"), static=True)


def _cloud(path: str, recording: Any, ghost: int | None = None) -> None:
    """The scene's point cloud at /world/scene; `ghost` (an item index) is drawn faint so what sits inside it shows."""
    scene = _scene()
    # WHY thin points, not transparent ones: a point colour with alpha drew darker (black), not fainter, in 0.40
    radii = np.where(scene.owners == ghost, 0.0006, 0.0025) if ghost is not None else 0.0025
    recording.log("world/scene", rr.Points3D(scene.points, colors=scene.colors, radii=radii), static=True)


def _cloud_ghosting(index: int) -> Callable[[str, Any], None]:
    return lambda path, recording: _cloud(path, recording, ghost=index)


def _with(backdrop: Callable[[str, Any], None], msg: Any) -> Any:
    _BACKDROPS[id(msg)] = (msg, backdrop)
    return msg


_CAMERA_HEADER = Header(stamp=1_764_028_800.0, frame_id="camera_color_optical_frame")
_BASE_HEADER = Header(stamp=1_764_028_800.0, frame_id="base_link")


def _box_2d(item: _Item, index: int) -> BoundingBox2D:
    """A detector's axis-aligned box: the item's pixel extent, a few pixels off (seeded)."""
    x0, y0, x1, y1 = _scene().footprints[index]
    jitter = rng(11 + index).integers(-3, 4, size=4)
    return BoundingBox2D.from_xyxy(x0 + jitter[0], y0 + jitter[1], x1 + jitter[2], y1 + jitter[3])


def _box_3d(item: _Item) -> BoundingBox3D:
    centre = Pose.from_xyz_rpy(item.x, item.y, item.size[2] / 2.0, 0.0, 0.0, math.radians(item.yaw_deg))
    return BoundingBox3D(center=centre, size=Vector3(*item.size))


def _results(item: _Item, pose: Pose | None = None) -> tuple[ObjectHypothesisWithPose, ...]:
    return tuple(
        ObjectHypothesisWithPose(hypothesis=ObjectHypothesis(c, s), pose=pose or Pose()) for c, s in item.results
    )


def _detection_2d(index: int) -> Detection2D:
    item = _ITEMS[index]
    return Detection2D(header=_CAMERA_HEADER, results=_results(item), bbox=_box_2d(item, index), id=item.track)


def _detection_3d(index: int) -> Detection3D:
    item = _ITEMS[index]
    box = _box_3d(item)
    return Detection3D(header=_BASE_HEADER, results=_results(item, box.center), bbox=box, id=item.track)


def _rotated_bottle_box() -> BoundingBox2D:
    bottle = _ITEMS[0]
    cx, cy = _to_pixel(bottle.x, bottle.y)
    # image y points down, so a turn counter-clockwise on the belt is clockwise (negative theta) in the image
    return BoundingBox2D(
        Pose2D(round(cx, 1), round(cy, 1), -math.radians(bottle.yaw_deg)),
        round(bottle.size[0] * _PX_PER_M, 1),
        round(bottle.size[1] * _PX_PER_M, 1),
    )


def _bottle_estimate() -> ObjectHypothesisWithPose:
    bottle = _ITEMS[0]
    yaw = math.radians(bottle.yaw_deg)
    pose = Pose.from_xyz_rpy(bottle.x, bottle.y, bottle.size[2] / 2.0, 0.0, 0.0, yaw)
    # position spread: 40 mm along the bottle, 15 mm across, 8 mm up (one view of a long object), in the base frame
    along = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    across = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    position = 0.040**2 * np.outer(along, along) + 0.015**2 * np.outer(across, across) + 0.008**2 * np.outer(up, up)
    covariance = np.zeros((6, 6))
    covariance[:3, :3] = position
    covariance[3:, 3:] = np.diag(np.radians([3.0, 3.0, 8.0]) ** 2)
    return ObjectHypothesisWithPose(hypothesis=ObjectHypothesis("bottle", 0.94), pose=pose, covariance=covariance)


_SAMPLES: dict[type, Any] = {}


def _sample(kind: type) -> Callable[[], Any]:
    """The sample of `kind`, built once (the scene draws on first use) so its backdrop matches it by identity."""

    def build() -> Any:
        if kind not in _SAMPLES:
            _SAMPLES[kind] = _BUILDERS[kind]()
        return _SAMPLES[kind]

    return build


_BUILDERS: dict[type, Callable[[], Any]] = {
    BoundingBox2D: lambda: _with(_picture, _rotated_bottle_box()),
    Detection2D: lambda: _with(_picture, _detection_2d(0)),
    Detection2DArray: lambda: _with(
        _picture,
        Detection2DArray(header=_CAMERA_HEADER, detections=tuple(_detection_2d(i) for i in range(len(_ITEMS)))),
    ),
    BoundingBox3D: lambda: _with(_cloud, _box_3d(_ITEMS[0])),
    Detection3D: lambda: _with(_cloud, _detection_3d(0)),
    Detection3DArray: lambda: _with(
        _cloud, Detection3DArray(header=_BASE_HEADER, detections=tuple(_detection_3d(i) for i in range(len(_ITEMS))))
    ),
    ObjectHypothesis: lambda: ObjectHypothesis("bottle", 0.91),
    # WHY the bottle ghosted: the estimate's frame and ellipsoid sit inside the bottle, where its points hid them.
    ObjectHypothesisWithPose: lambda: _with(_cloud_ghosting(0), _bottle_estimate()),
}  # fmt: skip


# ── the views ────────────────────────────────────────────────────────────────


def _image(name: str) -> list[rrb.View]:
    return [rrb.Spatial2DView(origin=f"/image/{part(name)}", name=name)]


# WHY a set eye: Rerun's automatic eye sits on the scene's bounding-box diagonal, which for a flat conveyor is nearly
# level (stills of 2026-11-24). This one looks down at about 40 degrees on a tabletop 0.6 m in front of the robot base,
# where perception data in base_link usually is; a double click in the view refits it to whatever was logged.
_EYE = rrb.EyeControls3D(position=(0.18, -0.46, 0.48), look_target=(0.6, -0.04, 0.02), eye_up=(0.0, 0.0, 1.0))


def _world(name: str) -> list[rrb.View]:
    return [rrb.Spatial3DView(origin="/world", name="world", eye_controls=_EYE)]


def _scores(name: str) -> list[rrb.View]:
    return [rrb.Spatial2DView(origin=f"/scores/{part(name)}", name=name)]


_OVERLAY = (
    "import robot_msg_viz as viz\n"
    "from robot_msgs.vision import {type}\n\n"
    "viz.show({var})  # the boxes, class-coloured and labelled\n\n"
    "# over the image they came from: log both under one name\n"
    "with viz.stream(\"camera\") as live:\n"
    "    live.log(camera=image)  # a robot_msgs.sensor.Image\n"
    "    live.log(camera={var})"
)


register(TypeView(
    type=BoundingBox2D,
    summary="A rectangle on the image, drawn rotated when it is, labelled with its size in pixels (and its angle).",
    sample=_sample(BoundingBox2D),
    views=_image,
    tags=("2d",),
))  # fmt: skip

register(TypeView(
    type=Detection2D,
    summary="One detected object: its box on the image in its class colour, labelled with its best class and score.",
    sample=_sample(Detection2D),
    views=_image,
    code=_OVERLAY.format(type="Detection2D", var="detection"),
    tags=("2d",),
))  # fmt: skip

register(TypeView(
    type=Detection2DArray,
    summary="Every detection in one image: a box per object, one colour per class, each labelled with class and score.",
    sample=_sample(Detection2DArray),
    views=_image,
    code=_OVERLAY.format(type="Detection2DArray", var="detections"),
    tags=("2d",),
))  # fmt: skip

register(TypeView(
    type=BoundingBox3D,
    summary="An oriented box in the 3D scene, labelled with its size in millimetres.",
    sample=_sample(BoundingBox3D),
    views=_world,
    tags=("3d",),
))  # fmt: skip

register(TypeView(
    type=Detection3D,
    summary="One detected object in 3D: its oriented box in its class colour, labelled with its best class and score.",
    sample=_sample(Detection3D),
    views=_world,
    tags=("3d",),
))  # fmt: skip

register(TypeView(
    type=Detection3DArray,
    summary="Every detection in one frame: an oriented box per object, one colour per class, each labelled with class "
    "and score.",
    sample=_sample(Detection3DArray),
    views=_world,
    tags=("3d",),
))  # fmt: skip

register(TypeView(
    type=ObjectHypothesis,
    summary="A class guess: a bar filled to its score on a 0-1 track, brighter when more confident, named by class.",
    sample=_sample(ObjectHypothesis),
    views=_scores,
    tags=("2d",),
))  # fmt: skip

register(TypeView(
    type=ObjectHypothesisWithPose,
    summary="A class guess with a pose: a frame at the object labelled with class and score, inside its 1σ position "
    "ellipsoid.",
    sample=_sample(ObjectHypothesisWithPose),
    views=_world,
    tags=("3d",),
))  # fmt: skip
