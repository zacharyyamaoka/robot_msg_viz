"""Stock views of robot_msgs.fiducial types (docs/CONTRACT.md: one TypeView per type, registered on import).

What each one shows:

- FiducialDetection: the marker's outline on the image plane in its id's colour, its top-left corner (the printed
  orientation) as a bigger dot, labelled with the dictionary, id, side in pixels and the decoder's margin.
- FiducialPose: the marker as a dark square at ``T_camera_marker`` in front of a small camera glyph (its frame
  ``x right, y down, z forward``), the marker's own axes, the other planar solution as a faint outline, the 1-sigma
  position ellipsoid when it has a covariance; one label with the reprojection errors and their ratio.
- FiducialArray: the detections on the image (`/image/<name>`) beside the poses in 3D (`/world/<name>`).
- FiducialMount / FiducialMap: each marker square on its parent frame (a robot link, a board, the world), the parent's
  axes labelled with its frame name, the covariance ellipsoid, and the mount's provenance (``meta.source``).
- FiducialBoard: the printed board as it is: a white plate, the chessboard's black squares (ChArUco), each marker as
  a dark square labelled with its id, the ChArUco corners as dots; one label with the layout and sizes in mm.
- FiducialTrack / FiducialTrackArray: each tracked marker's frame in the tracker's frame, green when a detection
  updated it this frame and amber when it is predicted only, its 1-sigma position ellipsoid, labelled with the id,
  how long since it was seen, and its NIS.

What is drawn and what is not: a marker is a dark square of its true size, not its bit pattern (the bits need
OpenCV's dictionary tables, which robot_msgs does not carry), so the id is a label. A FiducialTrack has no marker
length, so it draws a frame, never a square of a guessed size.

WHY the covariance is drawn as is under the marker's transform: robot_msgs.fiducial's 6×6 covariance is over the
pose's OWN axes (local tangent, robot_msgs.fiducial.covariance), so its position block already is the ellipsoid in
the marker frame; ROS's parent-axes convention (vision.py's ObjectHypothesisWithPose) needs the extra rotation.

WHY a set eye with y down for camera-frame types: a FiducialPose lives in the camera's optical frame (z forward,
y down), so Rerun's default z-up eye looked at the scene lying on its side; the eye sits behind and above the camera.

None of these types has a stock logger in robot_msgs.viz.rerun (2026-11-26, robot_msgs 0.6.0).
"""

from __future__ import annotations

# WHY empty (2026-11-25): this module registers its TypeViews on import; its functions are sample data and helpers,
# not API. The public API is robot_msg_viz's own __all__ (show/save/stream/blueprint), and robot-sdk's scanner reads
# __all__ to decide what a package publishes.
__all__: list[str] = []

import math
from functools import cache

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from robot_msgs.fiducial import (
    FiducialArray,
    FiducialBoard,
    FiducialDetection,
    FiducialMap,
    FiducialMount,
    FiducialPose,
    FiducialTrack,
    FiducialTrackArray,
    marker_object_points,
)
from robot_msgs.geometry import Pose, Quaternion
from robot_msgs.std import Header

from robot_msg_viz._engine import log, part
from robot_msg_viz.registry import TypeView, register
from robot_msg_viz.views._common import DEG, rng

# ── colours and sizes ────────────────────────────────────────────────────────

# One colour per marker id, the same mid-luminance set vision.py uses for classes (readable on both label pills).
PALETTE = (
    (20, 130, 205),
    (225, 120, 0),
    (0, 155, 110),
    (200, 80, 160),
    (205, 75, 30),
    (120, 95, 205),
    (150, 140, 0),
)
INK = (37, 99, 160)
MARKER = (28, 30, 34, 255)  # the black square as printed
PLATE = (236, 238, 240, 255)  # a board's white plate
FAINT = (150, 155, 165)
OBSERVED = (0, 155, 110)  # a track a detection updated this frame
PREDICTED = (225, 140, 0)  # a track carried by the motion model only
LABEL_DOT = 0.001  # m: a label's anchor point
LIFT = 0.0006  # m: a marker drawn this far in front of the plate it sits on, so the two never z-fight
LINE_2D = -1.75  # UI points (negative radius = screen space): crisp outlines at any zoom
CAMERA_DEPTH = 0.06  # m: depth of the camera glyph's frustum


def colour_of(marker_id: int) -> tuple[int, int, int]:
    return PALETTE[marker_id % len(PALETTE)]


def mm(metres: float) -> str:
    return f"{metres * 1000.0:.4g} mm".replace("-", "−")


def short(dictionary: str) -> str:
    """OpenCV's family name without its ``DICT_`` prefix: ``4X4_50``, ``APRILTAG_36h11``."""
    return dictionary.removeprefix("DICT_")


def xyzw(pose: Pose) -> list[float]:
    q = pose.orientation
    return [q.x, q.y, q.z, q.w]


def frame(recording: rr.RecordingStream, path: str, pose: Pose, axes: float) -> None:
    recording.log(path, rr.Transform3D(translation=pose.position.as_array(), quaternion=xyzw(pose)),
                  rr.TransformAxes3D(axes), static=True)  # fmt: skip


def label(recording: rr.RecordingStream, path: str, position, text: str, colour=FAINT) -> None:
    recording.log(path, rr.Points3D([position], radii=LABEL_DOT, colors=[colour], labels=[text]), static=True)


def square(recording: rr.RecordingStream, path: str, corners: np.ndarray, colour, rgba=MARKER) -> None:
    """A marker's black square through its four corners (top-left first), outlined in its id colour."""
    recording.log(f"{path}/face", rr.Mesh3D(vertex_positions=corners, triangle_indices=[[0, 1, 2], [0, 2, 3]],
                                            albedo_factor=rgba,
                                            face_rendering=rr.components.MeshFaceRendering.DoubleSided),
                  static=True)  # fmt: skip
    recording.log(f"{path}/edge", rr.LineStrips3D([np.vstack([corners, corners[:1]])], radii=-1.5, colors=[colour]),
                  static=True)  # fmt: skip


def ellipsoid(recording: rr.RecordingStream, path: str, covariance, colour) -> None:
    """The 1-sigma position ellipsoid, in the pose's own frame (robot_msgs.fiducial's local-tangent covariance)."""
    if covariance is None:
        return
    block = np.asarray(covariance)[:3, :3]
    variances, axes = np.linalg.eigh((block + block.T) / 2.0)
    if np.linalg.det(axes) < 0:
        axes[:, 0] = -axes[:, 0]
    recording.log(path, rr.Ellipsoids3D(half_sizes=[np.sqrt(np.clip(variances, 0.0, None))],
                                        quaternions=[Quaternion.from_matrix(axes).as_array()], colors=[colour],
                                        line_radii=-1.0), static=True)  # fmt: skip


def sigma_text(covariance) -> str:
    """The largest position and rotation 1-sigma, in mm and degrees (Zach's rule: angles in degrees)."""
    if covariance is None:
        return ""
    diagonal = np.sqrt(np.clip(np.diag(np.asarray(covariance)), 0.0, None))
    return f" · σ ≤ {mm(float(diagonal[:3].max()))}, {float(diagonal[3:].max()) * DEG:.2g}°"


def camera_glyph(recording: rr.RecordingStream, path: str, frame_id: str) -> None:
    """A small frustum at the camera origin, looking down +z (optical frame: x right, y down)."""
    d = CAMERA_DEPTH
    w, h = 0.52 * d, 0.39 * d  # a 4:3 field of view about 55° across
    corners = [[-w, -h, d], [w, -h, d], [w, h, d], [-w, h, d]]
    lines = [[[0.0, 0.0, 0.0], c] for c in corners] + [corners + corners[:1]]
    recording.log(path, rr.TransformAxes3D(0.02), static=True)
    recording.log(f"{path}/frustum", rr.LineStrips3D(lines, radii=-1.0, colors=[FAINT]), static=True)
    label(recording, f"{path}/label", [0.0, -h - 0.01, d], frame_id or "camera")


# ── FiducialDetection (2D) ───────────────────────────────────────────────────


def detection_text(detection: FiducialDetection) -> str:
    text = f"#{detection.id} {short(detection.dictionary)}\nside {float(np.mean(detection.side_lengths_px)):.3g} px"
    if detection.decision_margin is not None:
        text += f" · margin {detection.decision_margin:.3g}"
    if detection.hamming:
        text += f" · {detection.hamming} bit{'s' if detection.hamming > 1 else ''} corrected"
    if detection.corner_covariance is not None:
        sigma = math.sqrt(float(np.max(np.linalg.eigvalsh(np.asarray(detection.corner_covariance)))))
        text += f"\ncorners ±{sigma:.2g} px"
    return text


def draw_detection(recording: rr.RecordingStream, base: str, detection: FiducialDetection) -> None:
    corners = np.asarray(detection.corners, dtype=float)
    colour = colour_of(detection.id)
    recording.log(f"{base}/outline", rr.LineStrips2D([np.vstack([corners, corners[:1]])], radii=LINE_2D,
                                                     colors=[colour]), static=True)  # fmt: skip
    # WHY the top-left corner big: it is the printed orientation, the one thing a square outline cannot show
    recording.log(f"{base}/corners", rr.Points2D(corners, radii=[-5.0, -2.5, -2.5, -2.5], colors=[colour]),
                  static=True)  # fmt: skip
    recording.log(f"{base}/label", rr.Points2D([detection.center], radii=-2.0, colors=[colour],
                                               labels=[detection_text(detection)]), static=True)  # fmt: skip


@log.register
def _(detection: FiducialDetection, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    draw_detection(recording, f"image/{part(name or 'FiducialDetection')}/{detection.id}", detection)


# ── FiducialPose (3D, camera frame) ──────────────────────────────────────────


def pose_text(pose: FiducialPose) -> str:
    text = f"#{pose.id} {short(pose.dictionary)} · {mm(float(np.linalg.norm(pose.pose.position.as_array())))} away"
    if pose.reprojection_error is not None:
        text += f"\nreprojects {pose.reprojection_error:.2g} px"
    if pose.alt_reprojection_error is not None:
        text += f", other solution {pose.alt_reprojection_error:.2g} px (ratio {pose.ambiguity:.2g})"
    return text + sigma_text(pose.covariance).replace(" · ", "\n")


def draw_pose(recording: rr.RecordingStream, base: str, pose: FiducialPose, marker_length: float | None) -> None:
    colour = colour_of(pose.id)
    axes = marker_length or 0.03
    frame(recording, f"{base}/marker", pose.pose, axes)
    if marker_length is not None:
        square(recording, f"{base}/marker/square", np.asarray(marker_object_points(marker_length)), colour)
    if pose.alt_pose is not None and marker_length is not None:
        corners = np.asarray(marker_object_points(marker_length))
        rotation = pose.alt_pose.orientation.to_matrix()
        alt = corners @ rotation.T + pose.alt_pose.position.as_array()
        recording.log(f"{base}/other_solution", rr.LineStrips3D([np.vstack([alt, alt[:1]])], radii=-1.0,
                                                                colors=[FAINT]), static=True)  # fmt: skip
    ellipsoid(recording, f"{base}/marker/covariance", pose.covariance, colour)
    label(recording, f"{base}/marker/label", [0.0, -0.75 * axes, 0.0], pose_text(pose), colour)


@log.register
def _(pose: FiducialPose, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    base = f"world/{part(name or 'FiducialPose')}"
    camera_glyph(recording, f"{base}/camera", "camera")
    # A pose alone has no marker length; the sample's is the one its detector used (SAMPLE_LENGTH).
    draw_pose(recording, f"{base}/{pose.id}", pose, MARKER_LENGTHS.get(id(pose)))


# ── FiducialArray (image + 3D) ───────────────────────────────────────────────


@log.register
def _(array: FiducialArray, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    key = part(name or "FiducialArray")
    for detection in array.detections:
        draw_detection(recording, f"image/{key}/{detection.id}", detection)
    camera_glyph(recording, f"world/{key}/camera", array.header.frame_id)
    lengths = {d.key: d.marker_length for d in array.detections}
    for pose in array.poses:
        draw_pose(recording, f"world/{key}/{pose.id}", pose, lengths.get(pose.key))


# ── FiducialMount / FiducialMap / FiducialBoard ──────────────────────────────


def mount_text(mount: FiducialMount) -> str:
    text = f"#{mount.id} {short(mount.dictionary)} · {mm(mount.marker_length)}"
    source = mount.header.meta.get("source") if mount.header.meta else None
    if source:
        text += f" · {source}"
    return text + sigma_text(mount.covariance).replace(" · ", "\n")


def parent_frame(recording: rr.RecordingStream, path: str, frame_id: str, axes: float) -> None:
    recording.log(path, rr.TransformAxes3D(axes), static=True)
    label(recording, f"{path}/label", [0.0, 0.0, -0.25 * axes], frame_id or "parent")


def draw_mount(recording: rr.RecordingStream, base: str, mount: FiducialMount, *, labelled: bool = True) -> None:
    colour = colour_of(mount.id)
    frame(recording, base, mount.pose, 0.6 * mount.marker_length)
    corners = np.asarray(marker_object_points(mount.marker_length)) + [0.0, 0.0, LIFT]
    square(recording, f"{base}/square", corners, colour)
    ellipsoid(recording, f"{base}/covariance", mount.covariance, colour)
    if labelled:
        label(recording, f"{base}/label", [0.0, 0.75 * mount.marker_length, LIFT], mount_text(mount), colour)


@log.register
def _(mount: FiducialMount, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    base = f"world/{part(name or 'FiducialMount')}"
    reach = max(float(np.linalg.norm(mount.pose.position.as_array())), mount.marker_length)
    parent_frame(recording, base, mount.frame_id, 0.35 * reach)
    draw_mount(recording, f"{base}/{mount.id}", mount)


def frame_groups(mounts) -> dict[str, list[FiducialMount]]:
    groups: dict[str, list[FiducialMount]] = {}
    for mount in mounts:
        groups.setdefault(mount.frame_id, []).append(mount)
    return groups


@log.register
def _(fiducial_map: FiducialMap, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    base = f"world/{part(name or 'FiducialMap')}"
    groups = frame_groups(fiducial_map.mounts)
    offset = 0.0
    for frame_id, mounts in groups.items():
        points = np.vstack([np.asarray(m.corners_in_parent()) for m in mounts] + [np.zeros((1, 3))])
        span = float(np.max(points.max(axis=0) - points.min(axis=0))) or 0.05
        path = f"{base}/{part(frame_id or 'parent')}"
        # WHY side by side: a map does not say how its frames relate (a link and a board), so each frame is drawn
        # at its own spot along x and its label says so, rather than stacked as if their origins coincided.
        recording.log(path, rr.Transform3D(translation=[offset - points[:, 0].min(), 0.0, 0.0]), static=True)
        parent_frame(recording, f"{path}/frame", frame_id + (" (placed beside the others)" if len(groups) > 1 else ""),
                     0.4 * span)  # fmt: skip
        for mount in mounts:
            draw_mount(recording, f"{path}/{mount.id}", mount, labelled=len(mounts) <= 6)
        if len(mounts) > 6:  # a big map: one label per marker would bury the drawing, so the ids ride on the dots
            centres = np.array([m.pose.position.as_array() for m in mounts])
            recording.log(f"{path}/ids", rr.Points3D(centres, radii=LABEL_DOT, colors=[colour_of(m.id) for m in mounts],
                                                     labels=[f"#{m.id}" for m in mounts]), static=True)  # fmt: skip
        offset += 1.4 * span


def board_text(board: FiducialBoard) -> str:
    width, height = board.size
    pitch = (f"squares {mm(board.square_length)}" if board.layout == "charuco"
             else f"gaps {mm(board.separation)}")  # fmt: skip
    return (f"{board.layout} {board.cols} × {board.rows} · {short(board.dictionary)} from #{board.ids[0]} · "
            f"markers {mm(board.marker_length)}, {pitch} · {mm(width)} × {mm(height)}")  # fmt: skip


@log.register
def _(board: FiducialBoard, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    base = f"world/{part(name or 'FiducialBoard')}"
    width, height = board.size
    # The board frame is OpenCV's: origin at the top-left corner, x right, y down, the printed face toward -z.
    parent_frame(recording, f"{base}/frame", board.header.frame_id or "board", 0.35 * max(width, height))
    plate = np.array([[0.0, 0.0, 0.0], [width, 0.0, 0.0], [width, height, 0.0], [0.0, height, 0.0]])
    square(recording, f"{base}/plate", plate, FAINT, PLATE)
    if board.layout == "charuco":
        s = board.square_length
        marked = set(board._cells())
        black = [(c, r) for r in range(board.rows) for c in range(board.cols) if (c, r) not in marked]
        vertices = np.array([[[c * s, r * s, -LIFT], [(c + 1) * s, r * s, -LIFT], [(c + 1) * s, (r + 1) * s, -LIFT],
                              [c * s, (r + 1) * s, -LIFT]] for c, r in black]).reshape(-1, 3)  # fmt: skip
        faces = [[4 * k, 4 * k + 1, 4 * k + 2] for k in range(len(black))]
        faces += [[4 * k, 4 * k + 2, 4 * k + 3] for k in range(len(black))]
        recording.log(f"{base}/squares", rr.Mesh3D(vertex_positions=vertices, triangle_indices=faces,
                                                   albedo_factor=MARKER,
                                                   face_rendering=rr.components.MeshFaceRendering.DoubleSided),
                      static=True)  # fmt: skip
        recording.log(f"{base}/charuco_corners", rr.Points3D(np.asarray(board.chessboard_corners()) - [0, 0, 2 * LIFT],
                                                             radii=-3.0, colors=[(225, 60, 60)]), static=True)  # fmt: skip
    for mount in board.mounts().mounts:
        # mounts face -z in the board frame (OpenCV's IPPE frame turned 180° about x), so LIFT along their own z
        draw_mount(recording, f"{base}/{mount.id}", mount, labelled=False)
    centres = np.array([m.pose.position.as_array() for m in board.mounts().mounts]) - [0.0, 0.0, 3 * LIFT]
    recording.log(f"{base}/ids", rr.Points3D(centres, radii=LABEL_DOT, colors=[(250, 250, 250)],
                                             labels=[f"#{i}" for i in board.ids]), static=True)  # fmt: skip
    label(recording, f"{base}/label", [width / 2.0, height + 0.08 * max(width, height), 0.0], board_text(board), INK)


# ── FiducialTrack / FiducialTrackArray ───────────────────────────────────────


def track_text(track: FiducialTrack, stamp: float | None) -> str:
    text = f"#{track.id} {short(track.dictionary)} · {'observed' if track.observed else 'predicted'}"
    if track.last_seen is not None and stamp is not None and not track.observed:
        text += f", seen {track.age(stamp):.3g} s ago"
    if track.nis is not None:
        text += f" · NIS {track.nis:.3g}"
    return text + sigma_text(track.covariance).replace(" · ", "\n")


def draw_track(recording: rr.RecordingStream, base: str, track: FiducialTrack, stamp: float | None) -> None:
    colour = OBSERVED if track.observed else PREDICTED
    frame(recording, base, track.pose, 0.03)
    # the frame's origin as a dot in the track's state colour, so observed and predicted read at a glance
    recording.log(f"{base}/state", rr.Points3D([[0.0, 0.0, 0.0]], radii=-6.0, colors=[colour]), static=True)
    ellipsoid(recording, f"{base}/covariance", track.covariance, colour)
    label(recording, f"{base}/label", [0.0, 0.0, 0.035], track_text(track, stamp), colour)


@log.register
def _(track: FiducialTrack, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    base = f"world/{part(name or 'FiducialTrack')}"
    parent_frame(recording, f"{base}/frame", "tracker frame", 0.08)
    draw_track(recording, f"{base}/{track.id}", track, None)


@log.register
def _(tracks: FiducialTrackArray, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    base = f"world/{part(name or 'FiducialTrackArray')}"
    parent_frame(recording, f"{base}/frame", tracks.header.frame_id or "tracker frame", 0.08)
    for track in tracks.tracks:
        draw_track(recording, f"{base}/{track.id}", track, tracks.header.stamp)


# ── samples: a 640 × 480 camera 0.4-0.6 m from a wrist board and two loose markers ──

FX, CX, CY = 615.0, 320.0, 240.0  # the catalogue's 640 × 480 colour camera
SAMPLE_LENGTH = 0.04  # m: the loose markers' black square
DICTIONARY = "DICT_4X4_50"
CAMERA = Header(stamp=1_764_028_800.0, frame_id="camera_color_optical_frame")
MARKER_LENGTHS: dict[int, float] = {}  # id(sample FiducialPose) -> its marker length (a pose alone carries none)


def facing(x: float, y: float, z: float, tilt_deg: float, turn_deg: float) -> Pose:
    """A marker at (x, y, z) in the camera's optical frame, facing the camera (its z toward it), upright as printed,
    tipped back by `tilt_deg` about its own x and turned by `turn_deg` about its normal."""
    upright = Pose.from_xyz_rpy(x, y, z, math.pi, 0.0, 0.0).to_matrix()  # 180° about x: marker y up = camera -y
    tip = Pose.from_xyz_rpy(0.0, 0.0, 0.0, math.radians(tilt_deg), 0.0, math.radians(turn_deg)).to_matrix()
    return Pose.from_matrix(upright @ tip)


def project(points: np.ndarray) -> np.ndarray:
    return np.column_stack([FX * points[:, 0] / points[:, 2] + CX, FX * points[:, 1] / points[:, 2] + CY])


def mirrored(pose: Pose, tilt_deg: float) -> Pose:
    """The other planar solution, roughly as IPPE returns it: the marker's tilt about its own x mirrored (a square
    seen at `tilt_deg` projects almost the same at `-tilt_deg`), a few mm further along the line of sight."""
    flip = Pose.from_xyz_rpy(0.0, 0.0, 0.0, math.radians(-2.0 * tilt_deg), 0.0, 0.0).to_matrix()
    matrix = pose.to_matrix() @ flip
    matrix[:3, 3] *= 1.01
    return Pose.from_matrix(matrix)


# (id, x, y, z, tilt deg, turn deg, px error, the other solution's px error): 7 is clear, 12 nearly ambiguous
LOOSE = ((7, -0.09, 0.03, 0.42, 25.0, 10.0, 0.42, 1.6), (12, 0.11, -0.05, 0.55, -15.0, -30.0, 0.38, 0.47))


@cache
def sample_poses() -> tuple[FiducialPose, ...]:
    out = []
    for marker_id, x, y, z, tilt, turn, error, alt_error in LOOSE:
        pose = facing(x, y, z, tilt, turn)
        sigma = 0.0008 * pose.position.z / 0.4  # position σ grows with distance; along the normal it is 3× worse
        covariance = np.diag([sigma**2, sigma**2, (3 * sigma) ** 2, math.radians(1.5) ** 2, math.radians(1.5) ** 2,
                              math.radians(0.4) ** 2])  # fmt: skip
        sample = FiducialPose(id=marker_id, dictionary=DICTIONARY, pose=pose, reprojection_error=error,
                              alt_pose=mirrored(pose, tilt), alt_reprojection_error=alt_error, covariance=covariance)  # fmt: skip
        MARKER_LENGTHS[id(sample)] = SAMPLE_LENGTH
        out.append(sample)
    return tuple(out)


@cache
def sample_detections() -> tuple[FiducialDetection, ...]:
    out = []
    for index, pose in enumerate(sample_poses()):
        corners = np.asarray(pose.corners_in_camera(SAMPLE_LENGTH))
        noise = rng(30 + index).normal(0.0, 0.3, size=(4, 2))
        out.append(FiducialDetection(id=pose.id, dictionary=DICTIONARY, corners=np.round(project(corners) + noise, 2),
                                     marker_length=SAMPLE_LENGTH, decision_margin=(52.0, 31.5)[index],
                                     hamming=(0, 1)[index]).with_corner_sigma(0.3))  # fmt: skip
    return tuple(out)


@cache
def sample_board() -> FiducialBoard:
    return FiducialBoard(layout="charuco", dictionary=DICTIONARY, cols=5, rows=4, marker_length=0.017,
                         square_length=0.023, first_id=20, header=Header(frame_id="wrist_board"))  # fmt: skip


@cache
def sample_mount() -> FiducialMount:
    """Marker 7 on the side of the wrist link, 30 mm out and 50 mm up, from CAD."""
    return FiducialMount(id=7, dictionary=DICTIONARY, marker_length=0.0275,
                         pose=Pose.from_xyz_rpy(0.0, 0.03, 0.05, -math.pi / 2, 0.0, 0.0),
                         covariance=np.diag([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4]),
                         header=Header(frame_id="link_6", meta={"source": "cad"}))  # fmt: skip


@cache
def sample_map() -> FiducialMap:
    """Four markers a bundle adjustment placed on a work table, in the world frame."""
    spots = ((0, 0.30, -0.20, 0.0), (1, 0.30, 0.20, 90.0), (2, 0.70, 0.20, 180.0), (3, 0.70, -0.20, -90.0))
    mounts = []
    for marker_id, x, y, yaw in spots:
        sigma = 0.0005 + 0.0004 * marker_id
        mounts.append(FiducialMount(id=marker_id, dictionary=DICTIONARY, marker_length=0.08,
                                    pose=Pose.from_xyz_rpy(x, y, 0.0, 0.0, 0.0, math.radians(yaw)),
                                    covariance=np.diag([sigma**2, sigma**2, (2 * sigma) ** 2, 1e-5, 1e-5, 1e-5]),
                                    header=Header(frame_id="world", meta={"source": "estimated"})))  # fmt: skip
    return FiducialMap(mounts=tuple(mounts))


@cache
def sample_tracks() -> FiducialTrackArray:
    """Three markers on a moving tray in the base frame: two seen this frame, one occluded for 0.4 s."""
    stamp = 1_764_028_800.0
    tracks = []
    for marker_id, x, y, observed, sigma in ((4, 0.40, -0.12, True, 0.002), (5, 0.52, 0.10, True, 0.0025),
                                             (6, 0.64, -0.04, False, 0.009)):  # fmt: skip
        tracks.append(FiducialTrack(id=marker_id, dictionary=DICTIONARY,
                                    pose=Pose.from_xyz_rpy(x, y, 0.05, 0.0, 0.0, math.radians(15.0 * marker_id)),
                                    covariance=np.diag([sigma**2, (1.6 * sigma) ** 2, sigma**2, 1e-4, 1e-4, 1e-4]),
                                    observed=observed, last_seen=stamp if observed else stamp - 0.4,
                                    nis=(2.1, 4.7, None)[marker_id - 4]))  # fmt: skip
    return FiducialTrackArray(header=Header(stamp=stamp, frame_id="base_link"), tracks=tuple(tracks))


# ── the views ────────────────────────────────────────────────────────────────

# WHY this eye: see the module docstring (camera frame is y down, z forward).
CAMERA_EYE = rrb.EyeControls3D(position=(0.22, -0.2, 0.1), look_target=(0.0, -0.01, 0.46), eye_up=(0.0, -1.0, 0.0))
# WHY from -z, y up the screen as -y: a board's printed face looks toward -z in OpenCV's board frame (origin top-left,
# x right, y down), so the default eye from +z saw the back of the plate. Fitted to the sample; a double click refits.
BOARD_EYE = rrb.EyeControls3D(position=(0.09, 0.15, -0.15), look_target=(0.0575, 0.046, 0.0), eye_up=(0.0, -1.0, 0.0))
TRACK_EYE = rrb.EyeControls3D(position=(0.4, -0.3, 0.24), look_target=(0.53, -0.02, 0.04), eye_up=(0.0, 0.0, 1.0))
MOUNT_EYE = rrb.EyeControls3D(position=(0.13, -0.13, 0.11), look_target=(0.0, 0.02, 0.035), eye_up=(0.0, 0.0, 1.0))
TABLE_EYE = rrb.EyeControls3D(position=(0.05, -0.75, 0.6), look_target=(0.5, 0.0, 0.0), eye_up=(0.0, 0.0, 1.0))


def image(name: str) -> list[rrb.View]:
    return [rrb.Spatial2DView(origin=f"/image/{part(name)}", name=name)]


def camera_world(name: str) -> list[rrb.View]:
    return [rrb.Spatial3DView(origin="/world", name="camera frame", eye_controls=CAMERA_EYE)]


def table(name: str) -> list[rrb.View]:
    return [rrb.Spatial3DView(origin="/world", name="world", eye_controls=TABLE_EYE)]


def mount_view(name: str) -> list[rrb.View]:
    return [rrb.Spatial3DView(origin="/world", name="parent frame", eye_controls=MOUNT_EYE)]


def board_view(name: str) -> list[rrb.View]:
    return [rrb.Spatial3DView(origin="/world", name="board, printed side", eye_controls=BOARD_EYE)]


def tracker(name: str) -> list[rrb.View]:
    return [rrb.Spatial3DView(origin="/world", name="tracker frame", eye_controls=TRACK_EYE)]


register(TypeView(
    type=FiducialDetection,
    summary="The marker's outline on the image in its id's colour, the top-left corner bigger, labelled with id, side in px and decoder margin.",
    sample=lambda: sample_detections()[0],
    views=image,
    tags=("2d",),
))  # fmt: skip

register(TypeView(
    type=FiducialPose,
    summary="The marker as a dark square in front of the camera, its axes, the other planar solution faint, its 1-sigma ellipsoid; reprojection errors in the label.",
    sample=lambda: sample_poses()[0],
    views=camera_world,
    tags=("3d",),
))  # fmt: skip

register(TypeView(
    type=FiducialArray,
    summary="One image's markers: outlines on the image beside their poses in front of the camera, one colour per id.",
    sample=lambda: FiducialArray(header=CAMERA, detections=sample_detections(), poses=sample_poses()),
    views=lambda name: image(name) + camera_world(name),
    tags=("2d", "3d"),
))  # fmt: skip

register(TypeView(
    type=FiducialMount,
    summary="A marker square on its parent frame (labelled), its axes and 1-sigma ellipsoid, with where the mount came from.",
    sample=sample_mount,
    views=mount_view,
    tags=("3d",),
))  # fmt: skip

register(TypeView(
    type=FiducialMap,
    summary="Every mounted marker on its frame: dark squares of their true size, ids and provenance labelled, 1-sigma ellipsoids.",
    sample=sample_map,
    views=table,
    tags=("3d",),
))  # fmt: skip

register(TypeView(
    type=FiducialBoard,
    summary="The board as printed: white plate, chessboard squares, marker squares with their ids, ChArUco corners; sizes in mm.",
    sample=sample_board,
    views=board_view,
    tags=("3d",),
))  # fmt: skip

register(TypeView(
    type=FiducialTrack,
    summary="A tracked marker's frame, green when observed this frame and amber when predicted, with its 1-sigma ellipsoid and NIS.",
    sample=lambda: sample_tracks().tracks[2],
    views=tracker,
    tags=("3d",),
))  # fmt: skip

register(TypeView(
    type=FiducialTrackArray,
    summary="Every tracked marker in the tracker's frame: observed green, predicted amber with time since seen, 1-sigma ellipsoids.",
    sample=sample_tracks,
    views=tracker,
    tags=("3d",),
))  # fmt: skip
