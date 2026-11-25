"""Stock views of robot_msgs.shape types (docs/CONTRACT.md: one TypeView per type, registered on import).

- SolidPrimitive: the shaded solid in its own frame (origin triad poking out), its dimensions in one label.
- Mesh: the shaded mesh, its triangle and vertex counts (and its volume when it is closed) in one label.
- Plane: a finite translucent patch around the plane's point nearest the origin, its normal arrow, the
  offset from the world origin, and the equation as the patch's label (the plane itself is infinite).

Not supported, on purpose:

- MeshResource: it carries only a file name (``package://…``) and a scale, not the triangles. Drawing it
  truthfully needs that file, and a gallery sample may not read files outside this package (none ships
  in it), so it keeps the repr-as-text fallback rather than a placeholder that looks like the real mesh.

WHY an override for Mesh and SolidPrimitive (2026-11-24): robot_msgs' stock logger sends a bare Mesh3D
with no normals, which Rerun 0.40 draws as a flat, unlit silhouette (no edges, no depth). Here every
triangle corner gets a normal smoothed only across faces within ``CREASE_DEG`` of it, so a sphere
reads round and a box keeps its edges. Plane has no stock logger.
"""

from __future__ import annotations

# WHY empty (2026-11-25): this module registers its TypeViews on import; its functions are sample data and helpers,
# not API. The public API is robot_msg_viz's own __all__ (show/save/stream/blueprint), and robot-sdk's scanner reads
# __all__ to decide what a package publishes.
__all__: list[str] = []

import math

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from robot_msgs.geometry import Point, Vector3
from robot_msgs.shape import Mesh, Plane, SolidPrimitive

from robot_msg_viz._engine import log, part
from robot_msg_viz.registry import TypeView, register
from robot_msg_viz.views._common import origin_triad

CREASE_DEG = 40.0  # faces meeting at a sharper angle than this keep a hard edge
SOLID = (156, 178, 210, 255)  # a light steel: reads under Rerun's default light on the dark canvas
MESH = (205, 214, 226, 255)
PATCH = (90, 150, 230, 70)  # translucent: what is behind the plane stays visible
PATCH_EDGE = (120, 170, 240)
NORMAL = (240, 140, 0)  # the accent: the one arrow that says which way the plane faces
FAINT = (150, 150, 160)
GRID = (165, 170, 180)
LABEL_DOT = 0.002  # m: a label's anchor point, small enough to read as text only


def world(name: str) -> list[rrb.View]:
    """The shared 3D scene (robot_msgs logs spatial data under /world)."""
    return [rrb.Spatial3DView(origin="/world", name="world")]


def mm(metres: float) -> str:
    """A length for a label, in millimetres, to four significant figures (never silently rounded to 0)."""
    return f"{metres * 1000.0:.4g} mm".replace("-", "−")


# ── context ──────────────────────────────────────────────────────────────────


def nice(step: float) -> float:
    """The round length (1, 2 or 5 × 10^k) at or above `step`."""
    power = 10.0 ** math.floor(math.log10(step))
    return next(m * power for m in (1.0, 2.0, 5.0, 10.0) if m * power >= step * (1 - 1e-9))


def scale_grid(recording: rr.RecordingStream, path: str, low: np.ndarray, high: np.ndarray) -> None:
    """A faint square grid under the object, its square size in its label, so the object's size reads.

    WHY (2026-11-24, looked at in the real web viewer): Rerun 0.40's opening camera stands 1.5 bounding-box
    half-diagonals from the centre, so a lone bottle or cylinder filled the view and was cut off. A grid
    about twice the object's size under it gives the camera room and the eye a ruler; it scales
    with the message, so a screw and a bin both open framed.
    """
    span = float(np.max(high - low)) or 0.1
    cell = nice(span / 4.0)
    half = cell * math.ceil(0.75 * span / cell)
    cx, cy, z = (low[0] + high[0]) / 2.0, (low[1] + high[1]) / 2.0, float(low[2])
    ticks = np.arange(-half, half + cell / 2, cell)
    lines = [[[cx + t, cy - half, z], [cx + t, cy + half, z]] for t in ticks]
    lines += [[[cx - half, cy + t, z], [cx + half, cy + t, z]] for t in ticks]
    recording.log(path, rr.LineStrips3D(lines, radii=rr.Radius.ui_points(0.75), colors=[GRID]), static=True)
    corner = [cx + half, cy - half, z]
    recording.log(f"{path}/label", rr.Points3D([corner], radii=LABEL_DOT, colors=[GRID],
                                               labels=[f"squares {mm(cell)}"]), static=True)  # fmt: skip


# ── shading ──────────────────────────────────────────────────────────────────


def shaded(vertices: np.ndarray, triangles: np.ndarray, crease_deg: float = CREASE_DEG) -> dict[str, np.ndarray]:
    """Mesh3D fields with one vertex per triangle corner and crease-aware normals.

    Each corner's normal is the area-weighted sum of the face normals around its vertex whose
    direction is within `crease_deg` of its own face's, so curved surfaces shade smoothly and
    corners stay sharp. The geometry is unchanged: positions are the original vertices, repeated.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if len(triangles) == 0:
        return {"vertex_positions": vertices.astype(np.float32)}
    corners = vertices[triangles]  # (M, 3, 3)
    face = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])  # length = 2 * area
    unit = face / np.maximum(np.linalg.norm(face, axis=1, keepdims=True), 1e-18)
    faces_at: list[list[int]] = [[] for _ in range(len(vertices))]
    for f, tri in enumerate(triangles):
        for v in tri:
            faces_at[v].append(f)
    cos_crease = math.cos(math.radians(crease_deg))
    normals = np.zeros((len(triangles), 3, 3))
    for f, tri in enumerate(triangles):
        for k, v in enumerate(tri):
            around = np.asarray(faces_at[v])
            near = around[unit[around] @ unit[f] >= cos_crease]
            normals[f, k] = face[near].sum(axis=0)
    normals /= np.maximum(np.linalg.norm(normals, axis=2, keepdims=True), 1e-18)
    return {
        "vertex_positions": corners.reshape(-1, 3).astype(np.float32),
        "vertex_normals": normals.reshape(-1, 3).astype(np.float32),
        "triangle_indices": np.arange(3 * len(triangles), dtype=np.uint32).reshape(-1, 3),
    }


def is_closed(triangles: np.ndarray) -> bool:
    """Every edge shared by exactly two triangles, once in each direction (a watertight, consistent mesh)."""
    triangles = np.asarray(triangles, dtype=np.int64)
    if len(triangles) == 0:
        return False
    directed = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    forward = {tuple(e) for e in directed.tolist()}
    if len(forward) != len(directed):
        return False
    return all((b, a) in forward for a, b in forward)


# ── SolidPrimitive ───────────────────────────────────────────────────────────


def describe(solid: SolidPrimitive) -> str:
    """The primitive's kind and dimensions in plain words, lengths in mm."""
    d = solid.dimensions
    if solid.type == SolidPrimitive.BOX:
        return f"box {mm(d[0])} × {mm(d[1])} × {mm(d[2])} (x × y × z)"
    if solid.type == SolidPrimitive.SPHERE:
        return f"sphere, radius {mm(d[0])}"
    if solid.type == SolidPrimitive.CYLINDER:
        return f"cylinder along z, height {mm(d[0])}, radius {mm(d[1])}"
    if solid.type == SolidPrimitive.CONE:
        return f"cone along z, height {mm(d[0])}, base radius {mm(d[1])}, apex up"
    return f"prism along z, height {mm(d[0])}, {len(solid.polygon)}-corner outline"


@log.register
def _(solid: SolidPrimitive, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    base = f"world/{part(name or 'SolidPrimitive')}"
    mesh = solid.to_mesh(48)
    low, high = solid.bounding_box()
    low, high = low.as_array(), high.as_array()
    reach = float(np.max(np.abs(np.concatenate([low, high]))))
    # WHY the triad pokes out of the solid: ROS centres every primitive's bounding box on its origin
    # (a cone's origin is mid-height, not its base); an axis inside an opaque solid would be hidden.
    recording.log(base, rr.TransformAxes3D(1.35 * reach), static=True)
    recording.log(f"{base}/solid", rr.Mesh3D(**shaded(mesh.vertices, mesh.triangles), albedo_factor=SOLID), static=True)
    top = [0.0, 0.0, float(high[2]) + 0.25 * reach]
    recording.log(f"{base}/label", rr.Points3D([top], radii=LABEL_DOT, colors=[FAINT], labels=[describe(solid)]), static=True)
    scale_grid(recording, f"{base}/scale", low, high)


register(TypeView(
    type=SolidPrimitive,
    summary="The shaded solid centred on its own frame (red, green, blue axes), its kind and size in mm as a label.",
    sample=lambda: SolidPrimitive.cylinder(0.24, 0.07),
    views=world,
    tags=("3d",),
))  # fmt: skip


# ── Mesh ─────────────────────────────────────────────────────────────────────


def bottle(segments: int = 40) -> Mesh:
    """A 1.5 L drinks bottle (the thing BAM's cell sorts), revolved from its profile: a closed, outward mesh.

    The profile runs bottom to top as (radius, height) in metres; each interior row becomes a ring of
    `segments` vertices, and the two ends close on a centre vertex.
    """
    shoulder = [(0.044 - 0.027 * (1 - math.cos(math.pi * s / 7)) / 2, 0.205 + 0.07 * s / 7) for s in range(1, 8)]
    profile = [
        (0.036, 0.0), (0.042, 0.004), (0.044, 0.014),            # foot
        (0.044, 0.085), (0.040, 0.095), (0.044, 0.105),          # the grip groove
        (0.044, 0.205), *shoulder,                               # body, then the shoulder up to the neck
        (0.0145, 0.285), (0.0165, 0.287), (0.0165, 0.290),       # neck and the tamper ring
        (0.0150, 0.291), (0.0150, 0.295), (0.0160, 0.296),       # the cap
        (0.0160, 0.315), (0.0145, 0.317),
    ]  # fmt: skip
    angle = 2.0 * math.pi * np.arange(segments) / segments
    rings = [np.column_stack([r * np.cos(angle), r * np.sin(angle), np.full(segments, z)]) for r, z in profile]
    bottom, top = len(profile) * segments, len(profile) * segments + 1
    vertices = np.vstack([*rings, [[0.0, 0.0, 0.0], [0.0, 0.0, profile[-1][1]]]])
    triangles = []
    for k in range(len(profile) - 1):  # quads between ring k (lower) and ring k + 1 (upper), outward
        lower, upper = k * segments, (k + 1) * segments
        for i in range(segments):
            j = (i + 1) % segments
            triangles += [[lower + i, lower + j, upper + j], [lower + i, upper + j, upper + i]]
    last = (len(profile) - 1) * segments
    for i in range(segments):
        j = (i + 1) % segments
        triangles += [[bottom, j, i], [top, last + i, last + j]]  # base faces -z, cap top faces +z
    return Mesh(vertices=vertices, triangles=triangles)


@log.register
def _(mesh: Mesh, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    base = f"world/{part(name or 'Mesh')}"
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    closed = is_closed(triangles)
    fields = shaded(vertices, triangles)
    # WHY double-sided for an open mesh: its inside is visible through the hole and would otherwise vanish.
    rendering = None if closed else rr.components.MeshFaceRendering.DoubleSided
    recording.log(f"{base}/mesh", rr.Mesh3D(**fields, albedo_factor=MESH, face_rendering=rendering), static=True)
    if len(vertices) == 0:
        return
    text = f"{len(triangles):,} triangles · {len(vertices):,} vertices".replace(",", " ")
    if closed and mesh.volume() > 0.0:
        text += f" · closed, {mesh.volume() * 1e6:.4g} cm³"
    elif not closed:
        text += " · open surface"
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    top = [(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, high[2] + 0.12 * float(np.max(high - low))]
    recording.log(f"{base}/label", rr.Points3D([top], radii=LABEL_DOT, colors=[FAINT], labels=[text]), static=True)
    scale_grid(recording, f"{base}/scale", low, high)


register(TypeView(
    type=Mesh,
    summary="The shaded triangle mesh, with its triangle and vertex counts (and volume, when it is closed) as a label.",
    sample=bottle,
    views=world,
    tags=("3d",),
))  # fmt: skip


# ── Plane ────────────────────────────────────────────────────────────────────


def equation(plane: Plane) -> str:
    """``a x + b y + c z + d = 0`` as written in the message (zero terms dropped, 3 significant figures)."""
    terms = []
    for value, axis in zip(plane.coef[:3], "xyz", strict=True):
        if value == 0.0:
            continue
        sign = "−" if value < 0 else "+"
        size = abs(float(value))
        body = axis if math.isclose(size, 1.0) else f"{size:.3g} {axis}"
        terms.append((sign, body))
    text = ("−" if terms[0][0] == "−" else "") + terms[0][1]
    text += "".join(f" {sign} {body}" for sign, body in terms[1:])
    if plane.d != 0.0:
        text += f" {'−' if plane.d < 0 else '+'} {abs(plane.d):.3g}"
    return f"{text} = 0"


def in_plane_axes(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane, the first as close to world x as the plane allows."""
    seed = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = seed - (seed @ normal) * normal
    first /= np.linalg.norm(first)
    return first, np.cross(normal, first)


@log.register
def _(plane: Plane, name: str | None = None, *, recording: rr.RecordingStream) -> None:
    base = f"world/{part(name or 'Plane')}"
    normal = plane.normal.as_array()
    offset = plane.d / float(np.linalg.norm(plane.coef[:3]))  # signed distance of the world origin
    nearest = -offset * normal  # the plane's point closest to the world origin
    distance = abs(offset)
    # WHY this size: the plane is infinite; a patch about as big as its distance from the origin reads as
    # "a plane over there" without hiding the scene, and never shrinks below 0.4 m.
    half = max(0.2, 0.6 * distance)
    u, v = in_plane_axes(normal)
    corners = [nearest + half * (su * u + sv * v) for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    recording.log(f"{base}/patch", rr.Mesh3D(
        vertex_positions=corners, triangle_indices=[[0, 1, 2], [0, 2, 3]], vertex_normals=[normal] * 4,
        albedo_factor=PATCH, face_rendering=rr.components.MeshFaceRendering.DoubleSided,
    ), static=True)  # fmt: skip
    ticks = np.linspace(-half, half, 5)
    grid = [[nearest + t * u - half * v, nearest + t * u + half * v] for t in ticks]
    grid += [[nearest - half * u + t * v, nearest + half * u + t * v] for t in ticks]
    recording.log(f"{base}/grid", rr.LineStrips3D(grid, radii=0.0015, colors=[PATCH_EDGE]), static=True)
    recording.log(f"{base}/edge", rr.LineStrips3D([[*corners, corners[0]]], radii=0.003, colors=[PATCH_EDGE],
                                                  labels=[equation(plane)]), static=True)  # fmt: skip
    nx, ny, nz = normal
    recording.log(f"{base}/normal", rr.Arrows3D(
        origins=[nearest], vectors=[0.5 * half * normal], radii=0.006, colors=[NORMAL],
        labels=[f"normal ({nx:.2f}, {ny:.2f}, {nz:.2f})"],
    ), static=True)  # fmt: skip
    if distance > 1e-9:
        recording.log(f"{base}/offset", rr.LineStrips3D([[np.zeros(3), nearest]], radii=0.002, colors=[FAINT],
                                                        labels=[f"{distance:.3g} m from the origin"]), static=True)
    origin_triad(recording, length=0.1)


register(TypeView(
    type=Plane,
    summary="A translucent patch of the (infinite) plane around its point nearest the origin, its normal arrow and its equation.",
    sample=lambda: Plane.from_point_normal(
        Point(0.45, 0.0, 0.30), Vector3(0.0, -math.sin(math.radians(20.0)), math.cos(math.radians(20.0)))
    ),
    views=world,
    tags=("3d",),
))  # fmt: skip
