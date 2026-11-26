"""Sample data for the stream and dataset views: one UR5e pick, recorded in sim and on the real arm.

Not a view module (it registers nothing): `streams.py` and `runs.py` draw these. Everything is built in code from
seeded numpy (docs/CONTRACT.md: no network, no files), cached, so a sample is the same object every call.

The story (Zach, 2026-11-25: "record every sensor/actuator stream in sim and on the real robot, then compare each
stream"): the plan is `trajectory.pick_move` played at a speed factor; each source records

- ``joints``        JointTrajectory, the encoders;
- ``servo_state``   ServoStateTrajectory, what the drives report (position, velocity, effort, temperature, bus
                    voltage, fault code);
- ``servo_command`` ServoCommandTrajectory, what was sent (targets, gravity feed-forward, gains, caps, latency);
- ``imu``           ImuTrajectory, a wrist IMU on the tool flange;
- ``wrench``        WrenchTrajectory, the wrist F/T sensor: the grasp's contact, the payload, the release;
- ``camera``        ImageTrajectory, the overhead belt camera;
- ``scan``          PointCloud, a depth scan of the belt.

`full_run` (the Run, Recording and RunComparison samples) has every stream; the three-run `sweep` (the Dataset and
DatasetComparison samples) only the joints, the drive log and the F/T sensor.

Sim follows the plan closely (an 8 ms servo lag, no sag, clean sensors); the real arm lags 15 ms, sags under
gravity, runs warmer, its sensors are noisy and biased, its wrist rings (6 Hz) after each stop, and at full speed wrist 1
trips a following-error fault for 60 ms that sim never shows.
"""

from __future__ import annotations

import functools

import numpy as np
from robot_msgs.compare import Dataset, Recording, Run
from robot_msgs.geometry.rotations import quat_from_euler, quat_to_matrix
from robot_msgs.sensor import PointCloud
from robot_msgs.std import Header
from robot_msgs.trajectory import (
    ImageTrajectory,
    ImuTrajectory,
    JointTrajectory,
    ServoCommandTrajectory,
    ServoStateTrajectory,
    WrenchTrajectory,
)

from robot_msg_viz.views import vision
from robot_msg_viz.views._common import rng
from robot_msg_viz.views.trajectory import pick_move, tool_path

RATE_HZ = 50.0
SPEEDS = (0.5, 0.75, 1.0)  # the sweep's speed factors (1.0 = the plan as written)
G = 9.80665
FAULT_FOLLOWING = 33  # the drive's "following error" code (a made-up but plausible board code)

# per joint: link inertia seen at the joint (kg·m²), viscous friction (N·m·s/rad), Coulomb friction (N·m)
_INERTIA = np.array([2.2, 3.1, 1.1, 0.12, 0.10, 0.04])
_VISCOUS = np.array([4.0, 4.0, 3.0, 0.6, 0.6, 0.4])
_COULOMB = np.array([3.0, 3.5, 2.2, 0.6, 0.5, 0.4])


@functools.cache
def plan(speed: float = 1.0) -> JointTrajectory:
    """The pick at `speed` (time divided by it), 50 Hz, with its derivatives."""
    base = pick_move(RATE_HZ)
    return JointTrajectory(t=np.asarray(base.t) / speed, joint_names=base.joint_names,
                           positions=base.positions, limits=base.limits, header=base.header).with_derivatives()  # fmt: skip


def _gravity(q: np.ndarray) -> np.ndarray:
    """A UR5e-like gravity load (N·m) on each joint: the lift and elbow carry the arm, the wrists a little."""
    lift, elbow = q[:, 1], q[:, 1] + q[:, 2]
    return np.stack([np.zeros(len(q)), 46.0 * np.cos(lift) + 13.0 * np.cos(elbow), 13.0 * np.cos(elbow),
                     1.4 * np.cos(elbow + q[:, 3]), 0.25 * np.sin(q[:, 4]), np.zeros(len(q))], axis=1)  # fmt: skip


def _lagged(t: np.ndarray, command: np.ndarray, tau: float) -> np.ndarray:
    out = np.empty_like(command)
    out[0] = command[0]
    for k, step in enumerate(np.diff(t)):
        out[k + 1] = out[k] + (1.0 - np.exp(-step / tau)) * (command[k + 1] - out[k])
    return out


@functools.cache
def _arm(source: str, speed: float) -> dict[str, np.ndarray]:
    """What `source` ("sim" or "real") measured on the arm at `speed`: q, qd, effort, temperature, voltage, fault."""
    target = plan(speed)
    t = np.asarray(target.t)
    real = source == "real"
    noise = rng(int(100 * speed) + (7 if real else 3))
    q = _lagged(t, np.asarray(target.positions), 0.015 if real else 0.008)
    if real:
        # gravity sag on the lift and elbow (harmonic drive compliance), and the encoders' 0.01° of noise
        q = q + np.radians([0.0, -0.18, 0.09, 0.0, 0.0, 0.0]) * np.cos(q[:, [1]])
        q = q + np.radians(noise.normal(0.0, 0.01, q.shape))
    qd = np.gradient(q, t, axis=0)
    qdd = np.gradient(qd, t, axis=0)
    friction = (1.4 if real else 1.0) * (_VISCOUS * qd + _COULOMB * np.tanh(qd / 0.02))
    effort = _gravity(q) + _INERTIA * qdd + friction
    if real:
        effort = effort * 1.04 + noise.normal(0.0, 0.25, effort.shape)
    ambient = np.array([34.0, 38.0, 36.0, 31.0, 31.0, 30.0]) if real else np.full(6, 25.0)
    heat = np.cumsum(effort**2, axis=0) * np.gradient(t)[:, None] * (0.0009 if real else 0.0006)
    temperature = np.round(ambient + heat, 1)
    bus = 48.0 - (0.004 if real else 0.002) * np.abs(effort).sum(axis=1, keepdims=True)
    voltage = np.repeat(bus + (noise.normal(0.0, 0.02, (len(t), 1)) if real else 0.0), 6, axis=1)
    fault = np.zeros(q.shape, dtype=np.int64)
    if real and speed >= 1.0:
        # wrist 1 trips its following-error limit at its fastest moment, for 60 ms, then the drive recovers
        peak = int(np.argmax(np.abs(np.asarray(target.velocities)[:, 3])))
        fault[peak : peak + 3, 3] = FAULT_FOLLOWING
    return {"t": t, "q": q, "qd": qd, "effort": effort, "temperature": temperature, "voltage": voltage, "fault": fault}


@functools.cache
def joints(source: str, speed: float = 1.0) -> JointTrajectory:
    arm = _arm(source, speed)
    return JointTrajectory(t=arm["t"], joint_names=plan(speed).joint_names, positions=arm["q"],
                           velocities=arm["qd"], header=Header(stamp=0.0, frame_id="base_link"))  # fmt: skip


@functools.cache
def servo_states(source: str = "real", speed: float = 1.0) -> ServoStateTrajectory:
    """The drives' own report at 50 Hz: what `from_servo_states` makes of a ServoState log."""
    arm = _arm(source, speed)
    return ServoStateTrajectory(
        t=arm["t"], joint_names=plan(speed).joint_names, positions=arm["q"], velocities=arm["qd"],
        effort=arm["effort"], temperature=arm["temperature"], voltage=arm["voltage"], fault=arm["fault"],
        header=Header(stamp=0.0, frame_id="base_link"),
    )  # fmt: skip


@functools.cache
def servo_commands(source: str = "real", speed: float = 1.0) -> ServoCommandTrajectory:
    """What the controller sent: the plan's targets, a gravity feed-forward, the gains and caps, and how late each
    command left after the state it was computed from (sim: a steady 1 ms; real: 1.8 ms with jitter)."""
    target = plan(speed)
    t = np.asarray(target.t)
    n = len(t)
    real = source == "real"
    noise = rng(int(100 * speed) + (17 if real else 13))
    latency = 0.0018 + np.abs(noise.normal(0.0, 0.0004, n)) if real else np.full(n, 0.001)
    stiffness = np.tile([900.0, 900.0, 600.0, 120.0, 120.0, 80.0], (n, 1))
    if real:
        stiffness = stiffness * 0.85  # the real cell runs softer gains (tuned on the rig)
    return ServoCommandTrajectory(
        t=t, joint_names=target.joint_names, positions=target.positions, velocities=target.velocities,
        feedforward_effort=_gravity(np.asarray(target.positions)), stiffness=stiffness,
        damping=np.tile([30.0, 30.0, 20.0, 4.0, 4.0, 2.5], (n, 1)), max_effort=np.tile([150.0, 150.0, 150.0, 28.0, 28.0, 28.0], (n, 1)),
        channels={"latency": latency}, header=Header(stamp=0.0, frame_id="base_link"),
    )  # fmt: skip


@functools.cache
def imu(source: str = "real", speed: float = 1.0) -> ImuTrajectory:
    """A wrist IMU on the tool flange through the pick: the tool's turn, its accelerations and gravity, in the IMU's
    frame. Real: gyro and accelerometer noise and bias, and a 6 Hz ring after each stop (the wrist's resonance)."""
    path = tool_path(RATE_HZ)
    t = np.asarray(path.t) / speed
    positions = np.asarray(path.positions)
    quaternions = np.asarray(path.orientations)
    rotations = quat_to_matrix(quaternions)
    accel_world = np.gradient(np.gradient(positions, t, axis=0), t, axis=0)
    specific = np.einsum("nji,nj->ni", rotations, accel_world + [0.0, 0.0, G])  # R^T (a + g): what it reads
    yaw = np.unwrap(np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0]))
    yaw_rate = np.gradient(yaw, t)
    gyro = np.einsum("nji,j->ni", rotations, [0.0, 0.0, 1.0]) * yaw_rate[:, None]
    roll = np.zeros(len(t))
    if source == "real":
        noise = rng(int(100 * speed) + 29)
        speed_now = np.linalg.norm(np.gradient(positions, t, axis=0), axis=1)
        moving = speed_now > 0.01
        stops = np.nonzero(moving[:-1] & ~moving[1:])[0] + 1  # the samples where a move ends
        roll = np.zeros(len(t))
        for stop in stops:
            after = np.clip(t - t[stop], 0.0, None)
            roll += (t >= t[stop]) * np.exp(-after / 0.15) * np.sin(2 * np.pi * 6.0 * after)
        roll = np.radians(0.6) * roll
        gyro = gyro + np.column_stack([np.gradient(roll, t), np.zeros(len(t)), np.zeros(len(t))])
        gyro = gyro + noise.normal(0.0, 0.002, gyro.shape) + [0.0012, -0.0008, 0.0005]
        specific = specific + noise.normal(0.0, 0.04, specific.shape) + [0.03, -0.02, 0.05]
        specific[:, 1] += G * roll
    turn = quat_from_euler(roll, np.zeros(len(t)), np.zeros(len(t)))
    orientations = _quat_multiply(quaternions, turn)
    return ImuTrajectory(t=t, orientations=orientations, angular_velocities=gyro, linear_accelerations=specific,
                         header=Header(stamp=0.0, frame_id="imu_link"))  # fmt: skip


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a.T
    bx, by, bz, bw = b.T
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], axis=1)  # fmt: skip


def _ease(t: np.ndarray, start: float, length: float) -> np.ndarray:
    s = np.clip((t - start) / length, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


@functools.cache
def wrench(source: str = "real", speed: float = 1.0) -> WrenchTrajectory:
    """The wrist F/T sensor (tool z down, N and N·m) through the pick: free air, the fingers press into the part
    (contact at the bottom of the descent), the 0.8 kg part hangs on the way over, released at the place pose.
    Real: a contact spike that overshoots, 0.4 N of bias, noise; the part's weight is the same in both."""
    path = tool_path(RATE_HZ)
    t = np.asarray(path.t) / speed
    s = 1.0 / speed
    descent_end, lift_end = (1.4 + 0.25 + 0.7) * s, (1.4 + 0.25 + 0.7 + 0.25) * s
    place = t[-1] - 0.05 * s
    press = _ease(t, descent_end - 0.08 * s, 0.08 * s) * (1.0 - _ease(t, lift_end - 0.02 * s, 0.06 * s))
    carry = _ease(t, lift_end - 0.02 * s, 0.1 * s) * (1.0 - _ease(t, place, 0.04 * s))
    accel_z = np.gradient(np.gradient(np.asarray(path.positions)[:, 2], t), t)
    fz = -18.0 * press - 0.8 * (G + accel_z) * carry
    force = np.column_stack([1.1 * press + 0.3 * carry, -0.5 * press, fz])
    lever = np.array([0.012, -0.006, 0.0])  # the contact sits 13 mm off the sensor axis
    if source == "real":
        noise = rng(int(100 * speed) + 41)
        spike = 9.0 * np.exp(-np.maximum(t - descent_end, 0.0) / (0.03 * s)) * (t >= descent_end - 0.02 * s) * press
        force = force + np.column_stack([np.zeros(len(t)), np.zeros(len(t)), -spike])
        force = force + noise.normal(0.0, 0.15, force.shape) + [0.4, -0.25, 0.4]
    torque = np.cross(lever, force)
    if source == "real":
        torque = torque + rng(int(100 * speed) + 43).normal(0.0, 0.004, torque.shape)
    return WrenchTrajectory(t=t, forces=force, torques=torque, header=Header(stamp=0.0, frame_id="ft_sensor"))


# ── the belt camera and its depth scan ──────────────────────────────────────────────────────────────────────────────

CAMERA_SHRINK = 4  # 640 × 480 → 160 × 120: a recognisable picture, small enough for the gallery
BELT_SPEED = 0.3  # m/s, along the image's x


def _belt_frame(picture: np.ndarray, metres: float) -> np.ndarray:
    shift = int(round(metres * vision._PX_PER_M))
    moved = np.roll(picture, shift, axis=1).astype(np.float64)
    h, w = moved.shape[:2]
    return moved.reshape(h // CAMERA_SHRINK, CAMERA_SHRINK, w // CAMERA_SHRINK, CAMERA_SHRINK, 3).mean(axis=(1, 3))


@functools.cache
def camera(source: str = "real", frames: int = 16, rate_hz: float = 10.0) -> ImageTrajectory:
    """The overhead belt camera (rgb8, 160 × 120) as the belt carries the parts past at 0.3 m/s.

    Sim renders the scene clean; the real camera is 6 % brighter (auto-exposure), vignetted at the corners, and
    noisy, so the error frames show where the two differ."""
    picture = vision._scene().picture
    t = np.arange(frames) / rate_hz
    stack = np.stack([_belt_frame(picture, BELT_SPEED * ti) for ti in t])
    if source == "real":
        h, w = stack.shape[1:3]
        y, x = np.mgrid[0:h, 0:w]
        r2 = ((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2
        vignette = 1.0 - 0.22 * r2
        stack = stack * 1.06 * vignette[None, :, :, None] + rng(51).normal(0.0, 3.0, stack.shape)
    frames_u8 = np.clip(np.round(stack), 0, 255).astype(np.uint8)
    exposure = np.full(frames, 0.008) if source == "sim" else 0.008 * (1.0 + 0.04 * np.sin(np.arange(frames)))
    return ImageTrajectory(t=t, encoding="rgb8", frames=frames_u8, channels={"exposure_s": exposure},
                           header=Header(stamp=0.0, frame_id="belt_camera_optical_frame"))  # fmt: skip


@functools.cache
def scan(source: str = "real") -> PointCloud:
    """A depth scan of the belt and its parts (every 3rd point of the belt scene's cloud, ~4 k points). Real: 1.5 mm of
    depth noise, and the bottle reads 6 mm low (its cap sits crushed): where a sim-to-real check should point."""
    scene = vision._scene()
    keep = np.arange(0, len(scene.points), 3)
    points = scene.points[keep].astype(np.float64)
    colors = scene.colors[keep]
    if source == "real":
        noise = rng(61)
        points = points + np.column_stack([np.zeros((len(points), 2)), noise.normal(0.0, 0.0015, len(points))])
        bottle = scene.owners[keep] == 0
        points[bottle, 2] -= 0.006
    return PointCloud(points=points.astype(np.float32), channels={"rgb": colors},
                      header=Header(stamp=0.0, frame_id="base_link"))  # fmt: skip


# ── recordings, runs, the sweep ─────────────────────────────────────────────────────────────────────────────────────


def recording(source: str, speed: float, *, full: bool = False) -> Recording:
    """One source's recording; `full` adds the command log, the IMU, the camera and the scan to the joints, the drive
    log and the F/T sensor."""
    if source == "target":
        return Recording(streams={"joints": plan(speed)}, header=Header(meta={"planner": "min-jerk pick v3"}))
    streams = {"joints": joints(source, speed), "servo_state": servo_states(source, speed)}
    if full:
        streams["servo_command"] = servo_commands(source, speed)
        streams["imu"] = imu(source, speed)
    streams["wrench"] = wrench(source, speed)
    if full:
        streams["camera"] = camera(source, frames=6)
        streams["scan"] = scan(source)
    meta = {"engine": "mujoco 3.3"} if source == "sim" else {"rig": "ur5e cell 2", "firmware": "5.21"}
    return Recording(streams=streams, header=Header(meta=meta))


def run(speed: float, *, full: bool = False) -> Run:
    return Run(
        name=f"pick_{speed:g}x",
        recordings={source: recording(source, speed, full=full) for source in ("target", "sim", "real")},
        tags={"speed": speed, "payload_kg": 0.8, "gripper": "2f-85"},
        title=f"pick at {speed:g}× speed",
    )


@functools.cache
def full_run() -> Run:
    """The full-speed pick with every stream: the command log, the IMU, the camera and the scan too."""
    return run(1.0, full=True)


@functools.cache
def sweep() -> Dataset:
    """Three runs of the pick, 0.5× to 1× speed, each with the joints, the drive log and the F/T sensor.

    WHY not more (`full_run` has every stream): the gallery's .rrd stays under 2 MB (docs/CONTRACT.md); four runs of
    every stream were 4.5 MB, four of these three streams 2.3 MB. The drill-down reads the same for any stream."""
    return Dataset(runs=tuple(run(speed) for speed in SPEEDS), title="pick speed sweep")
