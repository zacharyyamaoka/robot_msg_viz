# robot_msg_viz

**A stock [Rerun](https://rerun.io) viewer for every [robot_msgs](https://github.com/zacharyyamaoka/robot_msgs) type.**
Show one message, save it to a file, or stream them live. Each type opens in a preconfigured layout (a Rerun
blueprint) instead of a blank canvas.

```python
import robot_msg_viz as viz

viz.show(pose)                            # the stock view, in the desktop viewer
viz.show(target=planned, real=measured)   # keyword names become entity names, so the two overlay
viz.save("cloud.rrd", cloud)              # the blueprint is inside: `rerun cloud.rrd`

with viz.stream("arm") as live:           # live; the layout grows as new types arrive
    for state in joint_states:
        live.log(state)                   # time from header.stamp, else seconds since the stream began

viz.blueprint(image, detections)          # the stock layout, for a recording of your own
```

Robotics viewers mostly draw the same things again and again. Rerun and Foxglove let you build any viewer, but neither
ships a stock one per message type with sample data. This package does: **53 of robot_msgs' 68 types** have a curated
view today, each with a sample, a screenshot and an Open link. The other 15 are configuration or parameters with
nothing of their own to draw (`Header`, `JointTolerance`, the step conditions, …). They are listed, never hidden.

| Domain | Types with a stock view |
|---|---|
| std | 4 of 5 (ColorRGBA, DiagnosticRecord, LogRecord, Metrics) |
| geometry | 12 of 12 |
| shape | 3 of 4 (MeshResource needs its mesh file) |
| sensor | 5 of 5 (JointState, Image, CameraInfo, PointCloud, Imu) |
| vision | 8 of 8 |
| trajectory | 8 of 8 |
| control | 4 of 10 (ServoState, ServoCommand, JointCommandSequence, JointCommandStep) |
| manipulation | 5 of 6 |
| compare | 4 of 10 (Comparison, ErrorStats, Outcome, Violation; the rest is traj_compare's) |

Use-case dashboards (object detection, grasping, trajectory generation, …) live in the sibling package
[robot_dashboards](https://github.com/zacharyyamaoka/robot_dashboards) and appear on the same landing page.

## Install

```bash
pip install "robot-msg-viz @ git+https://github.com/zacharyyamaoka/robot_msg_viz"
```

It pins `rerun-sdk==0.40.0`, the same as robot_msgs and traj_compare. An `.rrd` must be opened by a viewer at least as
new as the SDK that wrote it.

## From the shell

```bash
robot-msg-viz list                 # every type: ✓ has a stock view, · not yet
robot-msg-viz demo PointCloud      # its sample in the desktop viewer (--web for the browser)
robot-msg-viz save Imu imu.rrd     # its sample as a file
```

## The landing page

A Types tab lists every robot_msgs type: All or Supported, searchable, grouped by domain. A Use cases tab lists every
installed dashboard. Each entry has:
- its screenshot;
- **Open in browser**: Rerun's hosted web viewer plays the sample, nothing to install;
- **Open on desktop**: copies `uvx --from rerun-sdk==0.40.0 rerun <url>`. Rerun registers no URL scheme, so a web page
  cannot start the native viewer itself;
- **Download .rrd** and the code.

```bash
python -m robot_msg_viz.site build website/dist --shots <dir of stills>
python -m robot_msg_viz.site serve website/dist --port 4493
```

`serve` also runs the bundled web viewer locally, so Open in browser works offline.

## How it fits with robot_msgs

robot_msgs already has `msg.show()` (Rerun) and a stock logger per type in `robot_msgs.viz.rerun`. This package builds
on that logger rather than copying it. Its dispatch table (`viz.log`) falls through to robot_msgs' logger, and a view
module overrides a type only where the stock one is missing or not enough. Examples:
- a JointState as plots in degrees instead of a static table;
- a PointCloud coloured by height;
- a CameraInfo as a frustum.

For a JointTrajectory with its arm, use robot_msgs' URDF viewer directly (`traj.show(robot="ur5e.urdf")`, or
`viz.log(traj, "plan", recording=rec, robot=urdf)`, which passes `robot=` through to it).

## Adding a view

One `TypeView` per type, in `src/robot_msg_viz/views/<domain>.py`. The contract and the gates it must pass are in
[docs/CONTRACT.md](docs/CONTRACT.md):
- its sample is the type;
- a real logger draws it;
- its `.rrd` carries a blueprint whose every view has data, read back by `rerun rrd print`;
- streaming puts it on the timeline.

Then shoot it in the real web viewer and look at it:

```bash
uv run python tools/shoot.py /tmp/shots --only geometry/Twist
```

The shooter flags a blank still by comparing it with the same recording shot with its data dropped.
[robot_viz_agent](https://github.com/zacharyyamaoka/robot_viz_agent) scores every entry against these gates.

## Develop

```bash
env -u VIRTUAL_ENV uv run --directory /home/bam/robot_msg_viz --extra test python -m pytest -q
```

MIT licensed.
