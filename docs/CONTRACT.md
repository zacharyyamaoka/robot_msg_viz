# robot_msg_viz contract (v1, frozen 2026-11-24)

What a stock view is, what it must do, and how it is checked. `tests/test_contract.py`,
`tests/test_ruler_selftest.py` and `src/robot_msg_viz/ruler.py` are the ruler: a lane adding views never edits them.

## The public API (stable)

```python
import robot_msg_viz as viz

viz.show(*msgs, web=False, **named)       # open the viewer; keyword names become entity names
viz.save(path, *msgs, **named) -> Path    # .rrd with the default blueprint inside
viz.save_series(path, [(t, msg), ...])    # a timed recording on timeline "t"
viz.stream(name, spawn=True, web=False, save=None, blueprint=None) -> Stream   # live: live.log(msg, t=None, **named)
viz.blueprint(*msgs, timed=False, **named) -> rrb.Blueprint                   # the stock layout, for your own recording
viz.recording(*msgs, **named) -> (RecordingStream, Blueprint)
viz.log(msg, name, recording=rec)         # the dispatch table (ours over robot_msgs.viz.rerun)
viz.catalog() / viz.register(TypeView) / viz.view_for(type)
```

Plugins add use cases through the entry-point group `robot_msg_viz.gallery` (a function returning
`list[GalleryEntry]`, see `gallery.py`). This package never imports a plugin.

## Layers

`robot_msgs` (types + the stock logger `robot_msgs.viz.rerun`) → **robot_msg_viz** (views, samples, streaming,
gallery) → `robot_dashboards` (use cases). A view module never edits robot_msgs: when the stock logger is missing or
not enough for a type, it registers an override on `robot_msg_viz._engine.log` (`@log.register`).

## Entity layout (robot_msgs', so recordings overlay with `msg.show()` and traj_compare)

| Path | What |
|---|---|
| `/world/<name>/…` | anything spatial: frames, points, arrows, meshes, clouds, boxes, frustums |
| `/image/<name>/…` | 2D images and what is drawn on them (boxes, labels) |
| `/camera/<name>` | a pinhole (CameraInfo) |
| `/<field>/<column>/<name>` | scalar series on timeline `t` (seconds), e.g. `/positions/elbow/real` |
| `/notes/<name>` | text (markdown) |

`<name>` is the keyword the caller passed, else the type name. Use `robot_msg_viz._engine.part(name)` for a segment.

## A TypeView (one per robot_msgs type, in `views/<domain>.py`)

```python
register(TypeView(
    type=Pose,
    summary="One line: what you see in the viewer.",
    sample=lambda: ...,            # a showcase message: deterministic, small, looks like real robot data
    views=lambda name: [...],      # the rrb views this message needs when logged under `name`
    series=None,                   # or lambda: [(t, msg), ...] when the type reads as a signal over time
    code=None,                     # optional gallery snippet (default: viz.show(x) / viz.save(...))
    tags=("3d",),                  # optional: "3d", "2d", "plot", "text", "stream"
    page=None,                     # optional: lambda msg, name, **options: rrb.Blueprint, the whole layout when the
                                   #   message is shown alone (JointTrajectory: robot_msgs' stock trajectory viewer)
    options=None,                  # optional: lambda: {"robot": urdf}, what the gallery shows the sample with
))
```

`show` / `save` / `blueprint` / `recording` take `robot=`, `ghosts=` (and then `tool=`) as robot_msgs' `show` does;
they reach the type's `page` and its logger, and need exactly one message.

### Gates (tests/test_contract.py)

1. `sample()` is an instance of the type and returns an equal value every call (seed any randomness); a `series()`
   has ≥2 messages of the type at increasing times.
2. A real logger draws it: `viz.has_dedicated_logger(type)` (not the repr-as-text fallback).
3. Its gallery .rrd passes `rerun rrd verify`, carries a default blueprint with a viewport, and every view's
   `origin` has data at or under it in the file (read back by `rerun rrd print`, never by our own bookkeeping).
4. Streaming it puts each message on the timeline (≥2 rows), so a live view can be scrubbed.

### Rules beyond the gates (checked by looking, then by the agent's ruler)

- **Look at it.** `uv run python tools/shoot.py OUT --only <key>` shoots the real Rerun 0.40 web viewer headless;
  open the PNG and check the view shows the message clearly (not a blank grid, not a speck far from the camera).
  `OUT/shots.json` flags blank stills and viewer errors.
- **Angles in degrees on screen** (Zach's rule): plots of angles show degrees, the series named `… (deg)`; the data
  you log stays SI where it is geometry (transforms, quaternions).
- **Frame the content.** A 3D view of a small thing sets a sensible eye (`rrb.EyeControls3D(position=…, look_target=…)`)
  or logs enough context (a world origin triad, a ground grid) that the scale reads.
- **Colour by meaning**: class ids through `rr.AnnotationContext`, scores as a colour ramp, x/y/z = red/green/blue.
- **Small**: a gallery .rrd stays under 2 MB (it ships on the landing page).
- **No network, no files outside the package**: samples are built in code (seeded numpy), not downloaded.
- A type that cannot have a meaningful view (pure configuration) stays unsupported and listed; never fake one.
