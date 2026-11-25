# robot_msg_viz — agent notes

Read `docs/CONTRACT.md` first: the API, the entity layout, what a TypeView is and the gates it must pass.

- Tests (the ruler; never edit `tests/test_contract.py`, `tests/test_ruler_selftest.py`, `src/robot_msg_viz/ruler.py`):
  `env -u VIRTUAL_ENV uv run --directory /home/bam/robot_msg_viz --extra test python -m pytest -q`
- Stills in the real Rerun 0.40 web viewer (headless; never a native window, never the desktop browser pane):
  `env -u VIRTUAL_ENV uv run --directory /home/bam/robot_msg_viz python tools/shoot.py <out-dir> --only <key>`
  then LOOK at `<out-dir>/shots/<slug>.png`.
- The stock logger lives in robot_msgs (`/home/bam/robot_msgs/src/robot_msgs/viz/rerun.py`): read it, never edit it
  from here; override a type with `@log.register` in `src/robot_msg_viz/views/<domain>.py`.
- Example messages for every type: `/home/bam/robot_msgs/tests/catalog.py` (minimal; gallery samples should look like
  real robot data).
- The gallery site: `src/robot_msg_viz/site/` builds `website/dist/` from `gallery.entries()`.
