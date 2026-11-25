# The gallery site

The landing page for robot_msg_viz: a **Types** tab (every robot_msgs type, grouped by domain; the ones with a stock
view are clickable, the rest are greyed "no view yet"; an All / Supported filter that the browser remembers; search)
and a **Use cases** tab (dashboards that plugins such as robot_dashboards register: object detection, grasping,
trajectory generation, ...). Every supported entry gets a page with its screenshot, **Open in browser** (Rerun's web
viewer playing the entry's sample .rrd, inline), **Open on desktop** (copies a one-line `uvx ... rerun <url>`),
**Download .rrd**, and the code that makes it.

Everything is generated from `robot_msg_viz.gallery.entries()`; nothing is written per page. The builder is
`src/robot_msg_viz/site/` (stdlib HTML, robotmsgs.com's look; no npm).

## Build and look at it

```bash
env -u VIRTUAL_ENV uv run --directory /home/bam/robot_msg_viz python tools/shoot.py /tmp/rmv-shots
env -u VIRTUAL_ENV uv run --directory /home/bam/robot_msg_viz python -m robot_msg_viz.site build /home/bam/robot_msg_viz/website/dist --shots /tmp/rmv-shots/shots
env -u VIRTUAL_ENV uv run --directory /home/bam/robot_msg_viz python -m robot_msg_viz.site serve /home/bam/robot_msg_viz/website/dist --port 4493
```

1. `tools/shoot.py` shoots every supported entry in the real Rerun 0.40 web viewer (headless) and writes
   `shots/<slug>.png`; skip it and every card shows a "no screenshot yet" placeholder.
2. `build` writes `index.html`, `types/<slug>.html`, `use-cases/<slug>.html`, `rrd/<slug>.rrd`,
   `shots/<slug>.webp` + `<slug>-480.webp` and `gallery.json`. All links are relative, so the folder works under any
   URL prefix. It clears only what it writes, never the whole folder.
3. `serve` serves the folder on 127.0.0.1 with `Access-Control-Allow-Origin: *` and starts Rerun's bundled web viewer
   on the next port; `/viewer.json` tells the pages to use that local viewer. Ctrl-C stops both.

## Deploying

Upload the built folder to any static host. With no `/viewer.json` the pages use the hosted viewer pinned to the
SDK version (`https://app.rerun.io/version/0.40.0/`), which fetches the .rrd from your host, so the host must send
`Access-Control-Allow-Origin: *` on `rrd/*` (Cloudflare Pages: a `_headers` file with `/rrd/*` and that header).
