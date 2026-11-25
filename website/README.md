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
env -u VIRTUAL_ENV uv run --directory /home/bam/robot_msg_viz python -m robot_msg_viz.site build /home/bam/robot_msg_viz/website/dist --no-plugins --shots /home/bam/robot_msg_viz/website/media/shots --hero /home/bam/robot_msg_viz/website/media/hero.mp4
env -u VIRTUAL_ENV uv run --directory /home/bam/robot_msg_viz python -m robot_msg_viz.site serve /home/bam/robot_msg_viz/website/dist --port 4493
```

1. `build` writes `index.html`, `types/<slug>.html`, `use-cases/<slug>.html`, `rrd/<slug>.rrd`,
   `shots/<slug>.webp` + `<slug>-480.webp`, `media/hero.mp4` + `hero.webp` and `gallery.json`. All links are
   relative, so the folder works under any URL prefix. It clears only what it writes, never the whole folder.
2. `serve` serves the folder on 127.0.0.1 with `Access-Control-Allow-Origin: *` and starts Rerun's bundled web viewer
   on the next port; `/viewer.json` tells the pages to use that local viewer. Ctrl-C stops both.

Which entries: `--no-plugins` builds robot_msg_viz's own types only, as the public page does (a plugin may be private;
nothing it ships may reach the public page). Without it, every installed plugin's use cases are listed too.
`--use-cases-only --title … --intro … --viewer --private` is a plugin's own page (robot_dashboards builds its page this
way): no Types tab, its own headline and intro, Rerun's web viewer served from the page itself (`--viewer`, from the
pinned rerun-sdk, the wasm in parts under 20 MiB for Cloudflare Workers' 25 MiB per-file limit) and Open on desktop
from the downloaded file (`--private`: behind a sign-in no viewer can fetch a URL).

## Media (committed)

The stills and the hero video are committed in `website/media/` (~3 MB) and CI builds the pages and `.rrd` files
around them. `tests/test_site_public.py` fails when a supported type has no still, when a still is not a type's, or when
the media pass 6 MB. After a view changes, shoot it again and look at it:

```bash
env -u VIRTUAL_ENV uv run --directory /home/bam/robot_msg_viz python tools/shoot.py /tmp/rmv-shots --only trajectory/JointTrajectory
env -u VIRTUAL_ENV uv run --directory /home/bam/robot_msg_viz python tools/site_media.py /tmp/rmv-shots website/media/shots --kind type
env -u VIRTUAL_ENV uv run --directory /home/bam/robot_msg_viz python tools/hero_video.py website/media/hero.mp4 --plan website/hero.json
```

`tools/hero_video.py` records the hero in the real Rerun web viewer from `website/hero.json` (the V1 JointTrajectory
first, its tabs clicked through, then three more views), cuts the viewer's loading out and writes an H.264 mp4 and its
WebP poster.

## Deploying

`.github/workflows/site.yml` tests and builds on every push and PR and deploys to GitHub Pages on a push to main:
https://zacharyyamaoka.github.io/robot_msg_viz/. With no `/viewer.json` the pages use the hosted viewer pinned to
the SDK version (`https://app.rerun.io/version/0.40.0/`), which fetches the .rrd from the page's host; GitHub Pages
sends `Access-Control-Allow-Origin: *`.
