"""Build the 2026-11-25 report: your Rerun pick applied (V1 lone, V2 target vs run) and the two landing pages.

    python3 /home/bam/robot_msg_viz/docs/build_landing_report.py

Reads the real captures the builders left (reports/media/landing-pages-2026-11-25/ here, and traj_compare's
reports/media/rerun-v2-comparison-2026-11-25/), re-encodes them small (ffmpeg, Pillow) and writes one self-contained
page, reports/media/landing-pages-and-rerun-pick-2026-11-25.html (media inline, so it lives in the gitignored
reports/media/). WHY inline and small: the page must open from Zach's phone (SendUserFile) and stay under the desktop
preview's ~1.9 MB ceiling for a media page.
"""

from __future__ import annotations

import base64
import html
import io
import subprocess
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
MEDIA = REPO / "reports" / "media" / "landing-pages-2026-11-25"
V2 = Path("/home/bam/traj_compare/reports/media/rerun-v2-comparison-2026-11-25")
OUT = REPO / "reports" / "media" / "landing-pages-and-rerun-pick-2026-11-25.html"
UR_CAPTION = ("Your UR workbench: a UR5e sim run (ctc_tuned_v2, r0) against its target on workspace-dev-029, at 3.11 s. "
              "It tracks within 0.1 mm, so the magnifier picks ×500 (it was ×2000 before today's cap), and all seven "
              "tabs are there")  # fmt: skip


def still(path: Path, width: int = 960, quality: int = 70) -> str:
    image = Image.open(path).convert("RGB")
    if image.width > width:
        image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, "WEBP", quality=quality, method=6)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def clip(path: Path, name: str, width: int = 880, crf: int = 33, fps: int = 15) -> str:
    """The clip re-encoded small, cached under MEDIA/report-cache/<name>.mp4 so a rebuild never re-encodes.

    WHY its own systemd scope: the shared agent Bash pool shrinks to a 4 GiB floor when the machine is full, and the
    OOM killer took ffmpeg there three times on 2026-11-25; the opt-out scope is the sanctioned route
    (zach_brain Settings/Memory/reference_agent_memory_cap.md).
    """
    out = MEDIA / "report-cache" / f"{name}.mp4"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["systemd-run", "--user", "--scope", "--quiet", "--collect", "-p", "MemoryMax=1G", "-p", "MemorySwapMax=0",
             "-p", "OOMPolicy=continue", "--",
             "ffmpeg", "-v", "error", "-y", "-threads", "1", "-i", str(path), "-an", "-vf", f"scale={width}:-2,fps={fps}",
             "-c:v", "libx264", "-threads", "1", "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(out)],
            check=True,
        )  # fmt: skip
    return "data:video/mp4;base64," + base64.b64encode(out.read_bytes()).decode()


def figure(src: str, caption: str) -> str:
    return f'<figure><img src="{src}" alt="{html.escape(caption)}" loading="lazy"><figcaption>{caption}</figcaption></figure>'


def video(src: str, poster: str, caption: str) -> str:
    return (f'<figure><video src="{src}" poster="{poster}" autoplay muted loop playsinline controls></video>'
            f"<figcaption>{caption}</figcaption></figure>")  # fmt: skip


def cmd(text: str) -> str:
    return f"<pre><code>{html.escape(text)}</code></pre>"


def main() -> None:
    ur_workbench = V2 / "ur-workbench-v2.png"
    rmv_poster = still(MEDIA / "robot-msg-viz-hero-poster.webp", 880)
    rd_poster = still(MEDIA / "robot-dashboards-hero-poster.webp", 880)
    v2_poster = still(V2 / "error-1-t1.2.png", 880)

    body = f"""
<header>
  <p class="eyebrow">robot_msg_viz · robot_dashboards · traj_compare · 2026-11-25</p>
  <h1>Your Rerun pick is the stock viewer, and robot_msg_viz has a live landing page</h1>
  <p class="verdict"><b>A lone JointTrajectory opens V1. A run with a target opens V2 "Target vs run".</b>
  <a href="https://zacharyyamaoka.github.io/robot_msg_viz/">zacharyyamaoka.github.io/robot_msg_viz</a> is live.
  Everything is pushed to <code>main</code>: robot_msg_viz <code>4469a06</code>, robot_msgs <code>e5f375d</code>,
  traj_compare <code>15e9f4b</code>, robot_dashboards <code>523a4d7</code>.
  <b>robot_dashboards' page is built but not live.</b> The repo is private, so your rule puts it on Cloudflare behind a
  sign-in, and that needs your one-time setup (or make the repo public and it moves to github.io).</p>
</header>

<section>
  <h2>1 · Your pick: V1 for a lone trajectory, V2 when there's a target</h2>
  <p>You said a normal trajectory without a comparison keeps V1. V1 is now <b>robot_msg_viz's stock JointTrajectory
  view</b>: <code>viz.show(traj, robot="ur5e.urdf")</code> shows the arm on the left and the tabs
  q · qd · qdd · qddd · q / qd / qdd · TCP on the right, all in degrees. With no URDF, you get the same tabs at full
  width, minus TCP; the old one-plot-per-field view is gone.</p>
  <div class="grid2">
    {figure(still(MEDIA / "v1-joint-trajectory-ur5e.png"), "V1 with a URDF: the gallery's real UR5e sample")}
    {figure(still(MEDIA / "v1-joint-trajectory-no-urdf.png"), "V1 without a URDF: the same tabs at full width")}
  </div>
  <p>For a <b>run that has a target</b>, Open in Rerun is now V2, the pick page's recommendation:</p>
  <ul>
    <li>the run is drawn as the steel arm, with its deviation drawn ×50 as a magenta skeleton and labelled "TCP off 3.3 mm";</li>
    <li>an Error tab comes first: RMS 1.09 mm, worst 2.11 mm at 1.42 s, each joint's error in degrees, and a
    target-vs-run table at the playhead.</li>
  </ul>
  <p>It's in traj_compare's comparison report, its dataset drill-down and its runs workbench (with one run ticked).
  It switches on wherever the robot comes with a URDF.</p>
  {video(clip(V2 / "hero.mp4", "v2"), v2_poster, "V2 in the real Rerun 0.40 web viewer, on the Trajectory Tracking Agent's p6s run vs its target (UR5e workspace-dev-029), half speed")}
  <div class="grid2">
    {figure(still(V2 / "error-3-t1.4.png"), "Error tab at 1.4 s: the run's own score, each joint in degrees, target vs run")}
    {figure(still(V2 / "skeleton-x50-1.2-1.5s-zoom.png"), "The ×50 skeleton leaving the steel arm between 1.2 and 1.5 s")}
  </div>
  <h3>In your UR workbench</h3>
  <p>arm_motion_bench <code>7d51971</code> gives the UR5e its mounted URDF, which is the same file the V1
  recordings already use. Open in Rerun on a UR run with its target now opens V2, both from the runs workbench (with
  one run ticked) and from the dashboard drill-down. A lone plan in the coverage workbench still opens V1. The robot is
  loaded when you click, so no site rebuild is needed.</p>
  <p>Checking this run turned up two defects in what had just landed, and both are fixed (traj_compare
  <code>5e13af2</code>, then <code>cb42943</code>; CI green on 3.11, 3.12 and 3.13):</p>
  <ul>
    <li><b>The magnifier overshot.</b> On a run tracked to 0.1 mm it picked ×2000, which turned 0.04° joint errors into
    77° and threw the skeleton far off the arm. It now also keeps every magnified joint within 35°, which gives ×500
    here. The pick round's tour stays at ×50.</li>
    <li><b>The two paths showed different tabs</b> (5 vs 7). A run stored without accelerations is now differentiated
    on every V2 path, with the drill-down's own helper.</li>
  </ul>
  {figure(still(ur_workbench), UR_CAPTION)}
</section>

<section>
  <h2>2 · robot_msg_viz: live on GitHub Pages</h2>
  <p>The gallery site the package already built is now deployed from <code>main</code> on every push: CI runs the
  tests, builds the pages and recordings, runs a privacy grep, and deploys. The index opens on a hero video recorded
  from the real viewer, with V1 first. Every type page keeps <b>Open in browser</b>, where Rerun's hosted viewer
  plays that type's sample .rrd straight from github.io (served with <code>access-control-allow-origin: *</code>).
  The README's first line and the repo's homepage link to the page.</p>
  {video(clip(MEDIA / "robot-msg-viz-hero.mp4", "rmv"), rmv_poster, "The live index's hero: recorded from the real Rerun web viewer")}
  <div class="grid2">
    {figure(still(MEDIA / "live-03-open-in-browser-hosted-viewer.png"), "Live: JointTrajectory's page, Open in browser playing its .rrd in the hosted viewer")}
    {figure(still(MEDIA / "live-04-use-cases-empty.png"), "Live: the Use cases tab is empty on purpose while robot_dashboards is private")}
  </div>
</section>

<section>
  <h2>3 · robot_dashboards: built, waiting on hosting</h2>
  <p>The same builder, in a new plugin-only mode, gives the seven dashboards their own site:</p>
  <ul>
    <li>a hero video;</li>
    <li>a card per dashboard, each with the question it answers;</li>
    <li>a page per dashboard with stills, code, and the adapters → robot_msgs → viewers layering.</li>
  </ul>
  <p>Behind a sign-in, Rerun's hosted viewer can't fetch the recordings, so the page serves its own copy of the viewer.
  The viewer program is 43 MiB, which is over Cloudflare's 25 MiB per-file limit, so it ships in three parts that
  its loader joins back together.</p>
  {video(clip(MEDIA / "robot-dashboards-hero.mp4", "rd"), rd_poster, "robot_dashboards' index hero, recorded from the real viewer")}
  <div class="grid2">
    {figure(still(MEDIA / "dashboards-02-index-cards.png"), "The seven dashboards, each with the question it answers")}
    {figure(still(MEDIA / "dashboards-04-open-in-browser-self-hosted.png"), "Grasping, Open in browser, in the self-hosted viewer (checked locally)")}
  </div>
</section>

<section class="decisions">
  <h2>Where it stands</h2>
  <h3>Done and proved</h3>
  <ul>
    <li><b>V1 stock</b> (robot_msg_viz <code>1340d44</code>, on robot_msgs <code>e5f375d</code>):
      <ul>
        <li><code>tests/test_joint_trajectory_v1_stock.py</code> passes 4 of 4 on main and failed 4 of 4 on the old code;</li>
        <li>robot_viz_agent's ruler still marks JointTrajectory verified (all four gates pass, 1.87 MB);</li>
        <li>the <code>robot=</code> recordings read back identical before and after, and robot_msgs has 1237 tests passing.</li>
      </ul></li>
    <li><b>V2</b> (traj_compare <code>6049780</code>, the tests committed failing first, then <code>15e9f4b</code>):
      <ul>
        <li>14 new tests, checked against an arm with closed-form kinematics;</li>
        <li>842 tests pass, and CI is green on Python 3.11, 3.12 and 3.13;</li>
        <li>the real viewer logs no errors, and the numbers match the prototype exactly.</li>
      </ul></li>
    <li><b>Live page:</b> <code>/</code> returns 200; an .rrd returns 200 with <code>access-control-allow-origin: *</code>;
    both site CI runs are green; the hosted viewer loaded the .rrd from github.io. I re-checked all of this myself.</li>
    <li><b>The feedback trigger <code>fb-0a86ed9d</code></b> ("keep V1 for a normal trajectory") is claimed against the V1 spec.</li>
  </ul>
  <h3>Decided for you</h3>
  <ul>
    <li><b>V2 for comparisons.</b> Your answer only covered lone trajectories, and V2 was the recommendation. Undo:
    <code>git -C /home/bam/traj_compare revert 15e9f4b</code>.</li>
    <li><b>V2 is opt-in, through <code>RobotDisplay.urdf</code>,</b> so nothing already published changes until a consumer sets it.
    Your UR workbench is the first to set it.</li>
    <li><b>No URDF → V1's tabs</b> instead of the old view, and <code>viz.show</code> now takes <code>robot=</code> and
    <code>ghosts=</code> the way robot_msgs does. Undo: <code>git -C /home/bam/robot_msg_viz revert 1340d44</code>.</li>
    <li><b>No ghost arms in the gallery sample:</b> 1.87 MB without them against 2.49 MB with, and the ruler's limit is 2 MB.</li>
    <li><b>robot_dashboards goes to Cloudflare behind Access, not github.io,</b> following your 11-25 rule for private repos.
    The public robot_msg_viz page leaves out its 7 use cases for the same reason.</li>
    <li><b>Stills and hero videos are committed</b> (2.7 MB), and CI builds the pages and recordings from code.
    A test fails if a supported type has no still.</li>
    <li><b>GitHub Pages is on</b> for robot_msg_viz, with the repo homepage set to it. Undo:
    <code>gh api -X DELETE repos/zacharyyamaoka/robot_msg_viz/pages</code>.</li>
    <li><b>The magnifier's cap is 35°, not tighter.</b> At ×50 the tour's worst joint already sits 31° off, and a tighter
    cap would have dropped your pick's own ×50 to ×20. Undo: change <code>MAX_MAGNIFIED_DEG</code> in
    <code>traj_compare/src/traj_compare/rerun_target_run.py</code>.</li>
    <li><b>traj_compare's 3.11 memory gate is fixed in product code, not in its test</b> (<code>e413020</code>).
      <ul>
        <li><b>What went wrong:</b> the new tests pushed Python 3.11's interned-string table to the point where a single
        3.7 MB resize landed inside the traced window, which read as 177 KiB per run. That was a one-off allocation,
        not a leak.</li>
        <li><b>The first fix was ruled out.</b> It edited the gate's own test (<code>cb42943</code>). The Agent Activity
        agent rejected it, and rightly: it hid memory held only on the first write. That commit is reverted
        (<code>ebd4af7</code>), and the test is byte-identical to before.</li>
        <li><b>The fix that landed</b> builds a reuse write's per-run paths with <code>os.path</code>, so
        <code>pathlib</code> no longer interns three strings per run. The output is unchanged.</li>
        <li><b>Proof:</b> CI is green on 3.11, 3.12 and 3.13. The unedited test still goes red on both injected leaks:
        420 KiB held per run, and memory held only on the first write.</li>
        <li><b>Undo:</b> <code>git -C /home/bam/traj_compare revert e413020</code>.</li>
      </ul></li>
    <li><b>The UR workbench opts into V2</b> (arm_motion_bench <code>7d51971</code>). Undo:
    <code>git -C /home/bam/arm_motion_bench revert 7d51971</code>.
      <ul>
        <li>The next live build rewrites every run file once, because <code>dashboard_models.py</code> is part of
        each run's digest. Only the digest field changes, and on BAM FB that build takes about 400 s.</li>
        <li>arm_motion_bench's dashboard parity fixture was already red before this change, from traj_compare
        <code>a4c3c79</code> and <code>e7e6c57</code>. It's regenerated in its own commit, <code>30a2044</code>:
        only <code>index.html</code> and <code>coverage/index.html</code> moved, and every run file is byte-identical.</li>
        <li>A rig job was running, which normally makes those tests skip themselves silently. They ran anyway with a
        test-only stub, following your 11-23 "never throttle for the arm" rule.</li>
      </ul></li>
  </ul>
  <h3>Needs you</h3>
  <ul>
    <li><b>robot_dashboards hosting, pick one.</b>
      <ul>
        <li><b>(a) Make the repo public (my pick):</b> it's MIT-licensed, it's a plugin of the public robot_msg_viz,
        and you asked for github.io. The page then moves there with a one-line README switch, and robot_msg_viz's
        Use cases tab can list it again.</li>
        <li><b>(b) Keep it private:</b> run the Cloudflare setup once with your API token. It also turns on every
        private agent page that's waiting on the same setup:
        {cmd("python3 /home/bam/blackbox_agent/tools/cloudflare_pages.py setup --allow-email <your address>")}
        then {cmd("python3 /home/bam/robot_dashboards/tools/set_cloudflare_secrets.py")}</li>
        <li><b>If you say nothing,</b> it stays private and the page runs locally only.</li>
      </ul></li>
  </ul>
  <h3>Left, not blocked</h3>
  <ul>
    <li><b>robot_msgs 0.3.1 isn't on PyPI,</b> so CI pins the GitHub commit <code>e5f375d</code>. Publishing a
    release is your call.</li>
    <li><b>The pick round's splice ideas aren't built:</b> V4's events log as a tab and V3's limit tint as a toggle.</li>
  </ul>
  <h3>Deliberately not done</h3>
  <ul>
    <li>No audit, no Cloudflare credentials touched, and robot_dashboards wasn't made public.</li>
    <li>The Trajectory Tracking Agent's site wasn't rebuilt: none of its output changes.</li>
    <li>Other sessions' uncommitted <code>uv.lock</code> edits were left alone.</li>
  </ul>
</section>
"""

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rerun Pick and Landing Pages</title>
<style>
:root {{ --bg:#ffffff; --fg:#1b1d21; --muted:#5d636e; --line:#e3e5e9; --card:#f6f7f9; --accent:#c2185b; --code:#eef0f3; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0f1115; --fg:#e7e9ee; --muted:#9aa1ad;
  --line:#262a31; --card:#171a20; --accent:#f06292; --code:#1d2129; }} }}
:root[data-theme="dark"] {{ --bg:#0f1115; --fg:#e7e9ee; --muted:#9aa1ad; --line:#262a31; --card:#171a20; --accent:#f06292; --code:#1d2129; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1000px; margin:0 auto; padding:32px 16px 64px; }}
.eyebrow {{ color:var(--muted); font-size:13px; letter-spacing:.04em; text-transform:uppercase; margin:0 0 6px; }}
h1 {{ font-size:28px; line-height:1.25; margin:0 0 12px; }}
h2 {{ font-size:21px; margin:44px 0 10px; padding-top:20px; border-top:1px solid var(--line); }}
h3 {{ font-size:15px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin:22px 0 6px; }}
.verdict {{ background:var(--card); border-left:3px solid var(--accent); padding:12px 14px; border-radius:6px; }}
a {{ color:var(--accent); }}
code {{ background:var(--code); padding:1px 5px; border-radius:4px; font-size:.9em; overflow-wrap:anywhere; }}
pre {{ background:var(--code); padding:10px 12px; border-radius:6px; overflow-x:auto; margin:6px 0; }}
pre code {{ background:none; padding:0; }}
figure {{ margin:14px 0; }}
figure img, figure video {{ width:100%; height:auto; display:block; border:1px solid var(--line); border-radius:6px; background:#000; }}
figcaption {{ color:var(--muted); font-size:13.5px; margin-top:5px; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
@media (max-width:720px) {{ .grid2 {{ grid-template-columns:1fr; }} h1 {{ font-size:23px; }} }}
.decisions li {{ margin:4px 0; }}
</style></head><body><main>{body}</main></body></html>"""
    OUT.write_text(page)
    print(OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
