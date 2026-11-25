"""Build the 2026-11-24 report: one self-contained HTML page from the real stills, video and ledger.

    env -u VIRTUAL_ENV uv run --directory /home/bam/robot_msg_viz python docs/build_report.py

Reads robot_viz_agent's first full evaluation (its ledger row and stills), the recorded landing-page video and the
site screenshots under reports/media/robot-msg-viz-2026-11-24/, and writes
reports/media/robot-msg-viz-2026-11-24.html (media inline, so it lives in the gitignored reports/media/).
"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
MEDIA = REPO / "reports" / "media" / "robot-msg-viz-2026-11-24"
OUT = REPO / "reports" / "media" / "robot-msg-viz-2026-11-24.html"
AGENT = Path("/home/bam/robot_viz_agent")
EVAL = AGENT / "outputs" / "eval" / "2026-11-24-first"
CALIB = Path("/tmp/claude-1000/-home-bam-zach-brain--claude-worktrees-robot-msg-rerun-viz-4be20c/"
             "7e7b6ad1-d57a-4375-9897-ffd8dd9ab0f7/scratchpad/calib/pose-pair.png")  # fmt: skip

DASHBOARDS = [
    ("object_detection", "Object detection", "What did the detector see in each frame, and how sure was it over time?"),
    ("detection_3d", "3D detection", "Where are the objects in 3D, and what are they?"),
    ("rgbd_camera", "RGB-D camera", "What does the RGB-D camera see, in 2D and in 3D?"),
    ("grasping", "Grasping", "Which grasps did the planner propose, which is best, and is it feasible?"),
    ("pick_and_place", "Pick and place", "Where is the arm in the cycle, and did each phase go as planned?"),
    ("trajectory_generation", "Trajectory generation", "Did the generator turn these waypoints into a smooth trajectory inside the limits?"),
    ("robot_state", "Robot state", "What is the robot doing right now, and is anything unhealthy?"),
]  # fmt: skip


def webp(image: Image.Image, width: int, quality: int = 72) -> str:
    if image.width > width:
        image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "WEBP", quality=quality, method=6)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def still(path: Path, width: int = 960, crop_menu: bool = True) -> str:
    image = Image.open(path)
    if crop_menu:
        image = image.crop((0, 36, image.width, image.height))
    return webp(image, width)


def contact_sheet(ledger: dict) -> str:
    keys = [k for k, e in ledger["entries"].items() if e["status"] == "verified" and "/" in k]
    cols, w, h, label = 7, 260, 163, 20
    rows = (len(keys) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * (h + label)), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for i, key in enumerate(keys):
        png = EVAL / "shots" / f"{key.replace('/', '-')}.png"
        x, y = (i % cols) * w, (i // cols) * (h + label)
        draw.text((x + 6, y + 3), key, fill=(235, 200, 110), font=font)
        if png.exists():
            im = Image.open(png).convert("RGB").crop((0, 36, 1280, 800)).resize((w - 4, h - 4), Image.LANCZOS)
            sheet.paste(im, (x + 2, y + label + 2))
    return webp(sheet, 1820, quality=70)


def esc(text: str) -> str:
    return html.escape(text, quote=True)


ARCH = """<svg viewBox="0 0 980 300" role="img" aria-label="adapters to robot_msgs to dashboards to viewers" class="arch">
<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>
<g font-family="Inter, system-ui, sans-serif" font-size="13" fill="currentColor">
<rect x="10" y="40" width="190" height="220" rx="10" class="box"/><text x="105" y="64" text-anchor="middle" font-weight="700">Adapters (in)</text>
<text x="28" y="96">robot_msgs ✓ (identity)</text><text x="28" y="122">ROS 2 messages ✓</text><text x="28" y="148">MCAP / rosbag2 bags ✓</text>
<text x="28" y="174" class="muted">yours: adapters.register()</text><text x="28" y="200" class="muted">LeRobot (roadmap)</text>
<rect x="250" y="110" width="150" height="80" rx="10" class="core"/><text x="325" y="145" text-anchor="middle" font-weight="700">robot_msgs</text><text x="325" y="167" text-anchor="middle" class="muted">68 typed messages</text>
<rect x="450" y="20" width="250" height="120" rx="10" class="box"/><text x="575" y="44" text-anchor="middle" font-weight="700">robot_msg_viz</text>
<text x="466" y="70">a stock view per type: 53 of 68</text><text x="466" y="94">show · save · stream · blueprint</text><text x="466" y="118" class="muted">the landing page (Types · Use cases)</text>
<rect x="450" y="160" width="250" height="120" rx="10" class="box"/><text x="575" y="184" text-anchor="middle" font-weight="700">robot_dashboards</text>
<text x="466" y="210">7 use cases, a question each</text><text x="466" y="234">one renderer per viewer</text><text x="466" y="258" class="muted">plugs into the gallery (entry point)</text>
<rect x="750" y="40" width="220" height="220" rx="10" class="box"/><text x="860" y="64" text-anchor="middle" font-weight="700">Viewers (out)</text>
<text x="768" y="96">Rerun 0.40 ✓</text><text x="768" y="122" class="muted">Foxglove (roadmap)</text><text x="768" y="148" class="muted">Viser (roadmap)</text><text x="768" y="174" class="muted">static HTML (roadmap)</text>
<text x="768" y="214" class="muted">robot_viz_agent scores</text><text x="768" y="234" class="muted">every entry in Rerun</text>
<line x1="200" y1="150" x2="248" y2="150" stroke="currentColor" stroke-width="2" marker-end="url(#ah)"/>
<line x1="400" y1="140" x2="448" y2="90" stroke="currentColor" stroke-width="2" marker-end="url(#ah)"/>
<line x1="400" y1="160" x2="448" y2="215" stroke="currentColor" stroke-width="2" marker-end="url(#ah)"/>
<line x1="575" y1="140" x2="575" y2="158" stroke="currentColor" stroke-width="1.5" stroke-dasharray="4 3"/>
<line x1="700" y1="80" x2="748" y2="120" stroke="currentColor" stroke-width="2" marker-end="url(#ah)"/>
<line x1="700" y1="220" x2="748" y2="180" stroke="currentColor" stroke-width="2" marker-end="url(#ah)"/>
</g></svg>"""


def main() -> None:
    ledger = [json.loads(line) for line in (AGENT / "ledger.jsonl").read_text().splitlines() if line.strip()][-1]
    video = "data:video/mp4;base64," + base64.b64encode((MEDIA / "hero.mp4").read_bytes()).decode()
    poster = still(MEDIA / "site" / "use-case-grasping.png", 960, crop_menu=False)
    unsupported = [k for k, e in ledger["entries"].items() if e["status"] == "unsupported"]

    dash_cards = "".join(
        f'<figure class="card"><img src="{still(EVAL / "shots" / f"{key}.png")}" alt="{esc(title)} dashboard in the Rerun web viewer">'
        f'<figcaption><b>{esc(title)}</b> · <span class="q">{esc(q)}</span> · '
        f'<span class="muted">.rrd {ledger["entries"][key]["rrd_bytes"] / 1e6:.2f} MB</span></figcaption></figure>'
        for key, title, q in DASHBOARDS
    )  # fmt: skip
    site = still(MEDIA / "site-sheet.png", 1280, crop_menu=False)
    site2 = still(MEDIA / "site-sheet2.png", 1400, crop_menu=False)
    pose_pair = still(CALIB, 1280, crop_menu=False) if CALIB.exists() else ""
    sheet = contact_sheet(ledger)
    gates = ledger["failed_gates"]

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>robot_msg_viz · robot_dashboards · robot_viz_agent — Nov 24 2026</title>
<style>
:root {{ --bg:#fafaf9; --fg:#1c1c1e; --muted:#6b6b70; --card:#fff; --border:#e2e2e5; --accent:#1f6feb; --ok:#1a7f37; --warn:#9a6700; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#121214; --fg:#ececef; --muted:#9a9aa2; --card:#1b1b1f; --border:#2c2c32; --accent:#58a6ff; --ok:#3fb950; --warn:#d29922; }} }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 Inter, system-ui, sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:28px 16px 80px; }}
h1 {{ font-size:26px; margin:0 0 6px; }} h2 {{ font-size:19px; margin:42px 0 10px; border-bottom:1px solid var(--border); padding-bottom:6px; }}
.verdict {{ font-size:17px; background:var(--card); border:1px solid var(--border); border-left:4px solid var(--ok); padding:14px 16px; border-radius:10px; }}
.muted {{ color:var(--muted); }} code {{ font:13px ui-monospace, Menlo, monospace; background:var(--card); border:1px solid var(--border); padding:1px 5px; border-radius:5px; }}
pre {{ background:var(--card); border:1px solid var(--border); padding:12px 14px; border-radius:10px; overflow:auto; font:13px/1.5 ui-monospace, Menlo, monospace; }}
video, img {{ max-width:100%; border-radius:10px; border:1px solid var(--border); display:block; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(340px, 1fr)); gap:16px; }}
.card {{ margin:0; background:var(--card); border:1px solid var(--border); border-radius:12px; padding:8px; }}
.card figcaption {{ padding:8px 4px 2px; font-size:13.5px; }} .q {{ font-style:italic; }}
.three {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:14px; }}
.three > div {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:14px; }}
.kpi {{ font-size:30px; font-weight:700; }} table {{ border-collapse:collapse; width:100%; font-size:14px; }}
td, th {{ border-bottom:1px solid var(--border); padding:6px 8px; text-align:left; vertical-align:top; }}
.arch {{ width:100%; height:auto; color:var(--fg); }} .arch .box {{ fill:var(--card); stroke:var(--border); }} .arch .core {{ fill:none; stroke:var(--accent); stroke-width:2; }} .arch .muted {{ fill:var(--muted); }}
ul.tight li {{ margin:4px 0; }} .ok {{ color:var(--ok); font-weight:600; }}
</style></head><body><main>
<p class="muted">Nov 24 2026 · Daily Note "Robot Msg Rerun viewer" · Claude Code (Opus 5.5)</p>
<h1>A stock Rerun viewer for every robot_msgs type, and dashboards for the common jobs</h1>
<p class="verdict"><b>Built, verified and pushed (private).</b> {ledger["verified_types"]} of {ledger["types_total"]} robot_msgs types open in a stock Rerun view, and all {ledger["verified_use_cases"]} use-case dashboards are built. Every entry has a screenshot, sample data and Open in browser / Open on desktop on one landing page. robot_viz_agent's first full evaluation failed <b>no gate</b> on any entry. Making the repos public is the one step left for you.</p>

<video src="{video}" poster="{poster}" controls muted loop autoplay playsinline width="960"></video>
<p class="muted">The real landing page, served locally and driven in headless Chrome (14 s, the viewer's ~10 s wasm loads cut):
Types tab → Supported filter → PointCloud → Open in browser (the real Rerun 0.40 web viewer, orbited) → Use cases → Grasping → Open in browser.
GIF fallback beside this file: <code>robot-msg-viz-2026-11-24/hero.gif</code>.</p>

<h2>What you have</h2>
<div class="three">
<div><b>robot_msg_viz</b> · <span class="muted">/home/bam/robot_msg_viz</span><div class="kpi">{ledger["verified_types"]} / {ledger["types_total"]}</div>types with a stock view. <code>viz.show(msg)</code>, <code>viz.save(path, msg)</code>, <code>with viz.stream("arm") as live: live.log(msg)</code>, <code>viz.blueprint(...)</code>. It builds on robot_msgs' own logger and also holds the landing page.</div>
<div><b>robot_dashboards</b> · <span class="muted">/home/bam/robot_dashboards</span><div class="kpi">{ledger["verified_use_cases"]} / {ledger["use_cases_total"]}</div>use cases, each built around one question. Layers: adapters (robot_msgs, ROS 2 / MCAP, yours) → robot_msgs → dashboard → viewer (Rerun today; Foxglove, Viser and static HTML on the roadmap).</div>
<div><b>robot_viz_agent</b> · <span class="muted">/home/bam/robot_viz_agent</span><div class="kpi">{ledger["verified_types_pct"]:.1f} %</div>KPI <code>verified_types</code> under frozen ruler v1. G1 contract, G2 renders in the real viewer, G3 not blank, G4 size. Failed gates: {sum(gates.values())}. The first evaluation took {ledger["wall_time_s"] / 60:.0f} min.</div>
</div>

<h2>How it fits together</h2>
{ARCH}
<p class="muted">Your Nov 24 follow-up, in your words: "right now support rerun … foxglove etc. on the roadmap … adapters … an adapter for robot_msgs … anyone can easily extend by adding their own adapter or by converting their messages to robot_msgs."</p>

<h2>The seven use-case dashboards (real Rerun 0.40 web viewer, headless, at the recording's end)</h2>
<div class="grid">{dash_cards}</div>

<h2>Every type with a stock view ({ledger["verified_types"]})</h2>
<img src="{sheet}" alt="contact sheet of the {ledger['verified_types']} type views">
<p><b>Listed but without a view ({len(unsupported)}):</b> {", ".join(f"<code>{esc(k)}</code>" for k in unsupported)}.
These are configuration or parameters with nothing of their own to draw. Each lane said why: Header is metadata, the step conditions appear inside JointCommandSequence's lanes, the tolerances are bands traj_compare draws, and MeshResource needs its mesh file. They show greyed out on the site, never hidden.</p>

<h2>The landing page</h2>
<img src="{site}" alt="landing page: Types tab, Supported filter, Use cases tab, a use-case page">
<img src="{site2}" alt="phone width, and Open on desktop copying the command" style="margin-top:12px">
<ul class="tight">
<li><b>Types tab:</b> all 68, an All / Supported filter (remembered), search, grouped by robot_msgs domain. Unsupported types are greyed and not clickable.</li>
<li><b>Use cases tab:</b> cards by category, with chips for the types each one draws. Each use-case page lists its <b>viewers</b> (Rerun; Foxglove / Viser / static HTML as roadmap) and the <b>adapters</b> it takes (robot_msgs, ros2).</li>
<li><b>Open in browser</b> plays the sample in the real Rerun web viewer inline. Served locally it uses the bundled viewer; hosted it uses <code>app.rerun.io/version/0.40.0</code> with <code>persist=0</code>.</li>
<li><b>Open on desktop</b> copies <code>uvx --from rerun-sdk==0.40.0 rerun &lt;url&gt;</code>. Rerun registers no URL scheme, so a web page cannot start the native viewer itself (checked in the 0.40 binary and its docs).</li>
<li>No horizontal scroll at 390 px (measured: scrollWidth = innerWidth).</li>
</ul>

<h2>The agent and its ruler</h2>
<table><tr><th>Gate</th><th>What it checks</th><th>Failed</th></tr>
<tr><td>G1 contract</td><td>the sample is the type; a real logger draws it (not the text fallback); the .rrd passes <code>rerun rrd verify</code>; it carries a default blueprint; every view in it has data (read back by <code>rerun rrd print</code>); streaming lands on the timeline</td><td>{gates["G1_contract"]}</td></tr>
<tr><td>G2 renders</td><td>the real Rerun 0.40 web viewer loads it headless with no panic and no viewer error</td><td>{gates["G2_renders"]}</td></tr>
<tr><td>G3 not blank</td><td>the share of the view the data changes, against the same recording shot with every data entity dropped, is at least 0.05 %</td><td>{gates["G3_not_blank"]}</td></tr>
<tr><td>G4 size</td><td>.rrd under 2 MB (a type) or 8 MB (a use case)</td><td>{gates["G4_size"]}</td></tr></table>
<p><b>A ruler fix made before the first evaluation.</b> G3 first used the still's own pixel spread. In the dark theme an empty 3D grid scored 8.3 and an empty plot 10.8, while a real Pose scored 9.6, so spread could not tell blank from drawn. The oracle now compares each still with its data-less twin. The viewer renders deterministically (0.0 % difference shot to shot), and the sparsest real view, a lone Pose, changes 0.22 %.</p>
{f'<img src="{pose_pair}" alt="a Pose still beside its data-less twin">' if pose_pair else ''}

<h2>Use it</h2>
<pre>import robot_msg_viz as viz
viz.show(pose)                            # the stock view, desktop viewer
viz.show(target=planned, real=measured)   # names become entities: an overlay
viz.save("cloud.rrd", cloud)              # blueprint inside: rerun cloud.rrd
with viz.stream("arm") as live:           # live
    live.log(joint_state)

import robot_dashboards as dash
from robot_dashboards.adapters import ros2
dash.show("grasping", cloud=cloud, grasps=candidates)
dash.show("robot_state", ticks=ros2.bag_ticks("run/", joint_state="/joint_states"))   # straight from a ROS 2 bag</pre>

<h2>Prior art (vault note)</h2>
<ul class="tight">
<li><b>Stock per-type views exist; stock use-case dashboards do not.</b> Rerun's MCAP / ROS 2 importer maps ~20 ROS types, Foxglove lists the schema each panel takes, and RViz has each display declare its message type. None of them ships a per-type card with sample data, a screenshot and an open link, and none covers detection, grasping or trajectory use cases.</li>
<li><b>Your earlier query was Oct 3</b> ("find prior art … at least 20 good examples"), which became the Agent Work Dashboard note. You stated the same "robots are all similar, stock dashboards" thesis on Oct 28, but that went to data-ui's robotics templates, not to Rerun.</li>
<li><b>154 of 158 technical claims confirmed</b> by a skeptic agent. Note: <code>Projects/robot_msgs/Stock Robot Viewers and Dashboards — Prior Art (2026-11-24).md</code>.</li>
</ul>

<h2>Decision surface</h2>
<p><b>Done and proved</b></p>
<ul class="tight">
<li>Tests: robot_msg_viz 229 passed · robot_dashboards 30 passed, 5 skipped (static dashboards skip the stream test) · robot_viz_agent 33 passed.</li>
<li>The first full evaluation is ledger row 1: {ledger["verified_types"]} / {ledger["types_total"]} types, {ledger["verified_use_cases"]} / {ledger["use_cases_total"]} use cases, 0 failed gates.</li>
<li>The ROS 2 adapter round-trips a real MCAP bag written by robot_msgs into a dashboard. A registered custom adapter feeds save and stream.</li>
<li>All three repos are pushed to GitHub, private, in sync with local <code>main</code>.</li>
</ul>
<p><b>Decided for you</b> (each undoable)</p>
<ul class="tight">
<li><b>Two repos, as your note split them</b> (robot_msg_viz bare-bones per type, robot_dashboards per use case). The landing page lives in robot_msg_viz and finds use cases through a Python entry point, so neither package imports the other's internals. Undo: move <code>site/</code> or merge the repos later.</li>
<li><b>Built on robot_msgs' own Rerun logger, never a copy.</b> robot_msgs is not edited (a peer froze its output today with a golden test). Where a type needed more (JointState as plots in degrees, a PointCloud coloured by height, a CameraInfo frustum), robot_msg_viz overrides it in its own dispatch table. So <code>msg.show()</code> and <code>viz.show(msg)</code> differ for those types until you fold the overrides into robot_msgs. Undo: move a view's code into <code>robot_msgs/viz/rerun.py</code>.</li>
<li><b>Pinned rerun-sdk 0.40.0</b>, robot_msgs' and traj_compare's pin (latest is 0.43.2). Undo: bump all three together.</li>
<li><b>Stills in the dark theme</b>, matching robotmsgs.com and readable label contrast. Undo: <code>SHOOT_THEME=light</code>.</li>
<li><b>Named the agent <code>robot_viz_agent</code></b>, one agent covering both packages. Undo: <code>gh repo rename</code>.</li>
</ul>
<p><b>Needs you</b></p>
<ul class="tight">
<li><b>Make the three repos public</b>, as your note asked. A first public push can't be undone, so I left it for you. Recommended: flip all three, then set each repo's Pages source to "GitHub Actions" so robot_viz_agent's live page and a hosted landing page can deploy. Command: <code>gh repo edit zacharyyamaoka/robot_msg_viz --visibility public --accept-visibility-change-consequences</code> (likewise for robot_dashboards and robot_viz_agent).</li>
</ul>
<p><b>Left / next</b></p>
<ul class="tight">
<li><b>Hosting:</b> deploy the landing page (GitHub Pages once public; Cloudflare can't host Rerun's 54 MB wasm, but we link the hosted viewer, so that's fine).</li>
<li><b>Foxglove viewer</b> (roadmap step 1: robot_msgs.ros.write_bag already writes an MCAP Foxglove opens, plus one layout JSON per dashboard). Lichtblick may be the better open target.</li>
<li><b>Unify</b> <code>msg.show()</code> with <code>viz.show(msg)</code> (above).</li>
<li><b>robot-sdk index:</b> add robot_msg_viz and robot_dashboards to the viewers layer. Both carry <code>robot-sdk.toml</code>. I did not touch robot_sdk because a peer was committing there all evening.</li>
<li><b>A one-click Open on desktop when served locally</b> (the page asks the local server to start the native viewer, as traj_compare.serve does). Not built, because testing it would open a native Rerun window on your screen.</li>
</ul>
<p><b>Deliberately not done</b></p>
<ul class="tight">
<li>No edit to robot_msgs or traj_compare: peers were working in both all session. A robot_msgs bug found here (epoch stamps broke goal-phase checks) went to the peer, who fixed it in all five places (<code>31218b0</code>).</li>
<li>No PyPI release and no public deploy: both reach other people.</li>
</ul>
</main></body></html>"""
    OUT.write_text(page)
    print(OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
