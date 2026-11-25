#!/usr/bin/env node
// Stills of .rrd files in the REAL Rerun 0.40 web viewer, headless, many per run.
//
//   node tools/shoot.cjs <jobs.json>      jobs: [{"rrd": "/abs/x.rrd", "out": "/abs/x.png", "time": "end" | seconds}]
//
// Starts its own `rerun --serve-web` and a CORS file server on free ports, one headless Chrome for every job,
// and stops all three at the end. Prints one JSON line per job: {out, ok, playhead_s, panicked, errors}.
// Grown from traj_compare's round tool (docs/proposals/rerun-joint-trajectory-2026-11-24/shoot.cjs): the playhead
// is set through the viewer's own WebHandle (window._handle.set_time_for_timeline, nanoseconds on a duration
// timeline) because URL time params and arrow keys do not land (measured there, 2026-11-24). A recording with only
// static data has no timeline; its job skips the seek.
// Env: RERUN_BIN (default: this repo's .venv/bin/rerun), PLAYWRIGHT (module path), CHROME (default /usr/bin/google-chrome),
// SHOOT_WIDTH/SHOOT_HEIGHT (1280x800), SHOOT_WAIT_MS (8000: the wasm load and first frames).

const { spawn } = require('child_process');
const fs = require('fs');
const http = require('http');
const net = require('net');
const path = require('path');

const repo = path.resolve(__dirname, '..');
const { chromium } = require(process.env.PLAYWRIGHT || '/home/bam/aruco-studio/web/node_modules/playwright');
const RERUN = process.env.RERUN_BIN || path.join(repo, '.venv/bin/rerun');
const width = Number(process.env.SHOOT_WIDTH || 1280);
const height = Number(process.env.SHOOT_HEIGHT || 800);
const waitMs = Number(process.env.SHOOT_WAIT_MS || 8000);
const jobs = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));

const freePorts = async (n) => {
  const held = [];
  for (let i = 0; i < n; i += 1) held.push(await new Promise((resolve) => { const s = net.createServer(); s.listen(0, '127.0.0.1', () => resolve(s)); }));
  const ports = held.map((s) => s.address().port);
  await Promise.all(held.map((s) => new Promise((resolve) => s.close(resolve))));
  return ports;
};

(async () => {
  const [web, grpc, files] = await freePorts(3);
  const served = new Map(jobs.map((job, i) => [`/${i}/${path.basename(job.rrd)}`, job.rrd]));
  const server = http.createServer((req, res) => {
    const file = served.get(decodeURIComponent(req.url.split('?')[0]));
    if (!file || !fs.existsSync(file)) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { 'Access-Control-Allow-Origin': '*', 'Content-Type': 'application/octet-stream' });
    fs.createReadStream(file).pipe(res);
  }).listen(files, '127.0.0.1');
  const viewer = spawn(RERUN, ['--serve-web', '--web-viewer-port', String(web), '--port', String(grpc)], { stdio: 'ignore', detached: true });
  let browser;
  try {
    for (let i = 0; i < 80; i += 1) {
      const up = await new Promise((resolve) => http.get(`http://127.0.0.1:${web}/`, (r) => { r.resume(); resolve(r.statusCode === 200); }).on('error', () => resolve(false)));
      if (up) break;
      await new Promise((r) => setTimeout(r, 500));
    }
    browser = await chromium.launch({ executablePath: process.env.CHROME || '/usr/bin/google-chrome', headless: true,
      args: ['--enable-webgl', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
    for (let k = 0; k < jobs.length; k += 1) {
      const job = jobs[k];
      // WHY dark by default: the gallery site follows the OS theme and reads dark first (robotmsgs.com), and Rerun's
      // label backgrounds are low-contrast in the light theme (vision lane, 2026-11-24). SHOOT_THEME=light for light.
      const page = await browser.newPage({ viewport: { width, height }, colorScheme: process.env.SHOOT_THEME || 'dark' });
      const errors = [];
      // WHY skip api.github.com: the viewer checks GitHub for a newer release on load; a rate-limited 403 there says
      // nothing about the recording (it flagged good stills BAD for every wave-1 lane, 2026-11-24).
      const ours = (url) => !String(url || '').includes('api.github.com');
      page.on('console', (m) => { if (!ours((m.location() || {}).url)) return; if (m.type() === 'error' || /^\s*ERROR|panic/.test(m.text())) errors.push(m.text().slice(0, 300)); });
      page.on('pageerror', (e) => errors.push('pageerror ' + String(e).slice(0, 300)));
      page.on('response', (r) => { if (r.status() >= 400 && ours(r.url())) errors.push(`HTTP ${r.status()} ${r.url().slice(0, 160)}`); });
      const rrdUrl = `http://127.0.0.1:${files}/${k}/${encodeURIComponent(path.basename(job.rrd))}`;
      let result = { out: job.out, ok: false, playhead_s: null, panicked: null, errors };
      try {
        await page.goto(`http://127.0.0.1:${web}/?renderer=webgl&hide_welcome_screen=true&url=${encodeURIComponent(rrdUrl)}`, { timeout: 120000 });
        await page.waitForTimeout(job.wait_ms || waitMs);
        const state = await page.evaluate(([timeline, at]) => {
          const handle = window._handle;
          if (!handle) return { seeked: false, ns: null, panicked: null, note: 'no window._handle' };
          const recording = handle.get_active_recording_id();
          let seeked = false;
          try {
            const range = handle.get_timeline_time_range(recording, timeline);
            if (range && range.max !== undefined && range.max !== null) {
              handle.set_time_for_timeline(recording, timeline, at === 'end' ? range.max : Math.round(Number(at) * 1e9));
              seeked = true;
            }
          } catch (e) { /* static-only recording: no timeline to seek */ }
          let ns = null;
          try { ns = seeked ? handle.get_time_for_timeline(recording, timeline) : null; } catch (e) { ns = null; }
          return { seeked, ns, panicked: handle.has_panicked() };
        }, [job.timeline || 't', String(job.time ?? 'end')]);
        await page.waitForTimeout(state.seeked ? 2500 : 500);
        await page.mouse.move(width / 2, 5);
        fs.mkdirSync(path.dirname(job.out), { recursive: true });
        await page.screenshot({ path: job.out });
        result = { out: job.out, ok: !state.panicked, playhead_s: state.ns === null ? null : state.ns / 1e9, panicked: state.panicked, errors };
      } catch (e) {
        errors.push('shoot ' + String(e).slice(0, 300));
      }
      console.log(JSON.stringify(result));
      await page.close();
    }
  } finally {
    if (browser) await browser.close();
    server.close();
    try { process.kill(-viewer.pid, 'SIGTERM'); } catch (e) { /* already gone */ }
  }
})().catch((e) => { console.error(e); process.exit(1); });
