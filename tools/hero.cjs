#!/usr/bin/env node
// Record the gallery's hero video in the REAL Rerun 0.40 web viewer, headless: one segment per recording, each played,
// clicked and orbited as its plan says. tools/hero_video.py writes the plan, runs this, then cuts the viewer's loading
// out with ffmpeg (from the printed marks) and encodes the mp4 and its poster.
//
//   node tools/hero.cjs <plan.json>
//   plan: {"video_dir": "/abs", "width": 1280, "height": 800,
//          "segments": [{"rrd": "/abs/x.rrd", "seconds": 8, "play": [t0, t1] | null,
//                        "clicks": [[at_s, x, y], ...], "drags": [[at_s, x0, y0, x1, y1], ...]}]}
// Prints one JSON line: {"video": path, "segments": [{"start": s, "end": s, "errors": [...]}]} (seconds since the
// recording began). Env as tools/shoot.cjs: RERUN_BIN, PLAYWRIGHT, CHROME.
// WHY the playhead is moved through the viewer's own WebHandle (set_time_for_timeline, nanoseconds), from a timer
// inside the page: URL time params and keys do not land (tools/shoot.cjs, measured 2026-11-24), and a timer in the
// page keeps an even frame rate where one CDP round trip per frame would stutter.

const { spawn } = require('child_process');
const fs = require('fs');
const http = require('http');
const net = require('net');
const path = require('path');

const repo = path.resolve(__dirname, '..');
const { chromium } = require(process.env.PLAYWRIGHT || '/home/bam/aruco-studio/web/node_modules/playwright');
const RERUN = process.env.RERUN_BIN || path.join(repo, '.venv/bin/rerun');
const plan = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const width = plan.width || 1280;
const height = plan.height || 800;

const freePorts = async (n) => {
  const held = [];
  for (let i = 0; i < n; i += 1) held.push(await new Promise((resolve) => { const s = net.createServer(); s.listen(0, '127.0.0.1', () => resolve(s)); }));
  const ports = held.map((s) => s.address().port);
  await Promise.all(held.map((s) => new Promise((resolve) => s.close(resolve))));
  return ports;
};

(async () => {
  const [web, grpc, files] = await freePorts(3);
  const served = new Map(plan.segments.map((seg, i) => [`/${i}/${path.basename(seg.rrd)}`, seg.rrd]));
  const server = http.createServer((req, res) => {
    const file = served.get(decodeURIComponent(req.url.split('?')[0]));
    if (!file || !fs.existsSync(file)) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { 'Access-Control-Allow-Origin': '*', 'Content-Type': 'application/octet-stream' });
    fs.createReadStream(file).pipe(res);
  }).listen(files, '127.0.0.1');
  const viewer = spawn(RERUN, ['--serve-web', '--bind', '127.0.0.1', '--web-viewer-port', String(web), '--port', String(grpc)], { stdio: 'ignore', detached: true });
  let browser;
  let context;
  const out = { video: null, segments: [] };
  try {
    for (let i = 0; i < 80; i += 1) {
      const up = await new Promise((resolve) => http.get(`http://127.0.0.1:${web}/`, (r) => { r.resume(); resolve(r.statusCode === 200); }).on('error', () => resolve(false)));
      if (up) break;
      await new Promise((r) => setTimeout(r, 500));
    }
    browser = await chromium.launch({ executablePath: process.env.CHROME || '/usr/bin/google-chrome', headless: true,
      args: ['--enable-webgl', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
    context = await browser.newContext({ viewport: { width, height }, colorScheme: 'dark',
      recordVideo: { dir: plan.video_dir, size: { width, height } } });
    // WHY: the viewer asks GitHub for a newer release on load and shows a toast over the first seconds of every clip
    await context.route(/api\.github\.com/, (route) => route.fulfill({ status: 403, body: '' }));
    const page = await context.newPage();
    const t0 = Date.now();
    const now = () => (Date.now() - t0) / 1000;
    for (let k = 0; k < plan.segments.length; k += 1) {
      const seg = plan.segments[k];
      const errors = [];
      const onConsole = (m) => { if (m.type() === 'error' && !String((m.location() || {}).url).includes('api.github.com')) errors.push(m.text().slice(0, 200)); };
      page.on('console', onConsole);
      const rrdUrl = `http://127.0.0.1:${files}/${k}/${encodeURIComponent(path.basename(seg.rrd))}`;
      await page.goto(`http://127.0.0.1:${web}/?renderer=webgl&hide_welcome_screen=true&persist=0&url=${encodeURIComponent(rrdUrl)}`, { timeout: 120000 });
      for (let i = 0; i < 240; i += 1) {
        const ready = await page.evaluate(() => !!(window._handle && window._handle.get_active_recording_id())).catch(() => false);
        if (ready) break;
        await page.waitForTimeout(250);
      }
      // the first frames after the recording arrives: meshes upload, plots lay out
      if (seg.play) {
        await page.evaluate(([from]) => {
          const handle = window._handle;
          handle.set_time_for_timeline(handle.get_active_recording_id(), 't', Math.round(from * 1e9));
        }, [seg.play[0]]);
      }
      await page.mouse.move(width / 2, 4);
      await page.waitForTimeout(seg.settle_ms || 2500);
      const start = now();
      if (seg.play) {
        await page.evaluate(([from, to, speed]) => {
          const handle = window._handle;
          const recording = handle.get_active_recording_id();
          const began = performance.now();
          window.__heroTimer = setInterval(() => {
            const span = to - from;
            const t = from + (((performance.now() - began) / 1000) * speed) % span;
            try { handle.set_time_for_timeline(recording, 't', Math.round(t * 1e9)); } catch (e) { /* no timeline */ }
          }, 33);
        }, [seg.play[0], seg.play[1], seg.speed || 1.0]);
      }
      const actions = [
        ...(seg.clicks || []).map(([at, x, y]) => ({ at, run: async () => { await page.mouse.click(x, y); await page.mouse.move(width / 2, 4); } })),
        ...(seg.drags || []).map(([at, x0, y0, x1, y1, ms]) => ({ at, run: async () => {
          await page.mouse.move(x0, y0); await page.mouse.down();
          await page.mouse.move(x1, y1, { steps: Math.max(10, Math.round((ms || 1500) / 33)) }); await page.mouse.up();
          await page.mouse.move(width / 2, 4);
        } })),
      ].sort((a, b) => a.at - b.at);
      for (const action of actions) {
        const wait = start + action.at - now();
        if (wait > 0) await page.waitForTimeout(wait * 1000);
        await action.run();
      }
      const rest = start + seg.seconds - now();
      if (rest > 0) await page.waitForTimeout(rest * 1000);
      const end = now();
      await page.evaluate(() => clearInterval(window.__heroTimer)).catch(() => {});
      page.off('console', onConsole);
      out.segments.push({ start, end, errors });
    }
    const video = page.video();
    await context.close();
    context = null;
    out.video = video ? await video.path() : null;
  } finally {
    if (context) await context.close();
    if (browser) await browser.close();
    server.close();
    try { process.kill(-viewer.pid, 'SIGTERM'); } catch (e) { /* already gone */ }
  }
  console.log(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
