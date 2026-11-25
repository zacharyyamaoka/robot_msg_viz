#!/usr/bin/env node
// The report's hero video: the landing page driven in headless Chrome, recorded, with the moments printed so the
// builder can cut the viewer's wasm load out. The site must already be served (python -m robot_msg_viz.site serve).
//
//   node docs/record_hero.cjs http://127.0.0.1:4493/ OUT_DIR
//
// Prints one JSON line: {"video": path, "marks": {name: seconds since recording start}}.

const fs = require('fs');
const path = require('path');
const { chromium } = require(process.env.PLAYWRIGHT || '/home/bam/aruco-studio/web/node_modules/playwright');

const base = process.argv[2];
const out = path.resolve(process.argv[3]);
fs.mkdirSync(out, { recursive: true });

(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROME || '/usr/bin/google-chrome', headless: true,
    args: ['--enable-webgl', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const context = await browser.newContext({ viewport: { width: 1280, height: 800 }, colorScheme: 'dark',
    recordVideo: { dir: out, size: { width: 1280, height: 800 } } });
  const page = await context.newPage();
  const t0 = Date.now();
  const marks = {};
  const mark = (name) => { marks[name] = (Date.now() - t0) / 1000; };
  const viewerReady = async () => {
    // the iframe's canvas has painted once the viewer's own handle reports a recording
    for (let i = 0; i < 90; i += 1) {
      const frame = page.frames().find((f) => f !== page.mainFrame());
      if (frame) {
        const ok = await frame.evaluate(() => !!(window._handle && window._handle.get_active_recording_id())).catch(() => false);
        if (ok) return true;
      }
      await page.waitForTimeout(500);
    }
    return false;
  };
  try {
    await page.goto(`${base}index.html?renderer=webgl#types`);
    await page.waitForTimeout(1800); mark('index');
    await page.getByRole('button', { name: /^Supported/ }).first().click().catch(() => {});
    await page.waitForTimeout(1500); mark('supported');
    await page.mouse.wheel(0, 700); await page.waitForTimeout(1200);
    await page.mouse.wheel(0, -700); await page.waitForTimeout(600);
    await page.click('a[href*="sensor-PointCloud"]'); await page.waitForTimeout(1600); mark('type_page');
    // WHY the screenshot, not the Open button below it: clicking lower down scrolled the viewer half off-screen
    await page.evaluate(() => window.scrollTo(0, 0)); await page.click('.stage-shot'); mark('open_type');
    await viewerReady(); await page.waitForTimeout(2500); mark('type_loaded');
    await page.mouse.move(420, 460); await page.mouse.down(); await page.mouse.move(560, 420, { steps: 25 });
    await page.mouse.up(); await page.waitForTimeout(1500); mark('type_orbited');
    await page.goto(`${base}index.html?renderer=webgl#use-cases`); await page.waitForTimeout(1800); mark('use_cases');
    await page.click('a[href*="use-cases/grasping"]'); await page.waitForTimeout(1600); mark('use_case_page');
    await page.evaluate(() => window.scrollTo(0, 0)); await page.click('.stage-shot'); mark('open_use_case');
    await viewerReady(); await page.waitForTimeout(3000); mark('use_case_loaded');
  } finally {
    const video = page.video();
    await context.close();
    await browser.close();
    console.log(JSON.stringify({ video: video ? await video.path() : null, marks }));
  }
})().catch((e) => { console.error(e); process.exit(1); });
