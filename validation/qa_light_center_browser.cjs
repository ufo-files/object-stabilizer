// Real-file end-to-end diagnostic. Whole-star acceptance remains an independent,
// strict final gate, even when light-center alignment and playback checks pass.
const path = require('node:path');
const ROOT = path.resolve(__dirname, '..');
const { chromium } = require(path.resolve(ROOT, '../homepage-redesign/node_modules/playwright'));
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { execFileSync } = require('node:child_process');

(async () => {
  const browser = await chromium.launch({ headless: !process.argv.includes('--headed') });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1800 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:8080/');
    const health = await (await page.request.get('http://127.0.0.1:8080/api/health')).json();
    assert.equal(health.engine, 'auto-scene-1');
    await page.locator('#mode').selectOption('light');
    await page.locator('#file').setInputFiles('/Volumes/UFO Files Archive 2/ir/4E1A78FF-32B3-48D4-9399-7E77B2A45BB1.mp4');
    await page.locator('#selection').waitFor({ state: 'visible', timeout: 60000 });
    const c = await page.locator('#canvas').boundingBox();
    await page.mouse.move(c.x + 1264 / 1920 * c.width, c.y + 234 / 1080 * c.height);
    await page.mouse.down();
    await page.mouse.move(c.x + 1328 / 1920 * c.width, c.y + 298 / 1080 * c.height, { steps: 10 });
    await page.mouse.up();
    await page.waitForFunction(() => document.getElementById('coordinates').textContent.startsWith('Light center:'), {}, { timeout: 30000 });
    const submitted = page.waitForResponse(r => r.url().endsWith('/api/stabilize') && r.request().method() === 'POST');
    await page.locator('#run').click();
    const id = (await (await submitted).json()).id;
    await page.locator('#result').waitFor({ state: 'visible', timeout: 180000 });
    const job = await (await page.request.get('http://127.0.0.1:8080/api/jobs/' + id)).json();
    const report = await (await page.request.get('http://127.0.0.1:8080' + job.result.report)).json();
    assert.equal(report.engine, 'light-center-1');
    assert.equal(report.frames, 349);
    assert.equal(report.output_frames, 349);
    const destination = path.join(ROOT, 'validation/artifacts/light-center-browser.mp4');
    const downloading = page.waitForEvent('download');
    await page.locator('#download').click();
    await (await downloading).saveAs(destination);
    fs.writeFileSync(path.join(ROOT, 'validation/artifacts/light-center-browser-report.json'), JSON.stringify(report, null, 2));
    const timestamps = JSON.parse(execFileSync('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', destination])).frames.map(frame => Number(frame.best_effort_timestamp_time));
    assert.equal(timestamps.length, 349);
    await page.locator('#output').scrollIntoViewIfNeeded();
    console.log('Testing normal playback with the magnifier enabled, then measuring all 349 displayed frames.');
    const playback = await page.evaluate(async ({ window, timestamps }) => {
      const video = document.getElementById('output');
      video.muted = true; video.loop = false;
      if (video.readyState < 2) await new Promise((resolve, reject) => {
        video.addEventListener('loadeddata', resolve, { once: true });
        video.addEventListener('error', () => reject(Error('Video could not be decoded.')), { once: true });
        setTimeout(() => reject(Error('Video load timeout.')), 20000);
      });
      const before = video.getVideoPlaybackQuality();
      const ended = new Promise((resolve, reject) => {
        video.addEventListener('ended', resolve, { once: true });
        setTimeout(() => reject(Error('Playback timeout.')), 40000);
      });
      await video.play(); await ended;
      const quality = video.getVideoPlaybackQuality();
      video.pause();
      const canvas = document.createElement('canvas'); canvas.width = canvas.height = 65;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      const observations = new Map();
      function sample(index, mode) {
        ctx.drawImage(video, window[0] - 32, window[1] - 32, 65, 65, 0, 0, 65, 65);
        const pixels = ctx.getImageData(0, 0, 65, 65).data;
        const values = Array.from({ length: 4225 }, (_, p) => .2126 * pixels[p * 4] + .7152 * pixels[p * 4 + 1] + .0722 * pixels[p * 4 + 2]);
        const background = [...values].sort((a,b) => a-b)[2112];
        let highest = -Infinity, at = 0, total = 0, sx = 0, sy = 0;
        for (let pixel = 0; pixel < values.length; pixel++) {
          const luminance = values[pixel];
          if (luminance > highest) { highest = luminance; at = pixel; }
          const weight = Math.max(luminance - background, 0) ** 8;
          total += weight; sx += (pixel % 65) * weight; sy += Math.floor(pixel / 65) * weight;
        }
        observations.set(index, { frame: index, expectedTime: timestamps[index], inspectedTime: video.currentTime,
          mode, centerX: window[0] - 32 + sx / total, centerY: window[1] - 32 + sy / total,
          peakX: window[0] - 32 + at % 65, peakY: window[1] - 32 + Math.floor(at / 65), brightness: highest });
      }
      const rewound = new Promise(resolve => video.addEventListener('seeked', resolve, { once: true }));
      video.currentTime = 0; await rewound; sample(0, 'initial');
      video.playbackRate = .25;
      let callback;
      function inspect(_now, metadata) {
        const index = timestamps.reduce((best, time, candidate) => Math.abs(time - metadata.mediaTime) < Math.abs(timestamps[best] - metadata.mediaTime) ? candidate : best, 0);
        sample(index, 'quarter_speed');
        if (!video.ended) callback = video.requestVideoFrameCallback(inspect);
      }
      callback = video.requestVideoFrameCallback(inspect);
      const inspected = new Promise((resolve, reject) => {
        video.addEventListener('ended', resolve, { once: true });
        setTimeout(() => reject(Error('Inspection playback timeout.')), 70000);
      });
      await video.play(); await inspected; video.cancelVideoFrameCallback(callback); video.pause();
      for (let index = 0; index < timestamps.length; index++) if (!observations.has(index)) {
        const seeked = new Promise((resolve, reject) => {
          video.addEventListener('seeked', resolve, { once: true });
          setTimeout(() => reject(Error('Frame seek timeout: ' + index)), 5000);
        });
        video.currentTime = timestamps[index] + .008; await seeked; sample(index, 'seek');
      }
      return { duration: video.duration, width: video.videoWidth, height: video.videoHeight,
        decoded: quality.totalVideoFrames - before.totalVideoFrames,
        dropped: quality.droppedVideoFrames - before.droppedVideoFrames,
        observations: [...observations.values()].sort((a,b) => a.frame-b.frame) };
    }, { window: report.window, timestamps });
    assert.equal(playback.observations.length, 349);
    const first = playback.observations[0];
    const browserMaximumCenterDrift = Math.max(...playback.observations.map(f => Math.hypot(f.centerX - first.centerX, f.centerY - first.centerY)));
    const browserMaximumCenterStep = Math.max(...playback.observations.slice(1).map((f,i) => Math.hypot(f.centerX - playback.observations[i].centerX, f.centerY - playback.observations[i].centerY)));
    await page.screenshot({ path: path.join(ROOT, 'validation/artifacts/light-center-browser.png'), fullPage: true });
    const qa = { job: id, result: job.result, playback, browserMaximumCenterDrift, browserMaximumCenterStep,
      wholeStarAccepted: report.zero_motion_passed, errors, destination };
    fs.writeFileSync(path.join(ROOT, 'validation/artifacts/light-center-browser-qa.json'), JSON.stringify(qa, null, 2));
    console.log(JSON.stringify({ job: id, ...job.result, framesInspected: playback.observations.length,
      dropped: playback.dropped, browserMaximumCenterDrift, browserMaximumCenterStep, errors }, null, 2));
    assert.deepEqual(errors, []);
    assert.equal(playback.width, 1920); assert.equal(playback.height, 1080);
    assert.equal(playback.dropped, 0, 'Normal playback must not drop frames.');
    assert.ok(browserMaximumCenterDrift < .1 && browserMaximumCenterStep < .1, 'Browser light-center alignment exceeded 0.1 px.');
    if (!process.argv.includes('--diagnostics')) assert.equal(report.zero_motion_passed, true, 'The whole-star requirement is not met; center lock alone is insufficient.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
