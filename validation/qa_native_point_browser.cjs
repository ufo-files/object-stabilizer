// Peak diagnostics are insufficient: final acceptance also requires whole-reference QA.
const { chromium } = require('/Users/ufo-files-agent/projects/homepage-redesign/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { execFileSync } = require('node:child_process');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1800 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('http://127.0.0.1:8080/');
  assert.equal(await page.title(), 'Tripod Stabilizer');
  await page.locator('#file').setInputFiles('/Volumes/UFO Files Archive 2/ir/4E1A78FF-32B3-48D4-9399-7E77B2A45BB1.mp4');
  await page.locator('#selection').waitFor({ state: 'visible', timeout: 60000 });
  const c = await page.locator('#canvas').boundingBox();
  await page.mouse.move(c.x + 1264 / 1920 * c.width, c.y + 234 / 1080 * c.height);
  await page.mouse.down();
  await page.mouse.move(c.x + 1328 / 1920 * c.width, c.y + 298 / 1080 * c.height, { steps: 10 });
  await page.mouse.up();
  await page.waitForFunction(() => document.getElementById('coordinates').textContent.includes('(1296, 266)'), {}, { timeout: 30000 });
  const submitted = page.waitForResponse(response => response.url().endsWith('/api/stabilize') && response.request().method() === 'POST');
  await page.locator('#run').click();
  const id = (await (await submitted).json()).id;
  await page.locator('#result').waitFor({ state: 'visible', timeout: 180000 });
  const job = await (await page.request.get('http://127.0.0.1:8080/api/jobs/' + id)).json();
  const report = await (await page.request.get('http://127.0.0.1:8080' + job.result.report)).json();
  assert.equal(report.engine, 'native-peak-lock-3');
  assert.equal(report.frames, 349);
  assert.equal(report.output_frames, 349);
  assert.equal(report.peak_lock_passed, true);
  assert.equal(report.source_luma_preserved, true);
  assert.equal(report.zero_motion_passed, false);
  for (const frame of report.audit) {
    assert.equal(frame.peak_x, 1296); assert.equal(frame.peak_y, 266);
    assert.equal(frame.source_luma_preserved, true);
  }
  const destination = '/Users/ufo-files-agent/projects/object-stabilizer/validation/artifacts/native-point-browser.webm';
  const downloading = page.waitForEvent('download');
  await page.locator('#download').click();
  await (await downloading).saveAs(destination);
  fs.writeFileSync('/Users/ufo-files-agent/projects/object-stabilizer/validation/artifacts/point-lock-final-report.json', JSON.stringify(report, null, 2));
  const timestamps = JSON.parse(execFileSync('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', destination])).frames.map(frame => Number(frame.best_effort_timestamp_time));
  assert.equal(timestamps.length, 349);
  await page.locator('#output').scrollIntoViewIfNeeded();
  console.log('Export ready; testing visible playback, then inspecting all 349 browser frames at quarter speed.');
  const playback = await page.evaluate(async ({ anchor, timestamps }) => {
    const video = document.getElementById('output');
    video.muted = true; video.loop = false;
    if (video.readyState < 2) await new Promise((resolve, reject) => {
      video.addEventListener('loadeddata', resolve, { once: true });
      video.addEventListener('error', () => reject(Error('Video could not be decoded.')), { once: true });
      setTimeout(() => reject(Error('Video load timeout.')), 20000);
    });
    // Measure playback without pixel-readback instrumentation stalling the GPU.
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
      ctx.drawImage(video, anchor[0] - 32, anchor[1] - 32, 65, 65, 0, 0, 65, 65);
      const pixels = ctx.getImageData(0, 0, 65, 65).data;
      let highest = -Infinity, at = 0;
      for (let pixel = 0; pixel < 65 * 65; pixel++) {
        const luminance = .2126 * pixels[pixel * 4] + .7152 * pixels[pixel * 4 + 1] + .0722 * pixels[pixel * 4 + 2];
        if (luminance > highest) { highest = luminance; at = pixel; }
      }
      observations.set(index, { frame: index, expectedTime: timestamps[index], inspectedTime: video.currentTime,
        mode, x: anchor[0] - 32 + at % 65, y: anchor[1] - 32 + Math.floor(at / 65), brightness: highest });
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
      decoded: quality.totalVideoFrames, dropped: quality.droppedVideoFrames, observations: [...observations.values()].sort((a,b) => a.frame-b.frame) };
  }, { anchor: report.anchor, timestamps });
  assert.equal(playback.observations.length, 349);
  assert.equal(playback.width, 1920); assert.equal(playback.height, 1080);
  const browserMaximumPeakOffset = Math.max(...playback.observations.map(frame => Math.hypot(frame.x - 1296, frame.y - 266)));
  assert.deepEqual(errors, []);
  assert.equal(browserMaximumPeakOffset, 0);
  await page.screenshot({ path: '/Users/ufo-files-agent/projects/object-stabilizer/validation/artifacts/native-point-browser.png', fullPage: true });
  const qa = { job: id, result: job.result, playback, browserMaximumPeakOffset, errors, destination };
  fs.writeFileSync('/Users/ufo-files-agent/projects/object-stabilizer/validation/artifacts/native-point-browser-qa.json', JSON.stringify(qa, null, 2));
  console.log(JSON.stringify({ job: id, ...job.result, ...playback, observations: playback.observations.length, browserMaximumPeakOffset, errors }));
  await browser.close();
  assert.equal(playback.dropped, 0, 'Normal-speed playback must not drop frames.');
  assert.equal(report.zero_motion_passed, true, 'Peak lock alone does not pass whole-reference stabilization QA.');
})().catch(error => { console.error(error); process.exit(1); });
