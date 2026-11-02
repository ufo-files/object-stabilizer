import { sourcePoint, selectionBox } from './selection.js?v=auto-scene-1';

const $ = id => document.getElementById(id);
const canvas = $('canvas');
const context = canvas.getContext('2d');
const output = $('output');
const inspector = $('inspector');
const inspection = inspector.getContext('2d');
let bundleVersion = 0, currentResult, bundleUrl;
let video, image, box, start, anchor, sourceAnchor, inspectionWindow, busy = false, selectionVersion = 0;
let referenceFrame = 0, loadedFrame = 0, frameVersion = 0, frameLoading = false, scrubTimer;
const mode = () => $('mode').value;

async function request(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : 'Request failed.');
  return data;
}

function post(url, body) {
  return request(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

function point(event) {
  return sourcePoint(event.clientX, event.clientY, canvas.getBoundingClientRect(), canvas.width, canvas.height);
}

function syncFrameControls() {
  const count = video?.frame_count || 1;
  for (const id of ['referenceFrame', 'referenceFrameNumber']) {
    $(id).max = count; $(id).value = referenceFrame + 1; $(id).disabled = busy || !video;
  }
  $('frameTotal').textContent = `of ${count}`;
  $('previousFrame').disabled = busy || !video || referenceFrame <= 0;
  $('nextFrame').disabled = busy || !video || referenceFrame >= count - 1;
}

async function showFrame(index) {
  if (!video || busy || !Number.isInteger(index)) return;
  index = Math.max(0, Math.min(video.frame_count - 1, index));
  const version = ++frameVersion;
  referenceFrame = index; frameLoading = true;
  selectionVersion++; start = box = anchor = sourceAnchor = inspectionWindow = null;
  output.pause(); $('result').hidden = true; $('trackRange').hidden = true;
  syncFrameControls(); draw();
  $('status').textContent = `Loading source frame ${index + 1}…`;
  try {
    const next = new Image(); next.src = `/api/videos/${video.id}/frames/${index}`;
    await next.decode();
    if (version !== frameVersion) return;
    image = next; loadedFrame = index;
    $('status').textContent = `Source frame ${index + 1} of ${video.frame_count}. Draw the box around the moving light.`;
    $('coordinates').textContent = 'Select the light on this frame; tracking will run backward and forward.';
  } catch (error) {
    if (version === frameVersion) { referenceFrame = loadedFrame; syncFrameControls(); $('status').textContent = `Could not load source frame ${index + 1}. Still showing frame ${loadedFrame + 1}; try another frame.`; }
  } finally {
    if (version === frameVersion) { frameLoading = false; draw(); }
  }
}

$('referenceFrame').oninput = () => {
  clearTimeout(scrubTimer);
  referenceFrame = Number($('referenceFrame').value) - 1;
  frameVersion++; selectionVersion++; frameLoading = true;
  anchor = sourceAnchor = box = start = null;
  output.pause(); $('result').hidden = true; $('trackRange').hidden = true;
  syncFrameControls(); draw();
  scrubTimer = setTimeout(() => showFrame(referenceFrame), 150);
};
$('referenceFrameNumber').onchange = () => { clearTimeout(scrubTimer); showFrame(Number($('referenceFrameNumber').value) - 1); };
$('previousFrame').onclick = () => { clearTimeout(scrubTimer); showFrame(referenceFrame - 1); };
$('nextFrame').onclick = () => { clearTimeout(scrubTimer); showFrame(referenceFrame + 1); };

function draw() {
  if (!image) return;
  context.drawImage(image, 0, 0);
  context.strokeStyle = '#80ffdc';
  context.lineWidth = 2 * canvas.width / canvas.clientWidth;
  if (mode() === 'scene' && video?.analysis) {
    for (const [x, y] of video.analysis.points) {
      context.beginPath(); context.arc(x + .5, y + .5, 8, 0, 2 * Math.PI); context.stroke();
    }
  }
  if (mode() === 'light' && box) context.strokeRect(...box);
  if (mode() === 'light' && sourceAnchor) {
    const [x, y] = sourceAnchor.map(value => value + .5);
    context.beginPath();
    context.moveTo(x - 12, y); context.lineTo(x - 4, y);
    context.moveTo(x + 4, y); context.lineTo(x + 12, y);
    context.moveTo(x, y - 12); context.lineTo(x, y - 4);
    context.moveTo(x, y + 4); context.lineTo(x, y + 12);
    context.stroke();
  }
  $('run').disabled = busy || frameLoading || (mode() === 'scene' ? !video?.analysis?.ready : !anchor);
}

function describeMode() {
  const automatic = mode() === 'scene';
  $('heading').textContent = automatic ? 'Automatic scene stabilization.' : 'Selected-light stabilization.';
  $('intro').textContent = automatic
    ? 'Estimate camera movement from background detail across the image. Preserve the full frame and rigid geometry.'
    : 'Select the moving light on its brightest frame. Track backward to its appearance and forward to its disappearance.';
  $('cropOption').hidden = automatic;
  $('frameControls').hidden = automatic;
  syncFrameControls();
  $('cropToSelection').disabled = automatic || busy;
  canvas.style.cursor = automatic ? 'default' : 'crosshair';
  $('selectionTitle').textContent = automatic ? '2. Review automatic scene references' : '2. Find and select the moving light';
  $('selectionHelp').textContent = automatic
    ? 'No selection is needed. Background references are detected across the image; moving features are rejected during processing.'
    : 'Scrub to the brightest frame and draw a box around the light. The export holds it in place as it moves through the sky; the background will move.';
  $('run').textContent = automatic ? 'Stabilize the scene' : 'Stabilize on the selected light';
  if (automatic && video) {
    $('coordinates').textContent = video.analysis.ready
      ? `${video.analysis.reference_count} scene references detected automatically. Ready to measure the whole video.`
      : 'Not enough fixed scene detail was found for automatic stabilization.';
  } else {
    $('coordinates').textContent = 'Draw a box around the moving light on this frame.';
  }
  draw();
}

$('mode').onchange = async () => {
  clearTimeout(scrubTimer); frameVersion++; frameLoading = false;
  selectionVersion++; start = box = anchor = sourceAnchor = inspectionWindow = null;
  output.pause(); $('result').hidden = true;
  $('trackRange').hidden = true;
  describeMode();
  if (video && (referenceFrame !== loadedFrame || (mode() === 'scene' && loadedFrame !== 0))) await showFrame(mode() === 'scene' ? 0 : referenceFrame);
};

$('file').onchange = async () => {
  if (!$('file').files.length) return;
  try {
    bundleVersion++;
    busy = true; $('file').disabled = $('mode').disabled = $('cropToSelection').disabled = true;
    selectionVersion++; frameVersion++; clearTimeout(scrubTimer); referenceFrame = 0; frameLoading = false; syncFrameControls();
    start = box = anchor = sourceAnchor = inspectionWindow = null;
    $('trackRange').hidden = true;
    output.pause(); $('run').disabled = true;
    $('result').hidden = $('selection').hidden = true;
    $('status').textContent = 'Loading first frame…';
    const form = new FormData();
    form.append('file', $('file').files[0]);
    video = await request('/api/videos', { method: 'POST', body: form });
    if (!Number.isInteger(video.frame_count) || video.frame_count < 1) {
      video = null;
      throw Error('The running backend does not provide video frames. Restart the stabilizer backend and reload this page.');
    }
    loadedFrame = 0;
    image = new Image(); image.src = video.first;
    await image.decode();
    canvas.width = video.width; canvas.height = video.height;
    $('selection').hidden = false;
    describeMode();
    $('status').textContent = `${video.width} × ${video.height} · ${video.fps.toFixed(2)} fps · ${video.duration.toFixed(1)} seconds`;
  } catch (error) {
    $('status').textContent = error.message;
  } finally {
    busy = false; $('file').disabled = $('mode').disabled = false;
    $('cropToSelection').disabled = mode() !== 'light'; syncFrameControls(); draw();
  }
};

canvas.onpointerdown = event => {
  if (busy || frameLoading || !video || mode() !== 'light') return;
  selectionVersion++; anchor = sourceAnchor = box = null; $('trackRange').hidden = true;
  output.pause(); $('result').hidden = true;
  start = point(event); canvas.setPointerCapture(event.pointerId); draw();
};
canvas.onpointermove = event => {
  if (!start || busy) return;
  box = selectionBox(start, point(event)); draw();
};
canvas.onpointerup = async () => {
  if (!start || busy) return;
  start = null;
  const version = ++selectionVersion;
  if (!box || box[2] < 12 || box[3] < 12) {
    $('coordinates').textContent = 'Draw a box at least 12 × 12 pixels.';
    return;
  }
  $('coordinates').textContent = 'Locating the selected light…';
  try {
    const reference = await post('/api/reference', { video_id: video.id, box, reference_frame: referenceFrame, track_visible_segment: true });
    if (version !== selectionVersion) return;
    anchor = sourceAnchor = reference.anchor;
    inspectionWindow = reference.window;
    $('coordinates').textContent = `Light center: (${anchor[0].toFixed(3)}, ${anchor[1].toFixed(3)}).`;
    $('trackRange').hidden = false;
    $('trackRange').textContent = `Detected light: frames ${reference.first_visible_frame + 1}–${reference.last_visible_frame + 1}. Export: frames ${reference.start_frame + 1}–${reference.end_frame + 1}, including one frame before and after when available.` + (Object.values(reference.boundary_reasons).includes('ambiguous-match') ? ' A boundary has competing light candidates; review the export range.' : '');
    draw();
  } catch (error) {
    if (version === selectionVersion) $('coordinates').textContent = error.message;
  }
};
canvas.onpointercancel = () => { start = null; };

$('run').onclick = async () => {
  if (busy || frameLoading || (mode() === 'scene' ? !video?.analysis?.ready : !anchor)) return;
  bundleVersion++;
  busy = true; syncFrameControls(); $('run').disabled = $('file').disabled = $('mode').disabled = $('cropToSelection').disabled = true;
  output.pause();
  $('result').hidden = true;
  try {
    const job = await post('/api/stabilize', { video_id: video.id, mode: mode(), ...(mode() === 'light' ? { box, crop_to_selection: $('cropToSelection').checked, reference_frame: referenceFrame, track_visible_segment: true } : {}) });
    let state;
    do {
      await new Promise(resolve => setTimeout(resolve, 600));
      state = await request(`/api/jobs/${job.id}`);
      $('status').textContent = `${state.message} · ${Math.round(state.progress * 100)}%`;
    } while (['queued', 'running'].includes(state.status));
    if (state.status !== 'complete') throw Error(state.message);
    const result = state.result;
    currentResult = result; bundleUrl = null; bundleVersion++;
    $('download').disabled = false;
    $('download').textContent = 'Download ZIP';
    $('bundleStatus').textContent = 'Includes the original video, stabilized video and frames, and mild, standard and strong AI-enhanced videos with matching frames. AI processing runs locally and may take several minutes.';
    $('exportQuality').textContent = `${result.output_width} × ${result.output_height} pixels · ${result.fps.toFixed(3)} fps · High-quality MP4. Resolution follows your crop setting.`;
    anchor = result.anchor;
    inspectionWindow = result.window;
    output.src = result.url;
    const padded = result.crop_to_selection && (result.crop_box[2] !== result.output_width || result.crop_box[3] !== result.output_height);
    $('framing').textContent = result.crop_to_selection
      ? (padded
        ? `Cropped to your ${result.crop_box[2]} × ${result.crop_box[3]} selection · ${result.output_width} × ${result.output_height} encoded pixels, including black edge padding.`
        : `Cropped to your selection · ${result.output_width} × ${result.output_height} pixels.`)
      : 'Full frame · no crop.';
    $('segment').hidden = result.reference_frame == null;
    $('segment').textContent = result.reference_frame == null ? '' : `Source frames ${result.start_frame + 1}–${result.end_frame + 1} · Reference frame ${result.reference_frame + 1} · Light detected ${result.first_visible_frame + 1}–${result.last_visible_frame + 1}. Padding frames hold the adjacent correction; no light position is measured there.`;
    const pixels = value => value == null ? 'unmeasurable' : `${value.toFixed(3)} px`;
    if (result.mode === 'scene') {
      $('qa').textContent = `${result.analysis.reference_count} automatic scene references · ${result.motion_model === 'rigid' ? 'Rotation and translation' : 'Translation only'} · ${result.output_frames} encoded frames checked. Background alignment: ${pixels(result.max_step_pixels)} maximum step; ${pixels(result.max_residual_pixels)} maximum drift. ${result.unmeasurable_frames} frames could not be verified.`;
      $('appearance').textContent = result.zero_motion_passed
        ? 'All background alignment checks measured zero movement.'
        : 'This export has not passed zero-motion QA. Inspect the full video and frame audit before accepting it. Individual light profiles may change independently of the fitted camera correction.';
    } else {
      $('qa').textContent = `Measured light-center movement across ${result.output_frames} encoded frames: ${pixels(result.max_step_pixels)} between frames; ${pixels(result.max_residual_pixels)} from the selected reference position.`;
      $('appearance').textContent = result.zero_motion_passed
        ? 'All recorded spatial checks passed with zero measured movement.'
        : `Light-profile QA has not passed. Separate light-profile checks measure up to ${pixels(result.profile_metrics.profile_center_2_step_pixels)} movement between frames. A stable measured center does not certify a completely stationary light profile.`;
    }
    $('result').hidden = false;
    $('result').scrollIntoView({ block: 'start' });
    $('status').textContent = `Rendered ${result.frames} frames. ${result.zero_motion_passed ? 'Spatial checks passed.' : 'Zero-motion QA has not passed.'}`;
  } catch (error) {
    $('status').textContent = error.message;
  } finally {
    busy = false; $('file').disabled = $('mode').disabled = false;
    $('cropToSelection').disabled = mode() !== 'light'; syncFrameControls(); draw();
  }
};

function paintInspector() {
  if (!anchor || !inspectionWindow || $('result').hidden || $('magnifier').hidden || output.readyState < 2) return;
  const width = Math.min(65, output.videoWidth), height = Math.min(65, output.videoHeight);
  if (!width || !height) return;
  const sx = Math.max(0, Math.min(inspectionWindow[0] - 32, output.videoWidth - width));
  const sy = Math.max(0, Math.min(inspectionWindow[1] - 32, output.videoHeight - height));
  if (inspector.width !== width * 6 || inspector.height !== height * 6) {
    inspector.width = width * 6; inspector.height = height * 6;
    inspector.style.width = `min(100%, ${width * 6}px)`;
  }
  inspection.clearRect(0, 0, inspector.width, inspector.height);
  inspection.imageSmoothingEnabled = false;
  inspection.drawImage(output, sx, sy, width, height, 0, 0, width * 6, height * 6);
  const cx = (anchor[0] - sx + .5) * 6;
  const cy = (anchor[1] - sy + .5) * 6;
  inspection.strokeStyle = '#80ffdc'; inspection.lineWidth = 1;
  inspection.beginPath();
  inspection.moveTo(cx - 18, cy); inspection.lineTo(cx - 6, cy);
  inspection.moveTo(cx + 6, cy); inspection.lineTo(cx + 18, cy);
  inspection.moveTo(cx, cy - 18); inspection.lineTo(cx, cy - 6);
  inspection.moveTo(cx, cy + 6); inspection.lineTo(cx, cy + 18);
  inspection.stroke();
}

let inspectionCallback = null;
function queueInspection() {
  if (inspectionCallback !== null || output.paused || output.ended) return;
  const nextFrame = output.requestVideoFrameCallback
    ? callback => output.requestVideoFrameCallback(callback)
    : callback => requestAnimationFrame(callback);
  inspectionCallback = nextFrame(() => {
    inspectionCallback = null;
    paintInspector();
    queueInspection();
  });
}
output.addEventListener('play', queueInspection);
output.addEventListener('loadeddata', paintInspector);
output.addEventListener('seeked', paintInspector);
output.addEventListener('pause', () => {
  if (inspectionCallback !== null) {
    if (output.cancelVideoFrameCallback) output.cancelVideoFrameCallback(inspectionCallback);
    else cancelAnimationFrame(inspectionCallback);
    inspectionCallback = null;
  }
  paintInspector();
});
output.addEventListener('ended', paintInspector);

$('cropToSelection').onchange = () => {
  output.pause(); $('result').hidden = true;
  if (video) $('status').textContent = 'Crop option changed. Run stabilization to update the export.';
};

output.addEventListener('error', () => {
  $('retryPreview').hidden = false;
  $('status').textContent = 'The browser could not decode this preview. The export and frame audit remain available.';
});
output.addEventListener('loadeddata', () => { $('retryPreview').hidden = true; });
$('retryPreview').onclick = () => {
  $('retryPreview').hidden = true;
  $('status').textContent = 'Reloading the preview…';
  output.load();
};


function saveBundle(url) {
  const link=document.createElement('a');
  link.href=url;link.download='ufo-video-exports.zip';document.body.append(link);link.click();link.remove();
}
$('download').onclick = async () => {
  if (bundleUrl) { saveBundle(bundleUrl); return; }
  if (!currentResult) return;
  const version=++bundleVersion, parent=currentResult;
  $('download').disabled=true;
  $('download').textContent='Preparing ZIP…';
  $('bundleStatus').textContent='Preparing all videos, AI thresholds and PNG frames…';
  try {
    const job=await post(parent.bundle_request,{});
    let state;
    do {
      await new Promise(resolve=>setTimeout(resolve,1000));
      if (version!==bundleVersion) return;
      state=await request(`/api/jobs/${job.id}`);
      if (version!==bundleVersion) return;
      $('bundleStatus').textContent=`${state.message} · ${Math.round(state.progress*100)}%`;
    } while(state.status==='queued'||state.status==='running');
    if(state.status!=='complete') throw Error(state.message);
    bundleUrl=state.result.url;
    $('bundleStatus').textContent='ZIP ready: original, stabilized, all three AI denoising levels, and matching PNG frames. Model settings and timestamps are included.';
    saveBundle(bundleUrl);
  } catch(error) {
    if(version===bundleVersion) $('bundleStatus').textContent=error.message;
  } finally {
    if(version===bundleVersion) { $('download').disabled=false;$('download').textContent='Download ZIP'; }
  }
};
