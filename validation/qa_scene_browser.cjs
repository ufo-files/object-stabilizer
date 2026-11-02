// Actual automatic UI, export, normal playback, and every displayed frame.
// Geometry/coverage success does not bypass the final zero-motion gate.
const path = require('node:path');
const ROOT = path.resolve(__dirname, '..');
const { chromium } = require(path.resolve(ROOT, '../homepage-redesign/node_modules/playwright'));
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { execFileSync } = require('node:child_process');
const ARTIFACTS = path.join(ROOT, 'validation/artifacts');

(async () => {
  const browser = await chromium.launch({ headless: !process.argv.includes('--headed') });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1800 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:8080/');
    assert.equal(await page.locator('#mode').inputValue(), 'scene');
    const health = await (await page.request.get('http://127.0.0.1:8080/api/health')).json();
    assert.equal(health.engine, 'auto-scene-1');
    const uploaded = page.waitForResponse(r => r.url().endsWith('/api/videos') && r.request().method() === 'POST');
    await page.locator('#file').setInputFiles('/Volumes/UFO Files Archive 2/ir/4E1A78FF-32B3-48D4-9399-7E77B2A45BB1.mp4');
    const video = await (await uploaded).json();
    assert.equal(video.analysis.method, 'sparse_lights');
    assert.equal(video.analysis.reference_count, 4);
    await page.locator('#selection').waitFor({ state: 'visible', timeout: 60000 });
    await page.locator('#run').waitFor({ state: 'visible' });
    assert.equal(await page.locator('#run').isEnabled(), true, 'Automatic mode must work without drawing a box.');
    const submitted = page.waitForResponse(r => r.url().endsWith('/api/stabilize') && r.request().method() === 'POST');
    await page.locator('#run').click();
    const id = (await (await submitted).json()).id;
    await page.locator('#result').waitFor({ state: 'visible', timeout: 180000 });
    const job = await (await page.request.get('http://127.0.0.1:8080/api/jobs/' + id)).json();
    assert.equal(job.result.mode, 'scene');
    const report = await (await page.request.get('http://127.0.0.1:8080' + job.result.report)).json();
    assert.equal(report.engine, 'auto-scene-1');
    assert.equal(report.frames, 349); assert.equal(report.output_frames, 349);
    assert.equal(report.motion_model, 'translation');
    assert.equal(report.rigid_geometry_passed, true);
    const destination = path.join(ARTIFACTS, 'auto-scene-browser.mp4');
    const downloading = page.waitForEvent('download');
    await page.locator('#download').click();
    await (await downloading).saveAs(destination);
    fs.writeFileSync(path.join(ARTIFACTS, 'auto-scene-browser-report.json'), JSON.stringify(report, null, 2));
    const timestamps = JSON.parse(execFileSync('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', destination])).frames.map(frame => Number(frame.best_effort_timestamp_time));
    assert.equal(timestamps.length, 349);
    await page.locator('#output').scrollIntoViewIfNeeded();
    console.log('Automatic export ready. Testing normal playback, then inspecting all references in all 349 displayed frames.');
    const playback = await page.evaluate(async ({ points, timestamps }) => {
      const video = document.getElementById('output');
      video.muted = true; video.loop = false;
      if (video.readyState < 2) await new Promise((resolve, reject) => {
        video.addEventListener('loadeddata', resolve, { once: true });
        video.addEventListener('error', () => reject(Error('Video decode error.')), { once: true });
        setTimeout(() => reject(Error('Video load timeout.')), 20000);
      });
      const before = video.getVideoPlaybackQuality();
      const ended = new Promise((resolve, reject) => {
        video.addEventListener('ended', resolve, { once: true });
        setTimeout(() => reject(Error('Normal playback timed out.')), 40000);
      });
      await video.play(); await ended; video.pause();
      const after = video.getVideoPlaybackQuality();
      const canvas = document.createElement('canvas'); canvas.width = canvas.height = 65;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      const observations = new Map();
      const nearest = time => timestamps.reduce((best, value, index) => Math.abs(value-time)<Math.abs(timestamps[best]-time)?index:best,0);
      function sample(index, metadata, mode) {
        const references = points.map(point => {
          const wx = Math.round(point[0]), wy = Math.round(point[1]);
          ctx.drawImage(video, wx - 32, wy - 32, 65, 65, 0, 0, 65, 65);
          const rgba = ctx.getImageData(0, 0, 65, 65).data;
          const values = Array.from({ length: 4225 }, (_, p) => .2126 * rgba[p*4] + .7152 * rgba[p*4+1] + .0722 * rgba[p*4+2]);
          const median = array => [...array].sort((a,b)=>a-b)[Math.floor(array.length/2)];
          const background = median(values);
          const noise = Math.max(1, 1.4826 * median(values.map(v => Math.abs(v-background))));
          const signal = values.map(v => Math.max(0,v-background-3*noise));
          let peak=0; for(let p=1;p<4225;p++) if(signal[p]>signal[peak]) peak=p;
          const threshold = Math.max(8,signal[peak]*.08);
          const mask = new Uint8Array(4225), queue=[peak]; mask[peak]=1;
          for(let q=0;q<queue.length;q++) {
            const p=queue[q],x=p%65,y=Math.floor(p/65);
            for(let dy=-1;dy<=1;dy++) for(let dx=-1;dx<=1;dx++) {
              const nx=x+dx,ny=y+dy,k=ny*65+nx;
              if(nx>=0&&nx<65&&ny>=0&&ny<65&&!mask[k]&&signal[k]>threshold){mask[k]=1;queue.push(k);}
            }
          }
          const dilated=new Uint8Array(4225);
          for(const p of queue){const x=p%65,y=Math.floor(p/65);for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){const nx=x+dx,ny=y+dy;if(nx>=0&&nx<65&&ny>=0&&ny<65)dilated[ny*65+nx]=1;}}
          let total=0,sx=0,sy=0,bright=0,bx=0,by=0;
          for(let p=0;p<4225;p++){
            const mass=signal[p]*dilated[p],core=Math.max(values[p]-background,0)**8;
            total+=mass;sx+=(p%65)*mass;sy+=Math.floor(p/65)*mass;
            bright+=core;bx+=(p%65)*core;by+=Math.floor(p/65)*core;
          }
          return {x:wx-32+sx/total,y:wy-32+sy/total,brightX:wx-32+bx/bright,brightY:wy-32+by/bright};
        });
        observations.set(index,{frame:index,mediaTime:metadata.mediaTime,mode,references});
      }
      async function seekAndSample(index) {
        await new Promise((resolve,reject)=>{
          let callback;
          const timer=setTimeout(()=>{video.cancelVideoFrameCallback(callback);reject(Error('Frame seek timed out: '+index));},8000);
          const inspect=(_now,metadata)=>{
            if(nearest(metadata.mediaTime)===index){clearTimeout(timer);sample(index,metadata,'seek');resolve();}
            else callback=video.requestVideoFrameCallback(inspect);
          };
          callback=video.requestVideoFrameCallback(inspect);
          video.currentTime=timestamps[index]+.001;
        });
      }
      await seekAndSample(0);
      video.playbackRate=.25;
      let callback;
      const inspect=(_now,metadata)=>{sample(nearest(metadata.mediaTime),metadata,'quarter_speed');if(!video.ended)callback=video.requestVideoFrameCallback(inspect);};
      callback=video.requestVideoFrameCallback(inspect);
      const inspected=new Promise((resolve,reject)=>{video.addEventListener('ended',resolve,{once:true});setTimeout(()=>reject(Error('Frame inspection timed out.')),70000);});
      await video.play();await inspected;video.pause();video.cancelVideoFrameCallback(callback);
      for(let index=0;index<timestamps.length;index++) if(!observations.has(index)) await seekAndSample(index);
      return {width:video.videoWidth,height:video.videoHeight,duration:video.duration,
        decoded:after.totalVideoFrames-before.totalVideoFrames,dropped:after.droppedVideoFrames-before.droppedVideoFrames,
        observations:[...observations.values()].sort((a,b)=>a.frame-b.frame)};
    }, { points: report.analysis.points, timestamps });
    assert.equal(playback.observations.length, 349);
    const median=a=>{const sorted=[...a].sort((x,y)=>x-y);return (sorted[Math.floor((sorted.length-1)/2)]+sorted[Math.floor(sorted.length/2)])/2;};
    const first=playback.observations[0].references;
    const common=playback.observations.map(f=>({frame:f.frame,x:median(f.references.map((p,k)=>p.x-first[k].x)),y:median(f.references.map((p,k)=>p.y-first[k].y))}));
    const browserCommonMassDrift=Math.max(...common.map(p=>Math.hypot(p.x,p.y)));
    const browserCommonMassStep=Math.max(...common.slice(1).map((p,i)=>Math.hypot(p.x-common[i].x,p.y-common[i].y)));
    await page.screenshot({path:path.join(ARTIFACTS,'auto-scene-browser.png'),fullPage:true});
    const qa={job:id,video:video.id,result:job.result,playback,browserCommonMassDrift,browserCommonMassStep,errors,destination,zeroMotionAccepted:report.zero_motion_passed};
    fs.writeFileSync(path.join(ARTIFACTS,'auto-scene-browser-qa.json'),JSON.stringify(qa,null,2));
    console.log(JSON.stringify({job:id,mode:job.result.mode,referenceCount:report.analysis.reference_count,motionModel:report.motion_model,
      frames:report.frames,framesInspected:playback.observations.length,rigidGeometryPassed:report.rigid_geometry_passed,
      dropped:playback.dropped,browserCommonMassDrift,browserCommonMassStep,
      independentPatchDrift:report.max_residual_pixels,independentPatchStep:report.max_step_pixels,
      unmeasurableFrames:report.unmeasurable_frames,zeroMotionAccepted:report.zero_motion_passed,errors},null,2));
    assert.deepEqual(errors,[]);assert.equal(playback.width,1920);assert.equal(playback.height,1080);
    assert.equal(playback.dropped,0,'Normal playback must not drop frames.');
    if(!process.argv.includes('--diagnostics'))assert.equal(report.zero_motion_passed,true,'The automatic scene export has not passed zero-motion QA.');
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
