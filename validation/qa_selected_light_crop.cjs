const path=require('node:path'),assert=require('node:assert/strict'),fs=require('node:fs');
const ROOT=path.resolve(__dirname,'..');
const {chromium}=require(path.resolve(ROOT,'../homepage-redesign/node_modules/playwright'));
const {execFileSync}=require('node:child_process');
(async()=>{const browser=await chromium.launch({headless:true});try{
 const page=await browser.newPage({viewport:{width:1440,height:1800}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:8080/');
 assert.equal((await(await page.request.get('http://127.0.0.1:8080/api/health')).json()).version,14);
 assert.equal(await page.locator('#cropOption').isVisible(),false);
 await page.locator('#mode').selectOption('light');
 await page.locator('#file').setInputFiles('/Volumes/UFO Files Archive 2/ir/4E1A78FF-32B3-48D4-9399-7E77B2A45BB1.mp4');
 await page.locator('#selection').waitFor({state:'visible',timeout:60000});
 assert.equal(await page.locator('#cropToSelection').isChecked(),false);
 assert.equal(await page.locator('#cropOption').isVisible(),true);
 const c=await page.locator('#canvas').boundingBox();
 await page.mouse.move(c.x+1264/1920*c.width,c.y+234/1080*c.height);await page.mouse.down();
 await page.mouse.move(c.x+1328/1920*c.width,c.y+298/1080*c.height,{steps:10});await page.mouse.up();
 await page.waitForFunction(()=>document.getElementById('coordinates').textContent.startsWith('Light center:'));
 const sourceCanvas=await page.locator('#canvas').evaluate(c=>c.toDataURL());
 async function run(crop){
  const response=page.waitForResponse(r=>r.url().endsWith('/api/stabilize')&&r.request().method()==='POST');
  await page.locator('#run').click();const posted=await response;
  assert.equal(posted.request().postDataJSON().crop_to_selection,crop);
  const id=(await posted.json()).id;await page.locator('#result').waitFor({state:'visible',timeout:180000});
  const job=await(await page.request.get('http://127.0.0.1:8080/api/jobs/'+id)).json();assert.equal(job.status,'complete');
  const report=await(await page.request.get('http://127.0.0.1:8080'+job.result.report)).json();
  assert.equal(report.crop_to_selection,crop);assert.equal(report.frames,349);assert.equal(report.output_frames,349);
  assert.equal(report.rigid_translation_samples_passed,true);
  await page.waitForFunction(({w,h})=>{const v=document.getElementById('output');return v.readyState>=2&&v.videoWidth===w&&v.videoHeight===h;},{w:report.output_width,h:report.output_height});
  return {job,report};
 }
 const full=await run(false);assert.equal(full.report.output_width,1920);assert.equal(full.report.output_height,1080);
 console.log('Unchecked: actual export remains 1920x1080, all 349 frames audited.');
 await page.locator('#cropToSelection').check();assert.equal(await page.locator('#result').isVisible(),false);
 const cropped=await run(true);assert.deepEqual(cropped.report.crop_box,[1264,234,64,64]);
 assert.equal(cropped.report.output_width,64);assert.equal(cropped.report.output_height,64);
 assert.deepEqual(cropped.report.rows,full.report.rows,'Cropping must not change the stabilization trajectory.');
 assert.equal(await page.locator('#canvas').evaluate(c=>c.toDataURL()),sourceCanvas,'The selection crosshair must stay in source coordinates.');
 const destination=path.join(ROOT,'validation/artifacts/selected-light-crop.mp4');
 const download=page.waitForEvent('download');await page.locator('#download').click();await(await download).saveAs(destination);
 const timestamps=JSON.parse(execFileSync('ffprobe',['-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',destination])).frames.map(f=>Number(f.best_effort_timestamp_time));assert.equal(timestamps.length,349);
 await page.locator('#output').scrollIntoViewIfNeeded();console.log('Checked: actual 64x64 crop; testing playback and every displayed frame.');
 const playback=await page.evaluate(async({timestamps,anchor})=>{
  const v=document.getElementById('output');v.loop=false;v.muted=true;
  const before=v.getVideoPlaybackQuality();const ended=new Promise((resolve,reject)=>{v.addEventListener('ended',resolve,{once:true});setTimeout(()=>reject(Error('Playback timeout')),40000);});await v.play();await ended;v.pause();const after=v.getVideoPlaybackQuality();
  const canvas=document.createElement('canvas');canvas.width=v.videoWidth;canvas.height=v.videoHeight;const ctx=canvas.getContext('2d',{willReadFrequently:true});const observed=new Map();
  const nearest=t=>timestamps.reduce((a,time,i)=>Math.abs(time-t)<Math.abs(timestamps[a]-t)?i:a,0);
  function sample(index){ctx.drawImage(v,0,0);const p=ctx.getImageData(0,0,64,64).data;let max=-1,at=0;for(let k=0;k<4096;k++){const y=.2126*p[k*4]+.7152*p[k*4+1]+.0722*p[k*4+2];if(y>max){max=y;at=k;}}observed.set(index,{frame:index,maximum:max,x:at%64,y:Math.floor(at/64),distance:Math.hypot(at%64-anchor[0],Math.floor(at/64)-anchor[1])});}
  async function seek(index){await new Promise((resolve,reject)=>{let id;const timer=setTimeout(()=>{v.cancelVideoFrameCallback(id);reject(Error('Seek timeout'));},8000);const check=(_now,metadata)=>{if(nearest(metadata.mediaTime)===index){clearTimeout(timer);sample(index);resolve();}else id=v.requestVideoFrameCallback(check);};id=v.requestVideoFrameCallback(check);v.currentTime=timestamps[index]+.001;});}
  await seek(0);v.playbackRate=.25;let id;const inspect=(_now,m)=>{sample(nearest(m.mediaTime));if(!v.ended)id=v.requestVideoFrameCallback(inspect);};id=v.requestVideoFrameCallback(inspect);const finished=new Promise((resolve,reject)=>{v.addEventListener('ended',resolve,{once:true});setTimeout(()=>reject(Error('Inspection timeout')),70000);});await v.play();await finished;v.pause();v.cancelVideoFrameCallback(id);
  for(let i=0;i<timestamps.length;i++)if(!observed.has(i))await seek(i);
  return {dropped:after.droppedVideoFrames-before.droppedVideoFrames,observations:[...observed.values()].sort((a,b)=>a.frame-b.frame),inspectorWidth:document.getElementById('inspector').width,inspectorHeight:document.getElementById('inspector').height};
 },{timestamps,anchor:cropped.report.anchor});
 assert.equal(playback.dropped,0);assert.equal(playback.observations.length,349);
 assert.ok(playback.observations.every(f=>f.maximum>30&&f.distance<8),'The selected light must remain visible inside the cropped export.');
 assert.equal(playback.inspectorWidth,384);assert.equal(playback.inspectorHeight,384);
 await page.screenshot({path:path.join(ROOT,'validation/artifacts/selected-light-crop-browser.png'),fullPage:true});
 await page.locator('#mode').selectOption('scene');assert.equal(await page.locator('#cropOption').isVisible(),false);
 assert.deepEqual(errors,[]);
 fs.writeFileSync(path.join(ROOT,'validation/artifacts/selected-light-crop-qa.json'),JSON.stringify({full:full.job.result,cropped:cropped.job.result,playback,errors,stabilizationAccepted:cropped.report.zero_motion_passed},null,2));
 console.log(JSON.stringify({fullDimensions:[1920,1080],cropDimensions:[64,64],framesInspected:playback.observations.length,dropped:playback.dropped,trajectoryUnchanged:true,sourceCrosshairUnchanged:true,errors}));
}finally{await browser.close();}})().catch(e=>{console.error(e);process.exitCode=1;});
