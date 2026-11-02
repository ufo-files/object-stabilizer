const assert=require('node:assert/strict'),path=require('node:path');
const {chromium}=require('../..//homepage-redesign/node_modules/playwright');
(async()=>{const browser=await chromium.launch({headless:true});try{
 const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:8080/');assert.equal((await(await page.request.get('http://127.0.0.1:8080/api/health')).json()).version,18);
 await page.selectOption('#mode','light');await page.setInputFiles('#file',path.resolve('validation/artifacts/moving-light-source.avi'));
 await page.locator('#selection').waitFor({state:'visible',timeout:60000});assert.equal(await page.locator('#referenceFrame').getAttribute('max'),'121');
 await page.locator('#referenceFrame').fill('55');await page.waitForFunction(()=>document.getElementById('status').textContent.startsWith('Source frame 55 of 121'));
 const c=await page.locator('#canvas').boundingBox();await page.mouse.move(c.x+172/480*c.width,c.y+106/240*c.height);await page.mouse.down();await page.mouse.move(c.x+197/480*c.width,c.y+131/240*c.height,{steps:5});await page.mouse.up();
 await page.waitForFunction(()=>!document.getElementById('run').disabled,{timeout:30000});
 const range=await page.locator('#trackRange').textContent();console.log(range);assert.match(range,/10.*100/);assert.match(range,/9.*101/);
 for(const crop of [false,true]){
 if(crop)await page.check('#cropToSelection');
 const pending=page.waitForResponse(r=>r.url().endsWith('/api/stabilize')&&r.request().method()==='POST');await page.click('#run');const id=(await(await pending).json()).id;
 await page.locator('#result').waitFor({state:'visible',timeout:60000});
 const job=await(await page.request.get('http://127.0.0.1:8080/api/jobs/'+id)).json();assert.equal(job.status,'complete');
 const r=await(await page.request.get('http://127.0.0.1:8080'+job.result.report)).json();assert.equal(r.reference_frame,54);assert.equal(r.output_frames,93);assert.equal(r.start_frame,8);assert.equal(r.end_frame,100);assert.equal(r.rigid_translation_samples_passed,true);assert.equal(r.audit.length,93);assert.ok(r.max_residual_pixels<.15);
 await page.waitForFunction(({w,h})=>{const v=document.getElementById('output');return v.readyState>=2&&v.videoWidth===w&&v.videoHeight===h;},{w:r.output_width,h:r.output_height});
 const play=await page.evaluate(async()=>{const v=document.getElementById('output');v.muted=true;v.loop=false;await v.play();await new Promise(resolve=>v.addEventListener('ended',resolve,{once:true}));return {time:v.currentTime,duration:v.duration};});assert.ok(play.time>3.8);
 await page.selectOption('#frameExportSize','0');await page.click('#exportFrames');await page.locator('#downloadFrames').waitFor({state:'visible',timeout:60000});
 const download=page.waitForEvent('download');await page.click('#downloadFrames');
 const zipPath=path.resolve(`validation/artifacts/moving-light-${crop?'crop':'full'}-frames.zip`);await(await download).saveAs(zipPath);
 require('node:child_process').execFileSync('.venv/bin/python',['-c',`import zipfile,json,csv,io,sys,cv2,numpy as np
with zipfile.ZipFile(sys.argv[1]) as z:
 names=sorted(n for n in z.namelist() if n.endswith('.png'));assert len(names)==93
 m=json.loads(z.read('metadata.json'));assert m['fps']=='24/1';assert (m['width'],m['height'])==(int(sys.argv[2]),int(sys.argv[3]))
 rows=list(csv.DictReader(io.StringIO(z.read('frames.csv').decode())));assert [int(r['source_frame']) for r in rows]==list(range(9,102))
 assert abs(float(rows[-1]['timestamp_seconds'])-92/24)<.00001
 for n in names:
  a=cv2.imdecode(np.frombuffer(z.read(n),np.uint8),1);assert a.shape[:2]==(m['height'],m['width'])
`,zipPath,String(r.output_width),String(r.output_height)]);
 if(crop){
  await page.selectOption('#frameExportSize','1920');assert.match(await page.locator('#frameSizeHelp').textContent(),/1920 × 1920/);
  await page.click('#exportFrames');await page.locator('#downloadFrames').waitFor({state:'visible',timeout:120000});
  const enlarged=page.waitForEvent('download');await page.click('#downloadFrames');const enlargedPath=path.resolve('validation/artifacts/moving-light-enlarged-frames.zip');await(await enlarged).saveAs(enlargedPath);
  require('node:child_process').execFileSync('.venv/bin/python',['-c',`import zipfile,struct,sys
with zipfile.ZipFile(sys.argv[1]) as z:
 names=[n for n in z.namelist() if n.endswith('.png')];assert len(names)==93
 for n in names:
  with z.open(n) as f:assert struct.unpack('>II',f.read(24)[16:24])==(1920,1920)
`,enlargedPath]);
  await page.setViewportSize({width:390,height:844});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  await page.locator('#result').screenshot({path:'validation/artifacts/png-size-controls-mobile.png'});
  await page.setViewportSize({width:1440,height:1100});
 }
 console.log(JSON.stringify({crop,frames:r.output_frames,dimensions:[r.output_width,r.output_height],residual:r.max_residual_pixels,step:r.max_step_pixels,play}));
 }
 await page.locator('#referenceFrame').fill('80');await page.selectOption('#mode','scene');await page.waitForTimeout(500);assert.equal(await page.locator('#referenceFrame').inputValue(),'1');
 await page.selectOption('#mode','light');await page.setViewportSize({width:390,height:844});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 await page.screenshot({path:'validation/artifacts/moving-light-mobile.png',fullPage:true});assert.deepEqual(errors,[]);console.log('Scrubbing, backward tracking, both exports, playback, mode change and mobile layout passed.');
}finally{await browser.close();}})().catch(e=>{console.error(e);process.exitCode=1;});
