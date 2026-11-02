import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

test('selection zoom redraws the source frame and click uses zoomed coordinates',async()=>{
 const calls=[];
 const context=new Proxy({}, {get:(target,key)=>target[key]??((...args)=>calls.push([key,...args]))});
 const html=await readFile(new URL('../dist/index.html',import.meta.url),'utf8');
 const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],{
  value:'1',checked:m[1]==='cropToSelection',style:{},addEventListener(){},getContext:()=>context,
  width:800,height:400,clientWidth:800,getBoundingClientRect:()=>({left:0,top:0,width:800,height:400}),setPointerCapture(){},
 }]));
 const get=id=>elements.get(id);
 Object.assign(get('video'),{videoWidth:800,videoHeight:400,duration:10,currentTime:0,readyState:2,pause(){}});
 get('mode').value='point';get('patch').value='24';
 const file=new Blob(['video']);file.name='sample.mp4';get('file').files=[file];
 const originalDocument=globalThis.document,originalFetch=globalThis.fetch;
 globalThis.document={getElementById:get,createElement:()=>({getContext:()=>context}),addEventListener(){}};
 try{
  await import('../dist/app.js?selection-ui-test');
  await get('file').onchange();get('video').onloadeddata();
  calls.length=0;get('sourceZoom').value='4';get('sourceZoom').oninput();
  assert.ok(calls.some(c=>c[0]==='scale'&&c[1]===4&&c[2]===4),'source canvas must visibly render at 4x');
  assert.ok(calls.some(c=>c[0]==='drawImage'&&c[1]===get('video')),'zoom must redraw the video');
  const e={clientX:450,clientY:220,pointerId:1,button:0};
  get('source').onpointerdown(e);calls.length=0;get('source').onpointerup(e);
  const rect=calls.find(c=>c[0]==='strokeRect');
  assert.deepEqual(rect.slice(1),[400.5,193,24,24],'clicked patch must map back to original pixels');
  assert.equal(get('track').disabled,false);
  assert.ok(calls.some(c=>c[0]==='drawImage'&&c[1]===get('video')&&c[2]===-412.5&&c[3]===-205),'centered output uses the selected target');
  calls.length=0;get('zoomIn').onclick();assert.equal(calls.filter(c=>c[0]==='scale'&&c[1]===6).length,1,'selection zoom affects only the source');
  assert.equal(get('output').width,1080,'square crop exports at HD resolution');
  assert.equal(get('output').height,1080,'square crop preserves its aspect ratio');
  calls.length=0;get('fit').onclick();assert.ok(calls.some(c=>c[0]==='scale'&&c[1]===1));
  get('cropToSelection').checked=false;calls.length=0;get('cropToSelection').onchange();
  assert.equal(get('output').width,800);assert.equal(get('output').height,400);
  assert.match(get('selection').textContent,/Full frame/);
  assert.ok(calls.some(c=>c[0]==='drawImage'&&c[1]===get('video')&&c[2]===0&&c[3]===0&&c[4]===800&&c[5]===400),'the reference starts at its original position');
  assert.ok(!calls.some(c=>c[0]==='drawImage'&&c[2]<0),'unchecked preview never translates toward the target');
  let payload;
  globalThis.fetch=async(path,options)=>{
   let result;
   if(path==='/api/health')result={engine:'opencv'};
   else if(path==='/api/videos')result={id:'video',width:800,height:400};
   else if(path==='/api/track'){payload=JSON.parse(options.body);result={id:'job'};}
   else result={status:'complete',result:{rows:[{time:0,x:412,y:204.5,confidence:1},{time:1,x:416,y:201.5,confidence:1}],start:0,fps:30,rejected:0,mode:'flow'}};
   return {ok:true,json:async()=>result};
  };
  await get('track').onclick();
  assert.deepEqual(payload.point,[412,204.5]);
  assert.equal(payload.size,24,'the same box remains the tracking footprint');
  assert.equal(payload.center_selection,false);
  assert.equal(payload.crop_width,null);assert.equal(payload.crop_height,null);
  assert.equal(payload.output_width,800);assert.equal(payload.output_height,400);
  assert.equal(get('export').disabled,false);
  calls.length=0;get('zoomIn').onclick();
  assert.ok(calls.some(c=>c[0]==='drawImage'&&c[1]===get('video')&&c[2]===0&&c[3]===0),'first tracked reference is not moved to the center');
  get('video').currentTime=1;calls.length=0;get('zoomIn').onclick();
  assert.ok(calls.some(c=>c[0]==='drawImage'&&c[1]===get('video')&&c[2]===-4&&c[3]===3),'preview cancels reference displacement without recentering');
  const native=get('renderedPreview');let renders=0,plays=0;
  native.pause=()=>{};native.play=async()=>{plays++;};native.getAttribute=name=>native[name];
  get('video').play=()=>assert.fail('the original video must stay still during rendered preview');
  globalThis.fetch=async(path)=>{
   if(path.endsWith('/export')){renders++;return {ok:true,json:async()=>({id:'preview-export'})};}
   return {ok:true,json:async()=>({status:'complete',result:{download:'/rendered.mp4',filename:'stabilized.mp4',frames:60,fps:30,stability_qa:{status:'failed',decoded_frames:60,maximum_step_pixels:2.5,maximum_drift_pixels:3.2,unreliable_step_frames:1,unreliable_drift_frames:2}}})};
  };
  await get('preview').onclick();
  assert.match(get('qualityStatus').textContent,/Stability check FAILED/);
  assert.match(get('qualityStatus').textContent,/2.50 px/);
  assert.equal(native.src,'/rendered.mp4');assert.equal(native.hidden,false);assert.equal(get('output').hidden,true);
  assert.equal(get('video').currentTime,1,'preview does not seek the original video');
  native.onended();assert.match(get('status').textContent,/Stability check FAILED/);await get('preview').onclick();native.onended();
  assert.equal(renders,1,'replay reuses the exact encoded result');assert.equal(plays,2);
  get('cropToSelection').checked=true;get('cropToSelection').onchange();
  assert.equal(native.hidden,true,'framing changes hide the previous rendered preview');
  assert.equal(get('output').hidden,false);

  assert.equal(get('output').width,1080);assert.equal(get('output').height,1080);
  assert.equal(get('export').disabled,true,'framing changes invalidate the old export job');

 }finally{globalThis.document=originalDocument;globalThis.fetch=originalFetch;URL.revokeObjectURL(get('video').src);}
});

test('slider zoom centers an off-center target remembered from pointer movement',async()=>{
 const calls=[];
 const context=new Proxy({}, {get:(target,key)=>target[key]??((...args)=>calls.push([key,...args]))});
 const html=await readFile(new URL('../dist/index.html',import.meta.url),'utf8');
 const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],{
  value:'1',checked:m[1]==='cropToSelection',style:{},addEventListener(){},getContext:()=>context,
  width:800,height:400,clientWidth:800,getBoundingClientRect:()=>({left:0,top:0,width:800,height:400}),setPointerCapture(){},
 }]));
 const get=id=>elements.get(id);
 Object.assign(get('video'),{videoWidth:800,videoHeight:400,duration:10,currentTime:0,readyState:2,pause(){}});
 get('mode').value='point';get('patch').value='24';
 const file=new Blob(['video']);file.name='sample.mp4';get('file').files=[file];
 const originalDocument=globalThis.document;
 globalThis.document={getElementById:get,createElement:()=>({getContext:()=>context}),addEventListener(){}};
 try{
  await import('../dist/app.js?target-zoom-test');
  await get('file').onchange();get('video').onloadeddata();
  get('source').onpointermove({clientX:120,clientY:75});
  calls.length=0;get('sourceZoom').value='4';get('sourceZoom').oninput();
  assert.ok(calls.some(c=>c[0]==='translate'&&c[1]===-120&&c[2]===-75),'zoom must center the off-center target');
  const e={clientX:400,clientY:200,pointerId:1,button:0};
  get('source').onpointerdown(e);calls.length=0;get('source').onpointerup(e);
  assert.deepEqual(calls.find(c=>c[0]==='strokeRect').slice(1),[108,63,24,24]);
  // Later hover must not replace an explicitly selected target for slider zoom.
  get('source').onpointermove({clientX:600,clientY:300});
  calls.length=0;get('zoomIn').onclick();
  assert.ok(calls.some(c=>c[0]==='translate'&&c[1]===-120&&c[2]===-75));
 }finally{globalThis.document=originalDocument;URL.revokeObjectURL(get('video').src);}
});

test('touch dragging moves an existing box at zoom without resizing and stays inside the video',async()=>{
 const calls=[];
 const context=new Proxy({}, {get:(target,key)=>target[key]??((...args)=>calls.push([key,...args]))});
 const html=await readFile(new URL('../dist/index.html',import.meta.url),'utf8');
 const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],{
  value:'1',checked:m[1]==='cropToSelection',style:{},addEventListener(){},getContext:()=>context,
  width:800,height:400,clientWidth:800,getBoundingClientRect:()=>({left:0,top:0,width:800,height:400}),setPointerCapture(){},
 }]));
 const get=id=>elements.get(id);
 Object.assign(get('video'),{videoWidth:800,videoHeight:400,duration:10,currentTime:0,readyState:2,pause(){}});
 get('mode').value='box';get('patch').value='24';
 const file=new Blob(['video']);file.name='sample.mp4';get('file').files=[file];
 const originalDocument=globalThis.document;
 globalThis.document={getElementById:get,createElement:()=>({getContext:()=>context}),addEventListener(){}};
 let prevented=0;
 const event=(x,y,id=1)=>({clientX:x,clientY:y,pointerId:id,pointerType:'touch',isPrimary:true,button:0,preventDefault(){prevented++;}});
 try{
  await import('../dist/app.js?move-selection-test');
  await get('file').onchange();get('video').onloadeddata();
  const source=get('source');
  source.onpointerdown(event(200,100));source.onpointermove(event(400,200));source.onpointerup(event(400,200));
  get('sourceZoom').value='4';get('sourceZoom').oninput();
  source.onpointerdown(event(400,200));calls.length=0;
  source.onpointermove(event(480,240,2));
  assert.equal(calls.length,0,'another finger cannot move the selection');
  source.onpointermove(event(480,240));source.onpointerup(event(480,240));
  assert.deepEqual(calls.find(c=>c[0]==='strokeRect').slice(1),[220,110,200,100]);
  assert.equal(get('output').width,1920);assert.equal(get('output').height,960);
  assert.equal(get('track').disabled,false);
  // The view remains centered at (300,150), so this point is inside the moved box.
  source.onpointerdown(event(480,240));calls.length=0;
  source.onpointermove(event(5000,5000));source.onpointerup(event(5000,5000));
  assert.deepEqual(calls.find(c=>c[0]==='strokeRect').slice(1),[600,300,200,100]);
  assert.ok(prevented>=6,'selection gestures prevent browser touch handling');
 }finally{globalThis.document=originalDocument;URL.revokeObjectURL(get('video').src);}
});

test('file picker stays available and replacing an in-flight video ignores its old response',async()=>{
 const context=new Proxy({}, {get:(target,key)=>target[key]??(()=>{})});
 const html=await readFile(new URL('../dist/index.html',import.meta.url),'utf8');
 const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],{
  value:'1',checked:m[1]==='cropToSelection',style:{},addEventListener(){},getContext:()=>context,
  width:800,height:400,clientWidth:800,getBoundingClientRect:()=>({left:0,top:0,width:800,height:400}),setPointerCapture(){},
 }]));
 const get=id=>elements.get(id);
 Object.assign(get('video'),{videoWidth:800,videoHeight:400,duration:10,currentTime:0,readyState:2,pause(){}});
 get('mode').value='point';get('patch').value='24';
 const originalDocument=globalThis.document,originalFetch=globalThis.fetch;
 globalThis.document={getElementById:get,createElement:()=>({getContext:()=>context}),addEventListener(){}};
 let finishOldUpload;
 const first=new Blob(['old video']);first.name='first.mov';
 const second=new Blob(['new video']);second.name='second.mp4';
 globalThis.fetch=()=>new Promise(resolve=>{finishOldUpload=resolve;});
 try{
  await import('../dist/app.js?replace-video-test');
  get('file').files=[first];const pending=get('file').onchange();
  assert.equal(get('file').disabled,false,'preparing a video must not lock the picker');
  get('file').files=[second];await get('file').onchange();get('video').onloadeddata();
  const newSource=get('video').src;
  finishOldUpload({ok:true,json:async()=>({id:'old-video',preview:'/old-video.mp4'})});await pending;
  assert.equal(get('video').src,newSource,'old preparation cannot replace the newly chosen video');
  assert.match(get('details').textContent,/second.mp4/);
  get('file').files=[second];await get('file').onchange();get('video').onloadeddata();
  assert.notEqual(get('video').src,newSource,'selecting the same file reloads it');
  assert.equal(get('file').value,'1','retain the native selection while its File is used for playback and upload');
  get('file').files=[]; // A picker cancellation must retain the selected File separately.
  await get('file').onchange();
  const pointer={clientX:400,clientY:200,pointerId:1,button:0};
  get('source').onpointerdown(pointer);get('source').onpointerup(pointer);
  let uploaded=false;
  globalThis.fetch=async(path,options)=>{
   assert.equal(get('file').disabled,false,'tracking must not lock the picker');
   let result;
   if(path==='/api/health')result={engine:'opencv'};
   else if(path==='/api/videos'){assert.equal(await options.body.get('file').text(),'new video');uploaded=true;result={id:'new-video',width:800,height:400};}
   else if(path==='/api/track')result={id:'job'};
   else result={status:'complete',result:{rows:[{x:400,y:200,confidence:1}],start:0,fps:30,rejected:0}};
   return {ok:true,json:async()=>result};
  };
  await get('track').onclick();
  assert.equal(uploaded,true);
  assert.match(get('status').textContent,/Tracked 1 source frames/);
 }finally{globalThis.document=originalDocument;globalThis.fetch=originalFetch;URL.revokeObjectURL(get('video').src);}
});
