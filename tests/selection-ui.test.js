import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

test('selection zoom redraws the source frame and click uses zoomed coordinates',async()=>{
 const calls=[];
 const context=new Proxy({}, {get:(target,key)=>target[key]??((...args)=>calls.push([key,...args]))});
 const html=await readFile(new URL('../dist/index.html',import.meta.url),'utf8');
 const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],{
  value:'1',style:{},addEventListener(){},getContext:()=>context,
  width:800,height:400,clientWidth:800,getBoundingClientRect:()=>({left:0,top:0,width:800,height:400}),setPointerCapture(){},
 }]));
 const get=id=>elements.get(id);
 Object.assign(get('video'),{videoWidth:800,videoHeight:400,duration:10,currentTime:0,readyState:2,pause(){}});
 get('mode').value='point';get('patch').value='24';
 const file=new Blob(['video']);file.name='sample.mp4';get('file').files=[file];
 const originalDocument=globalThis.document;
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
  calls.length=0;get('zoomIn').onclick();assert.equal(calls.filter(c=>c[0]==='scale'&&c[1]===6).length,2,'input and output both render at 6x');
  calls.length=0;get('fit').onclick();assert.ok(calls.some(c=>c[0]==='scale'&&c[1]===1));
 }finally{globalThis.document=originalDocument;URL.revokeObjectURL(get('video').src);}
});

test('slider zoom centers an off-center target remembered from pointer movement',async()=>{
 const calls=[];
 const context=new Proxy({}, {get:(target,key)=>target[key]??((...args)=>calls.push([key,...args]))});
 const html=await readFile(new URL('../dist/index.html',import.meta.url),'utf8');
 const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],{
  value:'1',style:{},addEventListener(){},getContext:()=>context,
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
