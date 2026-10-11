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
  assert.equal(get('outputZoomControl').hidden,true,'output zoom stays hidden before tracking');
  calls.length=0;get('zoomIn').onclick();assert.ok(calls.some(c=>c[0]==='scale'&&c[1]===6));
  calls.length=0;get('fit').onclick();assert.ok(calls.some(c=>c[0]==='scale'&&c[1]===1));
 }finally{globalThis.document=originalDocument;URL.revokeObjectURL(get('video').src);}
});
