import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

test('multiple references map through zoom and remain independent of the crop box',async()=>{
 const context=new Proxy({}, {get:(target,key)=>target[key]??(()=>{})});
 const html=await readFile(new URL('../dist/index.html',import.meta.url),'utf8');
 const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],{
  value:'1',checked:m[1]==='cropToSelection',style:{},addEventListener(){},getContext:()=>context,
  width:800,height:400,clientWidth:800,getBoundingClientRect:()=>({left:0,top:0,width:800,height:400}),setPointerCapture(){},
 }]));
 const get=id=>elements.get(id);Object.assign(get('video'),{videoWidth:800,videoHeight:400,duration:10,currentTime:0,readyState:2,pause(){}});
 get('mode').value='box';get('patch').value='24';
 const file=new Blob(['video']);file.name='sample.mp4';get('file').files=[file];
 const originalDocument=globalThis.document,originalFetch=globalThis.fetch;
 globalThis.document={getElementById:get,createElement:()=>({getContext:()=>context}),addEventListener(){}};
 const pointer=(x,y)=>({clientX:x,clientY:y,pointerId:1,button:0,preventDefault(){}});
 const click=(x,y)=>{get('source').onpointerdown(pointer(x,y));get('source').onpointerup(pointer(x,y));};
 let payload;
 globalThis.fetch=async(path,options)=>{
  let result;
  if(path==='/api/health')result={engine:'opencv',features:['multiple_points','rigid_multi_reference','checked_rigid_rotation','primary_image_motion','joint_primary_motion','tripod_translation']};
  else if(path==='/api/videos')result={id:'multi-video',width:800,height:400};
  else if(path==='/api/track'){payload=JSON.parse(options.body);result={id:'job'};}
  else result={status:'complete',result:{rows:[{x:399.5,y:199.5,accepted:true,camera_transform:[[1,0,0],[0,1,0],[0,0,1]]}],crop_box:payload.crop_box,start:0,fps:30,rejected:0}};
  return {ok:true,json:async()=>result};
 };
 try{
  await import('../dist/app.js?multiple-ui-test');await get('file').onchange();get('video').onloadeddata();
  get('cropToSelection').checked=false;get('cropToSelection').onchange();
  get('multiplePoints').checked=true;get('multiplePoints').onchange();
  assert.equal(get('pointsControls').hidden,false);assert.equal(get('track').disabled,true);
  get('sourceZoom').value='4';get('sourceZoom').oninput();click(400,200);click(400,200);
  assert.match(get('pointCount').textContent,/^1 reference point/);click(600,300);
  assert.match(get('pointCount').textContent,/^2 reference points/);assert.equal(get('track').disabled,false);
  assert.equal(get('pointMagnifier').hidden,false);
  get('source').onkeydown({key:'ArrowRight',altKey:true,preventDefault(){}});
  assert.match(get('pointCount').textContent,/450\.10/);
  get('source').onkeydown({key:'ArrowLeft',altKey:true,preventDefault(){}});
  get('cropToSelection').checked=true;get('cropToSelection').onchange();assert.equal(get('track').disabled,true);
  get('selectionTool').value='box';get('selectionTool').onchange();
  get('source').onpointerdown(pointer(300,150));get('source').onpointermove(pointer(500,250));get('source').onpointerup(pointer(500,250));
  assert.match(get('pointCount').textContent,/^2 reference points/);assert.equal(get('track').disabled,false);
  await get('track').onclick();assert.equal(payload.correct_rotation,false);assert.deepEqual(payload.points,[[399.5,199.5],[449.5,224.5]]);
  assert.deepEqual(payload.crop_box,[375,187.5,50,25]);assert.equal(payload.output_width,1920);assert.equal(payload.output_height,960);
  get('undoPoint').onclick();assert.match(get('pointCount').textContent,/^1 reference point/);assert.equal(get('track').disabled,true);
  get('clearPoints').onclick();assert.match(get('pointCount').textContent,/^0 reference points/);
 }finally{globalThis.document=originalDocument;globalThis.fetch=originalFetch;URL.revokeObjectURL(get('video').src);}
});
