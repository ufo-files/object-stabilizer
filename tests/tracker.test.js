import {test} from 'node:test';
import assert from 'node:assert/strict';
import {Tracker,MotionSmoother,interpolateTrack,refineTrack} from '../dist/tracker.js';
function scene(dx=0,dy=0){const a=new Float32Array(100*80).fill(20);for(let y=20;y<36;y++)for(let x=30;x<46;x++)a[(y+dy)*100+x+dx]=40+((x*73+y*41+x*y*17)%190);return a;}
test('tracks a translated textured object',()=>{const t=new Tracker(scene(),100,80,{x:30,y:20,w:16,h:16});const r=t.update(scene(7,5),15);assert.ok(Math.abs(r.x-45)<1);assert.ok(Math.abs(r.y-33)<1);assert.ok(r.confidence>.99);});
test('holds position when the object disappears',()=>{const t=new Tracker(scene(),100,80,{x:30,y:20,w:16,h:16});const r=t.update(new Float32Array(8000).fill(20),15);assert.equal(r.x,38);assert.equal(r.y,28);assert.ok(r.confidence<.45);});
test('rejects textureless selections',()=>assert.throws(()=>new Tracker(new Float32Array(8000),100,80,{x:30,y:20,w:16,h:16}),/texture/));

test('finds a tiny target across a wide search radius',()=>{
 const make=(dx,dy)=>{const a=new Float32Array(160*120).fill(20);for(let y=50;y<54;y++)for(let x=70;x<74;x++)a[(y+dy)*160+x+dx]=60+((x*73+y*41)%190);return a;};
 const t=new Tracker(make(0,0),160,120,{x:67,y:47,w:10,h:10});
 let r=t.update(make(23,-17),40);assert.equal(r.accepted,false);
 t.update(make(23,-17),40);r=t.update(make(23,-17),40);assert.equal(r.accepted,true);assert.equal(r.x,95);assert.equal(r.y,35);assert.ok(r.confidence>.99);
});
test('preserves clicked anchor within a clipped edge patch',()=>{
 const t=new Tracker(scene(),100,80,{x:30,y:20,w:16,h:16},{x:32,y:22});
 const r=t.update(scene(7,5),15);assert.equal(r.x,39);assert.equal(r.y,27);
});

test('moderate-confidence false match cannot move the tracker',()=>{
 const t=new Tracker(scene(),100,80,{x:30,y:20,w:16,h:16});
 t.score=(_data,x,y)=>x===48&&y===28?.65:.2;
 const row=t.update(scene(),15);assert.equal(row.accepted,false);assert.equal(row.x,38);assert.equal(row.y,28);
});
test('recovery requires three consistent matches after loss',()=>{
 const t=new Tracker(scene(),100,80,{x:30,y:20,w:16,h:16});
 t.update(new Float32Array(8000).fill(20),15);
 assert.equal(t.update(scene(7,5),15).accepted,false);
 assert.equal(t.update(scene(7,5),15).accepted,false);
 assert.equal(t.update(scene(7,5),15).accepted,true);
});
test('isolated distant high-confidence matches are rejected',()=>{
 const t=new Tracker(scene(),100,80,{x:30,y:20,w:16,h:16});
 t.score=(_data,x,y)=>x===80&&y===28?1:.1;
 assert.equal(t.update(scene(),50).accepted,false);assert.equal(t.x,38);
 t.score=(_data,x,y)=>x===10&&y===28?1:.1;
 assert.equal(t.update(scene(),50).accepted,false);assert.equal(t.x,38);
});
test('smoother attenuates jitter and holds uncertain frames',()=>{
 const s=new MotionSmoother(0,0,15);
 const output=[];
 for(let i=0;i<20;i++)output.push(s.update({x:i%2?4:-4,y:0,confidence:1,accepted:true}).x);
 assert.ok(Math.max(...output)-Math.min(...output)<4);
 const before=s.x;s.update({x:100,y:100,confidence:.6,accepted:false});assert.equal(s.x,before);
});
test('reacquisition has a bounded output step and preserves raw coordinates',()=>{
 const s=new MotionSmoother(0,0,15,2);
 const row=s.update({x:100,y:100,confidence:1,accepted:true});
 assert.ok(Math.hypot(row.x,row.y)<=2.00001);assert.equal(row.rawX,100);
});
test('playback interpolates between smoothed samples',()=>{
 const rows=[{time:0,x:0,y:0},{time:1,x:10,y:20}];
 assert.deepEqual(interpolateTrack(rows,.5),{time:0,x:5,y:10});
 assert.equal(interpolateTrack(rows,-1),rows[0]);assert.equal(interpolateTrack(rows,2),rows[1]);
});

test('completed track removes filter lag during sustained motion at high zoom',()=>{
 const smoother=new MotionSmoother(0,0,15,2);
 const rows=Array.from({length:30},(_,i)=>({time:i/15,...smoother.update({x:i*6,y:i*3,confidence:1,accepted:true})}));
 assert.ok(rows.at(-1).rawX-rows.at(-1).x>50);
 const refined=refineTrack(rows,16);
 for(const row of refined)assert.ok(Math.hypot(row.x-row.rawX,row.y-row.rawY)<1e-6);
});
test('smoothing keeps accepted points within two output pixels of center',()=>{
 const rows=Array.from({length:20},(_,i)=>({time:i/15,x:i*2+(i%2?3:-3),y:10,confidence:1,accepted:true}));
 const refined=refineTrack(rows,8);
 for(let i=0;i<rows.length;i++)assert.ok(Math.hypot(refined[i].x-rows[i].x,refined[i].y-rows[i].y)*8<=2.00001);
});
test('short lost-track gap is reconstructed without trailing the moving subject',()=>{
 const rows=Array.from({length:6},(_,i)=>({time:i/15,x:i<1?0:i<4?0:i*4,y:0,confidence:i>0&&i<4?.3:1,accepted:!(i>0&&i<4)}));
 const refined=refineTrack(rows,4);
 for(let i=1;i<4;i++){assert.ok(Math.abs(refined[i].x-i*4)<1e-6);assert.equal(refined[i].reconstructed,true);}
});
