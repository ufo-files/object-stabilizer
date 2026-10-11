import {test} from 'node:test';
import assert from 'node:assert/strict';
import {Tracker} from '../dist/tracker.js';
function scene(dx=0,dy=0){const a=new Float32Array(100*80).fill(20);for(let y=20;y<36;y++)for(let x=30;x<46;x++)a[(y+dy)*100+x+dx]=40+((x*73+y*41+x*y*17)%190);return a;}
test('tracks a translated textured object',()=>{const t=new Tracker(scene(),100,80,{x:30,y:20,w:16,h:16});const r=t.update(scene(7,5),15);assert.ok(Math.abs(r.x-45)<1);assert.ok(Math.abs(r.y-33)<1);assert.ok(r.confidence>.99);});
test('holds position when the object disappears',()=>{const t=new Tracker(scene(),100,80,{x:30,y:20,w:16,h:16});const r=t.update(new Float32Array(8000).fill(20),15);assert.equal(r.x,38);assert.equal(r.y,28);assert.ok(r.confidence<.45);});
test('rejects textureless selections',()=>assert.throws(()=>new Tracker(new Float32Array(8000),100,80,{x:30,y:20,w:16,h:16}),/texture/));

test('finds a tiny target across a wide search radius',()=>{
 const make=(dx,dy)=>{const a=new Float32Array(160*120).fill(20);for(let y=50;y<54;y++)for(let x=70;x<74;x++)a[(y+dy)*160+x+dx]=60+((x*73+y*41)%190);return a;};
 const t=new Tracker(make(0,0),160,120,{x:67,y:47,w:10,h:10});
 const r=t.update(make(23,-17),40);assert.equal(r.x,95);assert.equal(r.y,35);assert.ok(r.confidence>.99);
});
test('preserves clicked anchor within a clipped edge patch',()=>{
 const t=new Tracker(scene(),100,80,{x:30,y:20,w:16,h:16},{x:32,y:22});
 const r=t.update(scene(7,5),15);assert.equal(r.x,39);assert.equal(r.y,27);
});
