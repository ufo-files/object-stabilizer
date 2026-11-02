import {test} from 'node:test';
import assert from 'node:assert/strict';
import {brightCenter} from '../dist/centering.js';
function image(width,height,light){
 const data=new Uint8ClampedArray(width*height*4);
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){
  const value=light(x,y),i=(y*width+x)*4;data[i]=data[i+1]=data[i+2]=value;data[i+3]=255;
 }
 return {data,width,height};
}
test('measures the rendered subpixel offset of a changing bright body',()=>{
 for(const [x,y] of [[32,24],[31.3,24.7],[34.2,21.4]]){
  const result=brightCenter(image(64,48,(px,py)=>10+180*Math.exp(-((px-x)**2+(py-y)**2)/8)));
  assert.ok(Math.abs(result.x-(x-32))<.12);assert.ok(Math.abs(result.y-(y-24))<.12);
 }
});
test('does not apply a centering correction to a blank or low-contrast frame',()=>{
 assert.equal(brightCenter(image(64,48,()=>10)),null);
 assert.equal(brightCenter(image(64,48,(x,y)=>10+Math.exp(-((x-32)**2+(y-24)**2)/8))),null);
});

test('reference-core centering ignores an offset dimmer halo',()=>{
 const pixels=image(64,48,(x,y)=>10+180*Math.exp(-((x-32)**2+(y-24)**2)/2.8)+120*Math.exp(-((x-32)**2+(y-29)**2)/2.8));
 const body=brightCenter(pixels),core=brightCenter(pixels,.8);
 assert.ok(body.y>1,'whole-body measurement is pulled toward the halo');
 assert.ok(Math.abs(core.y)<.15,'compact bright reference remains at its original position');
});
