import {test} from 'node:test';
import assert from 'node:assert/strict';
import {sourcePoint,zoomAt,pointBox} from '../dist/viewport.js';
test('zoom stays anchored under the pointer',()=>{
 const view={x:1152,y:648,zoom:1},p={x:1800,y:200};
 const before=sourcePoint(p,view,2304,1296),after=sourcePoint(p,zoomAt(view,p,8,2304,1296),2304,1296);
 assert.deepEqual(before,after);
});
test('selection maps back to original pixels after pan and zoom',()=>{
 assert.deepEqual(sourcePoint({x:1100,y:400},{x:800,y:300,zoom:4},2000,1000),{x:825,y:275});
});
test('point selection clips safely at frame edges',()=>{
 assert.deepEqual(pointBox({x:2,y:3},24,100,80),{x:0,y:0,w:14,h:15});
});
