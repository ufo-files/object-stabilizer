import {test} from 'node:test';
import assert from 'node:assert/strict';
import {sampleTrack} from '../dist/playback.js';
test('preview selects the actual source frame during timestamp gaps',()=>{
 const rows=[{time:0,x:10,y:20},{time:.2,x:100,y:200},{time:.267,x:110,y:210}];
 assert.equal(sampleTrack(rows,.15),rows[0]);
 assert.equal(sampleTrack(rows,.2),rows[1]);
 assert.equal(sampleTrack(rows,.25),rows[1]);
 assert.equal(sampleTrack(rows,.267),rows[2]);
 assert.equal(sampleTrack(rows,5),rows[2]);
});
