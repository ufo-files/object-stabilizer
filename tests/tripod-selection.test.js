import test from 'node:test';import assert from 'node:assert/strict';import {sourcePoint,selectionBox} from '../dist/selection.js';
test('drawn reference uses native pixels at different preview sizes',()=>{for(const scale of [.25,.5,1,2]){const rect={left:17,top:43,width:1920*scale,height:1080*scale};assert.deepEqual(sourcePoint(17+1296*scale,43+266*scale,rect,1920,1080),[1296,266]);}});
test('dragging in either direction selects the same reference',()=>{assert.deepEqual(selectionBox([1328,298],[1264,234]),[1264,234,64,64]);assert.deepEqual(selectionBox([1264,234],[1328,298]),[1264,234,64,64]);});
test('pointer outside preview clips to native frame bounds',()=>{assert.deepEqual(sourcePoint(-20,900,{left:0,top:0,width:960,height:540},1920,1080),[0,1080]);});
