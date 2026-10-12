import {Tracker,gray,MotionSmoother,MIN_CONFIDENCE,interpolateTrack,refineTrack} from './tracker.js';
import {sourcePoint,zoomAt,pointBox} from './viewport.js';
const $=id=>document.getElementById(id), video=$('video'), source=$('source'), output=$('output');
const ctx=source.getContext('2d'), out=output.getContext('2d');
const analysis=document.createElement('canvas'), ac=analysis.getContext('2d',{willReadFrequently:true});
let focus=null,anchor=null,view={x:480,y:270,zoom:1};
let box=null,drag=null,rows=[],start=0,busy=false,cancelled=false,url=null,playing=false,recorder=null,downloadUrl=null;
const status=t=>$('status').textContent=t;
const controls=()=>{ $('file').disabled=busy; $('seek').disabled=busy||!url; $('track').disabled=busy||!box; for(const id of ['preview','export','json'])$(id).disabled=busy||!rows.length; $('cancel').hidden=!busy; for(const id of ['fps','radius','mode','patch','sourceZoom','fit','zoomIn','zoomOut'])$(id).disabled=busy; };
function reset(){rows=[];$('progress').value=0;$('confidence').textContent='—';controls();}
function paint(){
 if(!video.videoWidth)return;
 ctx.fillStyle='#0a1011';ctx.fillRect(0,0,source.width,source.height);
 ctx.save();ctx.translate(source.width/2,source.height/2);ctx.scale(view.zoom,view.zoom);ctx.translate(-view.x,-view.y);
 ctx.drawImage(video,0,0,source.width,source.height);
 if(box){ctx.strokeStyle='#c0f397';ctx.lineWidth=Math.max(2,source.width/500)/view.zoom;ctx.strokeRect(box.x,box.y,box.w,box.h);}
 if(anchor){const r=10*source.width/source.clientWidth/view.zoom;ctx.beginPath();ctx.moveTo(anchor.x-r,anchor.y);ctx.lineTo(anchor.x+r,anchor.y);ctx.moveTo(anchor.x,anchor.y-r);ctx.lineTo(anchor.x,anchor.y+r);ctx.strokeStyle='#c0f397';ctx.lineWidth=2*source.width/source.clientWidth/view.zoom;ctx.stroke();}
 ctx.restore();
 $('time').textContent=video.currentTime.toFixed(2)+' s';
}
function read(){ac.drawImage(video,0,0,analysis.width,analysis.height);return gray(ac.getImageData(0,0,analysis.width,analysis.height).data);}
function centered(row){
 const zoom=view.zoom,x=row.x*video.videoWidth/analysis.width,y=row.y*video.videoHeight/analysis.height;
 out.fillStyle='#000';out.fillRect(0,0,output.width,output.height);
 out.save();out.translate(output.width/2,output.height/2);out.scale(zoom,zoom);out.drawImage(video,-x,-y);out.restore();
 $('confidence').textContent=Number.isFinite(row.confidence)?`Match ${Math.round(row.confidence*100)}%${row.accepted===false||row.confidence<MIN_CONFIDENCE?(row.reconstructed?' · estimated between reliable matches':' · uncertain, position held'):''} · ${view.zoom.toFixed(2)}×`:`${view.zoom.toFixed(2)}× · same zoom as input`;
}
function refreshOutput(){
 if(!video.videoWidth||video.readyState<2)return;
 const target=anchor||(box?{x:box.x+box.w/2,y:box.y+box.h/2}:view);
 centered(rows.length?rowAt(video.currentTime):{x:target.x,y:target.y,confidence:null});
}
function seek(t){return new Promise((resolve,reject)=>{
 if(Math.abs(video.currentTime-t)<.00001&&video.readyState>=2){resolve();return;}
 const timer=setTimeout(()=>finish(Error('Video seek timed out. Try another browser or video format.')),15000);
 const done=()=>finish(),error=()=>finish(Error('Video decoding failed.'));
 function finish(err){clearTimeout(timer);video.removeEventListener('seeked',done);video.removeEventListener('error',error);err?reject(err):resolve();}
 video.addEventListener('seeked',done,{once:true});video.addEventListener('error',error,{once:true});video.currentTime=t;
 });}
function stop(){playing=false;video.pause();if(recorder&&recorder.state!=='inactive')recorder.stop();}
$('file').onchange=async()=>{
 stop();if(url)URL.revokeObjectURL(url);box=null;reset();
 const file=$('file').files[0];if(!file)return;
 url=URL.createObjectURL(file);video.src=url;anchor=null;focus=null;
 status('Loading video…');
};
video.onloadeddata=()=>{
 if(!Number.isFinite(video.duration)||video.duration<=0){status('This video does not have a usable duration.');return;}
 source.width=output.width=video.videoWidth;source.height=output.height=video.videoHeight;
 analysis.width=video.videoWidth;analysis.height=video.videoHeight;
 view={x:source.width/2,y:source.height/2,zoom:1};$('sourceZoom').value=1;$('sourceZoomValue').textContent='1×';
 source.style.aspectRatio=output.style.aspectRatio=`${video.videoWidth} / ${video.videoHeight}`;
 $('seek').max=Math.max(0,video.duration-.001);$('seek').value=0;$('empty').hidden=true;$('empty').style.display='none';
 $('details').textContent=`${$('file').files[0].name} · ${video.videoWidth} × ${video.videoHeight} · ${video.duration.toFixed(2)} s`;
 paint();controls();status('Zoom in and click the object you want to keep centered.');
};
video.onerror=()=>status('Your browser cannot decode this video. Try an MP4 with H.264 encoding.');
$('seek').oninput=async()=>{stop();box=null;anchor=null;reset();$('selection').textContent='No selection';try{await seek(Number($('seek').value));paint();}catch(e){status(e.message);}};
function canvasPoint(e){const r=source.getBoundingClientRect();return {x:(e.clientX-r.left)*source.width/r.width,y:(e.clientY-r.top)*source.height/r.height};}
function point(e){const p=sourcePoint(canvasPoint(e),view,source.width,source.height);return {x:Math.max(0,Math.min(source.width-1,p.x)),y:Math.max(0,Math.min(source.height-1,p.y))};}
function setZoom(zoom,at={x:source.width/2,y:source.height/2}){
 view=zoomAt(view,at,Math.max(1,Math.min(16,zoom)),source.width,source.height);
 $('sourceZoom').value=view.zoom;$('sourceZoomValue').textContent=`${view.zoom.toFixed(2).replace(/\.?0+$/,'')}×`;paint();refreshOutput();
}
source.addEventListener('wheel',e=>{if(busy||!url)return;e.preventDefault();focus=point(e);setZoom(view.zoom*Math.exp(-e.deltaY*.002),canvasPoint(e));},{passive:false});
function zoomToTarget(zoom){
 const target=anchor||focus||{x:view.x,y:view.y};
 view.x=target.x;view.y=target.y;
 setZoom(zoom);
}
$('sourceZoom').oninput=()=>zoomToTarget(Number($('sourceZoom').value));
$('zoomIn').onclick=()=>zoomToTarget(view.zoom*1.5);
$('zoomOut').onclick=()=>zoomToTarget(view.zoom/1.5);
source.ondblclick=e=>{if(busy||!url)return;focus=point(e);view.x=focus.x;view.y=focus.y;setZoom(view.zoom*2);};
$('fit').onclick=()=>{view={x:source.width/2,y:source.height/2,zoom:1};$('sourceZoom').value=1;$('sourceZoomValue').textContent='1×';paint();refreshOutput();};
function selectPoint(p){focus=p;anchor=p;box=pointBox(p,Math.max(8,Math.min(128,Number($('patch').value)||24)),source.width,source.height);}
function selectionReady(){start=video.currentTime;$('selection').textContent=`${anchor?'Point · ':''}${Math.round(box.w)} × ${Math.round(box.h)} px`;status('Selection ready. Track the object to create a centered preview.');paint();refreshOutput();controls();}
source.onpointerdown=e=>{
 if(busy||!url||video.readyState<2)return;stop();
 const pan=e.shiftKey||e.button===2||$('mode').value==='pan';
 drag={p:point(e),screen:canvasPoint(e),clientX:e.clientX,clientY:e.clientY,view:{...view},pan};
 if(!pan){reset();box=null;anchor=null;}
 source.setPointerCapture(e.pointerId);
};
source.oncontextmenu=e=>e.preventDefault();
source.onpointermove=e=>{
 if(!drag){if(!busy&&url)focus=point(e);return;}
 if(drag.pan){const p=canvasPoint(e);view.x=drag.view.x-(p.x-drag.screen.x)/view.zoom;view.y=drag.view.y-(p.y-drag.screen.y)/view.zoom;paint();return;}
 if($('mode').value==='box'){const p=point(e);box={x:Math.min(drag.p.x,p.x),y:Math.min(drag.p.y,p.y),w:Math.abs(drag.p.x-p.x),h:Math.abs(drag.p.y-p.y)};paint();}
};
source.onpointerup=e=>{
 if(!drag)return;const d=drag;drag=null;if(d.pan)return;
 if($('mode').value==='point'){if(Math.hypot(e.clientX-d.clientX,e.clientY-d.clientY)>6){status('Click without dragging to select a point. Shift-drag to pan.');controls();return;}selectPoint(point(e));}
 if(!box||box.w<6||box.h<6){box=null;status('Select a point or draw a box at least 6 × 6 pixels.');paint();controls();return;}
 selectionReady();
};
source.onpointercancel=()=>{drag=null;paint();controls();};
$('patch').onchange=()=>{if(anchor){reset();selectPoint(anchor);selectionReady();}};
$('mode').onchange=()=>{source.style.cursor=$('mode').value==='pan'?'grab':'crosshair';};
$('cancel').onclick=()=>{const wasPlaying=playing;cancelled=true;stop();if(wasPlaying&&!recorder){busy=false;controls();status('Preview stopped.');}};
$('track').onclick=async()=>{
 busy=true;cancelled=false;rows=[];controls();
 try{
  await seek(start);const scale=analysis.width/source.width;
  const tracker=new Tracker(read(),analysis.width,analysis.height,{x:box.x*scale,y:box.y*scale,w:box.w*scale,h:box.h*scale},anchor?{x:anchor.x*scale,y:anchor.y*scale}:undefined);
  const fps=Number($('fps').value),radius=Math.max(1,Math.min(500,Number($('radius').value)||100)*scale);
  const smoother=new MotionSmoother(tracker.x,tracker.y,fps,Math.max(2,24/view.zoom));
  const count=Math.ceil((video.duration-start)*fps);
  for(let i=0;i<count;i++){
   if(cancelled)break;
   const t=start+i/fps;await seek(t);if(cancelled)break;
   const row=smoother.update(i===0?{x:tracker.x,y:tracker.y,confidence:1,accepted:true}:tracker.update(read(),radius));
   rows.push({time:t,...row});paint();centered(row);$('progress').value=(i+1)/count;
   status(`Tracking frame ${i+1} / ${count}`);
   await new Promise(resolve=>setTimeout(resolve,0));
  }
  if(cancelled){rows=[];status('Tracking cancelled.');}
  else{rows=refineTrack(rows,view.zoom);const uncertain=rows.filter(r=>r.accepted===false).length;status(`Tracked ${rows.length} frames. ${uncertain} uncertain matches.${uncertain?' Inspect the preview before exporting.':''}`);await seek(start);paint();centered(rows[0]);}
 }catch(e){rows=[];status(e.message);}finally{busy=false;controls();}
};
function rowAt(t){return interpolateTrack(rows,t);}
function loop(){if(!playing)return;paint();centered(rowAt(video.currentTime));if(video.ended){finishPlayback();return;}if(video.requestVideoFrameCallback)video.requestVideoFrameCallback(loop);else requestAnimationFrame(loop);}
function finishPlayback(){playing=false;video.pause();if(recorder&&recorder.state!=='inactive')recorder.stop();else{busy=false;controls();status('Preview finished.');}}
video.onended=finishPlayback;
async function play(){await seek(start);paint();centered(rows[0]);playing=true;await video.play();loop();}
$('preview').onclick=async()=>{busy=true;cancelled=false;controls();try{status('Playing centered preview…');await play();}catch(e){playing=false;busy=false;controls();status(e.message);}};
for(const id of ['fps','radius'])$(id).onchange=()=>{reset();status('Settings changed. Track the selection again.');};
function download(blob,name){if(downloadUrl)URL.revokeObjectURL(downloadUrl);downloadUrl=URL.createObjectURL(blob);const a=document.createElement('a');a.href=downloadUrl;a.download=name;a.click();}
$('json').onclick=()=>download(new Blob([JSON.stringify({source:$('file').files[0].name,sourceWidth:video.videoWidth,sourceHeight:video.videoHeight,analysisWidth:analysis.width,analysisHeight:analysis.height,selection:box,anchor,zoom:view.zoom,fps:Number($('fps').value),start,rows},null,2)],{type:'application/json'}),'tracking.json');
$('export').onclick=async()=>{
 if(!window.MediaRecorder||!output.captureStream){status('Video export is unavailable in this browser. You can save tracking data instead.');return;}
 const mime=['video/webm;codecs=vp9','video/webm;codecs=vp8','video/mp4','video/webm'].find(t=>MediaRecorder.isTypeSupported(t));
 if(!mime){status('No supported export encoder found.');return;}
 busy=true;cancelled=false;controls();let stream;
 try{
  await seek(start);centered(rows[0]);stream=output.captureStream(Number($('fps').value));
  recorder=new MediaRecorder(stream,{mimeType:mime,videoBitsPerSecond:12000000});const chunks=[];
  recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data);};
  recorder.onstop=()=>{stream.getTracks().forEach(t=>t.stop());if(!cancelled&&chunks.length){download(new Blob(chunks,{type:mime}),`stabilized.${mime.includes('mp4')?'mp4':'webm'}`);status('Export complete.');}else status('Export cancelled.');recorder=null;busy=false;controls();};
  recorder.onerror=()=>{cancelled=true;stop();status('The browser encoder failed.');};
  recorder.start(1000);status('Recording output… keep this tab visible.');await play();
 }catch(e){cancelled=true;stop();stream?.getTracks().forEach(t=>t.stop());busy=false;controls();status(e.message);}
};
document.addEventListener('visibilitychange',()=>{if(document.hidden&&recorder){cancelled=true;stop();status('Export cancelled because the tab was hidden.');}});
