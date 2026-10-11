import {Tracker,gray} from './tracker.js';
const $=id=>document.getElementById(id), video=$('video'), source=$('source'), output=$('output');
const ctx=source.getContext('2d'), out=output.getContext('2d');
const analysis=document.createElement('canvas'), ac=analysis.getContext('2d',{willReadFrequently:true});
let box=null,drag=null,rows=[],start=0,busy=false,cancelled=false,url=null,playing=false,recorder=null,downloadUrl=null;
const status=t=>$('status').textContent=t;
const controls=()=>{ $('file').disabled=busy; $('seek').disabled=busy||!url; $('track').disabled=busy||!box; for(const id of ['preview','export','json'])$(id).disabled=busy||!rows.length; $('cancel').hidden=!busy; for(const id of ['fps','radius','zoom'])$(id).disabled=busy; };
function reset(){rows=[];$('progress').value=0;$('confidence').textContent='—';controls();}
function paint(){
 if(!video.videoWidth)return;
 ctx.drawImage(video,0,0,source.width,source.height);
 if(box){ctx.strokeStyle='#c0f397';ctx.lineWidth=Math.max(2,source.width/500);ctx.strokeRect(box.x,box.y,box.w,box.h);}
 $('time').textContent=video.currentTime.toFixed(2)+' s';
}
function read(){ac.drawImage(video,0,0,analysis.width,analysis.height);return gray(ac.getImageData(0,0,analysis.width,analysis.height).data);}
function centered(row){
 const zoom=Number($('zoom').value),x=row.x*video.videoWidth/analysis.width,y=row.y*video.videoHeight/analysis.height;
 out.fillStyle='#000';out.fillRect(0,0,output.width,output.height);
 out.save();out.translate(output.width/2,output.height/2);out.scale(zoom,zoom);out.drawImage(video,-x,-y);out.restore();
 $('confidence').textContent=`Match ${Math.round(row.confidence*100)}%${row.confidence<.45?' · uncertain, position held':''}`;
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
 url=URL.createObjectURL(file);video.src=url;
 status('Loading video…');
};
video.onloadeddata=()=>{
 if(!Number.isFinite(video.duration)||video.duration<=0){status('This video does not have a usable duration.');return;}
 source.width=output.width=video.videoWidth;source.height=output.height=video.videoHeight;
 analysis.width=Math.min(960,video.videoWidth);analysis.height=Math.round(video.videoHeight*analysis.width/video.videoWidth);
 $('seek').max=Math.max(0,video.duration-.001);$('seek').value=0;$('empty').hidden=true;$('empty').style.display='none';
 $('details').textContent=`${$('file').files[0].name} · ${video.videoWidth} × ${video.videoHeight} · ${video.duration.toFixed(2)} s`;
 paint();controls();status('Drag a box around the subject.');
};
video.onerror=()=>status('Your browser cannot decode this video. Try an MP4 with H.264 encoding.');
$('seek').oninput=async()=>{stop();box=null;reset();$('selection').textContent='No selection';try{await seek(Number($('seek').value));paint();}catch(e){status(e.message);}};
function point(e){const r=source.getBoundingClientRect();return {x:Math.max(0,Math.min(source.width,(e.clientX-r.left)*source.width/r.width)),y:Math.max(0,Math.min(source.height,(e.clientY-r.top)*source.height/r.height))};}
source.onpointerdown=e=>{if(busy||!url||video.readyState<2)return;stop();reset();drag=point(e);box=null;source.setPointerCapture(e.pointerId);};
source.onpointermove=e=>{if(!drag)return;const p=point(e);box={x:Math.min(drag.x,p.x),y:Math.min(drag.y,p.y),w:Math.abs(drag.x-p.x),h:Math.abs(drag.y-p.y)};paint();};
source.onpointerup=()=>{if(!drag)return;drag=null;if(!box||box.w<6||box.h<6){box=null;status('Draw a box at least 6 × 6 pixels.');}else{start=video.currentTime;$('selection').textContent=`${Math.round(box.w)} × ${Math.round(box.h)} px`;status('Selection ready. Track the object to create a centered preview.');}controls();};
source.onpointercancel=()=>{drag=null;box=null;paint();controls();};
$('cancel').onclick=()=>{cancelled=true;stop();if(!recorder&&rows.length){busy=false;controls();status('Preview stopped.');}};
$('track').onclick=async()=>{
 busy=true;cancelled=false;rows=[];controls();
 try{
  await seek(start);const scale=analysis.width/source.width;
  const tracker=new Tracker(read(),analysis.width,analysis.height,{x:box.x*scale,y:box.y*scale,w:box.w*scale,h:box.h*scale});
  const fps=Number($('fps').value),radius=Math.max(1,Math.min(500,Number($('radius').value)||100)*scale);
  const count=Math.ceil((video.duration-start)*fps);
  for(let i=0;i<count;i++){
   if(cancelled)break;
   const t=start+i/fps;await seek(t);if(cancelled)break;
   const row=i===0?{x:tracker.x,y:tracker.y,confidence:1}:tracker.update(read(),radius);
   rows.push({time:t,...row});paint();centered(row);$('progress').value=(i+1)/count;
   status(`Tracking frame ${i+1} / ${count}`);
   await new Promise(resolve=>setTimeout(resolve,0));
  }
  if(cancelled){rows=[];status('Tracking cancelled.');}
  else{const uncertain=rows.filter(r=>r.confidence<.45).length;status(`Tracked ${rows.length} frames. ${uncertain} uncertain matches.${uncertain?' Inspect the preview before exporting.':''}`);await seek(start);paint();centered(rows[0]);}
 }catch(e){rows=[];status(e.message);}finally{busy=false;controls();}
};
function rowAt(t){const i=Math.min(rows.length-1,Math.max(0,Math.round((t-start)*Number($('fps').value))));return rows[i];}
function loop(){if(!playing)return;paint();centered(rowAt(video.currentTime));if(video.ended){finishPlayback();return;}if(video.requestVideoFrameCallback)video.requestVideoFrameCallback(loop);else requestAnimationFrame(loop);}
function finishPlayback(){playing=false;video.pause();if(recorder&&recorder.state!=='inactive')recorder.stop();else{busy=false;controls();status('Preview finished.');}}
video.onended=finishPlayback;
async function play(){await seek(start);paint();centered(rows[0]);playing=true;await video.play();loop();}
$('preview').onclick=async()=>{busy=true;cancelled=false;controls();try{status('Playing centered preview…');await play();}catch(e){playing=false;busy=false;controls();status(e.message);}};
$('zoom').oninput=()=>{if(rows.length)centered(rowAt(video.currentTime));};
for(const id of ['fps','radius'])$(id).onchange=()=>{reset();status('Settings changed. Track the selection again.');};
function download(blob,name){if(downloadUrl)URL.revokeObjectURL(downloadUrl);downloadUrl=URL.createObjectURL(blob);const a=document.createElement('a');a.href=downloadUrl;a.download=name;a.click();}
$('json').onclick=()=>download(new Blob([JSON.stringify({source:$('file').files[0].name,sourceWidth:video.videoWidth,sourceHeight:video.videoHeight,analysisWidth:analysis.width,analysisHeight:analysis.height,selection:box,fps:Number($('fps').value),start,rows},null,2)],{type:'application/json'}),'tracking.json');
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
