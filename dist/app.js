import {sampleTrack} from './playback.js';
import {request,waitForJob} from './local-api.js';
import {sourcePoint,zoomAt,pointBox} from './viewport.js';
const $=id=>document.getElementById(id), video=$('video'), source=$('source'), output=$('output');
const ctx=source.getContext('2d'), out=output.getContext('2d');
const analysis={width:0,height:0};
let uploadedId=null,trackingJob=null;
let focus=null,anchor=null,view={x:480,y:270,zoom:1};
let box=null,drag=null,rows=[],start=0,busy=false,cancelled=false,url=null,playing=false,downloadUrl=null;
const status=t=>$('status').textContent=t;
const controls=()=>{ $('file').disabled=busy; $('seek').disabled=busy||!url; $('track').disabled=busy||!box; for(const id of ['preview','export','json'])$(id).disabled=busy||!rows.length; $('cancel').hidden=!busy; for(const id of ['radius','trackingMode','mode','patch','sourceZoom','fit','zoomIn','zoomOut'])$(id).disabled=busy; };
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

function centered(row){
 const zoom=view.zoom,x=row.x*video.videoWidth/analysis.width,y=row.y*video.videoHeight/analysis.height;
 out.fillStyle='#000';out.fillRect(0,0,output.width,output.height);
 // OpenCV pixel centers are integer coordinates; Canvas image centers are n + 0.5.
 const pixelOffset=Number.isFinite(row.confidence)?.5:0;
 out.save();out.translate(output.width/2+pixelOffset,output.height/2+pixelOffset);out.scale(zoom,zoom);out.drawImage(video,-x-pixelOffset,-y-pixelOffset);out.restore();
 $('confidence').textContent=Number.isFinite(row.confidence)?`Match ${Math.round(row.confidence*100)}%${row.accepted===false||row.confidence<.55?(row.reconstructed?' · estimated between reliable matches':' · uncertain, position held'):''} · ${view.zoom.toFixed(2)}×`:`${view.zoom.toFixed(2)}× · same zoom as input`;
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
function stop(){playing=false;video.pause();}
$('file').onchange=async()=>{
 stop();if(url)URL.revokeObjectURL(url);box=null;reset();
 const file=$('file').files[0];if(!file)return;
 url=URL.createObjectURL(file);video.src=url;anchor=null;focus=null;uploadedId=null;trackingJob=null;
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
$('cancel').onclick=()=>{const wasPlaying=playing;cancelled=true;stop();if(wasPlaying){busy=false;controls();status('Preview stopped.');}};
$('track').onclick=async()=>{
 busy=true;cancelled=false;rows=[];trackingJob=null;controls();
 try{
  status('Connecting to local Python tracker…');
  let health;try{health=await request('/api/health');}catch{throw Error('Run the Python app locally, then open http://127.0.0.1:8080. GitHub Pages cannot run Python. See the repo README for setup.');}
  if(health.engine!=='opencv')throw Error('This page is not connected to the Python tracker.');
  if(!uploadedId){
   status('Loading video into the local service…');
   const form=new FormData();form.append('file',$('file').files[0]);
   const uploaded=await request('/api/videos',{method:'POST',body:form});uploadedId=uploaded.id;
   if(uploaded.width!==video.videoWidth||uploaded.height!==video.videoHeight){uploadedId=null;throw Error('Browser and Python disagree on video orientation. Convert this video to a standard MP4 first.');}
  }
  if(cancelled)throw Error('Cancelled.');
  const target=anchor||{x:box.x+box.w/2,y:box.y+box.h/2};
  const job=await request('/api/track',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({video_id:uploadedId,point:[Math.max(0,target.x-.5),Math.max(0,target.y-.5)],start,size:Math.max(8,Math.min(128,Math.round(Math.max(box.w,box.h)))),radius:Math.max(5,Math.min(500,Number($('radius').value)||100)),mode:$('trackingMode').value})});
  const result=await waitForJob(job.id,(progress,message)=>{$('progress').value=progress;status(message);},()=>cancelled);
  rows=result.rows;start=result.start;trackingJob=job.id;
  await seek(start);paint();centered(rows[0]);
  status(`Tracked ${rows.length} source frames at ${result.fps.toFixed(2)} fps. ${result.rejected} uncertain frames.${result.rejected?' Inspect flagged frames before exporting.':''}`);
 }catch(e){rows=[];status(e.message);}finally{busy=false;controls();}
};

function rowAt(t){return sampleTrack(rows,t);}
function loop(_now,metadata){if(!playing)return;paint();centered(rowAt(metadata?.mediaTime??video.currentTime));if(video.ended){finishPlayback();return;}if(video.requestVideoFrameCallback)video.requestVideoFrameCallback(loop);else requestAnimationFrame(loop);}
function finishPlayback(){playing=false;video.pause();busy=false;controls();status('Preview finished.');}
video.onended=finishPlayback;
async function play(){await seek(start);paint();centered(rows[0]);playing=true;await video.play();loop();}
$('preview').onclick=async()=>{busy=true;cancelled=false;controls();try{status('Playing centered preview…');await play();}catch(e){playing=false;busy=false;controls();status(e.message);}};
for(const id of ['trackingMode','radius'])$(id).onchange=()=>{reset();status('Settings changed. Track the selection again.');};
function download(blob,name){if(downloadUrl)URL.revokeObjectURL(downloadUrl);downloadUrl=URL.createObjectURL(blob);const a=document.createElement('a');a.href=downloadUrl;a.download=name;a.click();}
$('json').onclick=()=>download(new Blob([JSON.stringify({source:$('file').files[0].name,sourceWidth:video.videoWidth,sourceHeight:video.videoHeight,analysisWidth:analysis.width,analysisHeight:analysis.height,selection:box,anchor,zoom:view.zoom,engine:'opencv',start,rows},null,2)],{type:'application/json'}),'tracking.json');
$('export').onclick=async()=>{
 if(!trackingJob){status('Track with the local Python service before exporting.');return;}
 busy=true;cancelled=false;controls();
 try{
  const job=await request(`/api/jobs/${trackingJob}/export`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({zoom:view.zoom})});
  const result=await waitForJob(job.id,(progress,message)=>{$('progress').value=progress;status(message);},()=>cancelled);
  const link=document.createElement('a');link.href=result.download;link.download='stabilized.mp4';link.click();
  status(`MP4 export complete: ${result.frames} frames at ${result.fps.toFixed(2)} fps.`);
 }catch(e){status(e.message);}finally{busy=false;controls();}
};
