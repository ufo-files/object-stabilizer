// Fixed-template normalized cross correlation. Coordinates use analysis pixels.
export function gray(rgba) {
 const out = new Float32Array(rgba.length / 4);
 for(let i=0;i<out.length;i++) out[i]=(rgba[i*4]*.299+rgba[i*4+1]*.587+rgba[i*4+2]*.114);
 return out;
}
export const MIN_CONFIDENCE=.72;
export class Tracker {
 constructor(data,width,height,box,anchor) {
  this.width=width; this.height=height;
  this.x=anchor?.x ?? box.x+box.w/2; this.y=anchor?.y ?? box.y+box.h/2;
  this.small=Math.min(box.w,box.h)<=32;
  this.patchSize=Math.max(box.w,box.h);this.pending=null;this.vx=0;this.vy=0;this.lost=0;
  this.points=[]; let sum=0;
  const step=Math.max(1,Math.ceil(Math.max(box.w,box.h)/24));
  for(let y=Math.ceil(box.y);y<box.y+box.h;y+=step)
   for(let x=Math.ceil(box.x);x<box.x+box.w;x+=step) {
    const value=data[y*width+x]; this.points.push({dx:x-this.x,dy:y-this.y,value}); sum+=value;
   }
  const mean=sum/this.points.length; let energy=0;
  for(const p of this.points){p.value-=mean;energy+=p.value*p.value;}
  if(this.points.length<9 || energy/this.points.length<4) throw Error('Choose a larger region with visible texture or edges.');
  this.energy=energy;
 }
 score(data,x,y) {
  let sum=0,squared=0,dot=0;
  for(const p of this.points){
   const px=Math.round(x+p.dx),py=Math.round(y+p.dy);
   if(px<0||py<0||px>=this.width||py>=this.height) return -1;
   const v=data[py*this.width+px];sum+=v;squared+=v*v;dot+=v*p.value;
  }
  const energy=squared-sum*sum/this.points.length;
  return energy>1?dot/Math.sqrt(this.energy*energy):-1;
 }
 update(data,radius=30) {
  const ox=this.x,oy=this.y; let best={x:ox,y:oy,confidence:this.score(data,ox,oy)};
  // Small subjects can fall between coarse search samples. Search every pixel.
  const step=this.small?1:Math.max(1,Math.min(3,Math.ceil(radius/12)));
  const search=(cx,cy,r,s)=>{
   for(let dy=-r;dy<=r;dy+=s)for(let dx=-r;dx<=r;dx+=s){
    const x=cx+dx,y=cy+dy,confidence=this.score(data,x,y);
    if(confidence>best.confidence)best={x,y,confidence};
   }
  };
  search(ox,oy,Math.ceil(radius),step); if(step>1)search(best.x,best.y,step,1);
  let accepted=false;
  const distance=Math.hypot(best.x-ox,best.y-oy);
  const jumpLimit=Math.max(24,this.patchSize*2,Math.hypot(this.vx,this.vy)*3+8);
  if(best.confidence>=MIN_CONFIDENCE){
   if(this.lost===0&&distance<=jumpLimit){accepted=true;this.pending=null;}
   else{
    const tolerance=Math.max(3,this.patchSize/2);
    const consistent=this.pending&&Math.hypot(best.x-this.pending.x,best.y-this.pending.y)<=tolerance;
    this.pending={x:best.x,y:best.y,count:consistent?this.pending.count+1:1};
    accepted=this.pending.count>=3;
   }
  }else this.pending=null;
  if(accepted){this.vx=best.x-ox;this.vy=best.y-oy;this.x=best.x;this.y=best.y;this.lost=0;this.pending=null;}
  else{this.lost++;this.vx=0;this.vy=0;}
  return {x:this.x,y:this.y,confidence:best.confidence,accepted};
 }
}

// Smooth only accepted observations; uncertain frames hold the rendered position.
export class MotionSmoother {
 constructor(x,y,fps,maxStep=12){this.x=x;this.y=y;this.alpha=1-Math.exp(-1/(fps*.16));this.maxStep=maxStep;}
 update(row){
  if(row.accepted!==false&&row.confidence>=MIN_CONFIDENCE){
   let dx=(row.x-this.x)*this.alpha,dy=(row.y-this.y)*this.alpha;
   const distance=Math.hypot(dx,dy),scale=distance>this.maxStep?this.maxStep/distance:1;
   this.x+=dx*scale;this.y+=dy*scale;
  }
  return {...row,rawX:row.x,rawY:row.y,x:this.x,y:this.y};
 }
}
export function interpolateTrack(rows,time){
 if(time<=rows[0].time)return rows[0];
 if(time>=rows.at(-1).time)return rows.at(-1);
 let lo=0,hi=rows.length-1;
 while(hi-lo>1){const mid=(lo+hi)>>1;if(rows[mid].time<=time)lo=mid;else hi=mid;}
 const a=rows[lo],b=rows[hi],t=(time-a.time)/(b.time-a.time);
 return {...a,x:a.x+(b.x-a.x)*t,y:a.y+(b.y-a.y)*t};
}
