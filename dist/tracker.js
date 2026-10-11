// Fixed-template normalized cross correlation. Coordinates use analysis pixels.
export function gray(rgba) {
 const out = new Float32Array(rgba.length / 4);
 for(let i=0;i<out.length;i++) out[i]=(rgba[i*4]*.299+rgba[i*4+1]*.587+rgba[i*4+2]*.114);
 return out;
}
export class Tracker {
 constructor(data,width,height,box) {
  this.width=width; this.height=height;
  this.x=box.x+box.w/2; this.y=box.y+box.h/2;
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
  const step=Math.max(1,Math.ceil(radius/12));
  const search=(cx,cy,r,s)=>{
   for(let dy=-r;dy<=r;dy+=s)for(let dx=-r;dx<=r;dx+=s){
    const x=cx+dx,y=cy+dy,confidence=this.score(data,x,y);
    if(confidence>best.confidence)best={x,y,confidence};
   }
  };
  search(ox,oy,radius,step); search(best.x,best.y,step,1);
  if(best.confidence>=.45){this.x=best.x;this.y=best.y;}
  return {x:this.x,y:this.y,confidence:best.confidence};
 }
}
