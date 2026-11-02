// Match the visible bright body after the browser's own video sampling.
export function brightCenter(image,coreFraction=.2){
 const {data,width,height}=image,gray=new Float32Array(width*height),border=[];
 let peak=0;
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){
  const i=y*width+x,j=i*4,value=.299*data[j]+.587*data[j+1]+.114*data[j+2];
  gray[i]=value;peak=Math.max(peak,value);
  if(!x||!y||x===width-1||y===height-1)border.push(value);
 }
 border.sort((a,b)=>a-b);const background=border[border.length>>1];
 const deviation=border.map(value=>Math.abs(value-background)).sort((a,b)=>a-b);
 const noise=Math.max(.5,deviation[deviation.length>>1]*1.4826);
 if(peak-background<Math.max(2,noise*5))return null;
 const threshold=background+(peak-background)*coreFraction;
 let mass=0,xs=0,ys=0;
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){
  const weight=Math.max(0,gray[y*width+x]-threshold);mass+=weight;xs+=x*weight;ys+=y*weight;
 }
 return mass?{x:xs/mass-width/2,y:ys/mass-height/2,contrast:peak-background}:null;
}
