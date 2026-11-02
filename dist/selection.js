export function sourcePoint(clientX,clientY,rect,width,height){
  return [Math.max(0,Math.min(width,(clientX-rect.left)*width/rect.width)),Math.max(0,Math.min(height,(clientY-rect.top)*height/rect.height))];
}
export function selectionBox(start,end){return [Math.min(start[0],end[0]),Math.min(start[1],end[1]),Math.abs(end[0]-start[0]),Math.abs(end[1]-start[1])];}
