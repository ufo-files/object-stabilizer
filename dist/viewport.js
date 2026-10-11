export function sourcePoint(point,view,width,height) {
 return {x:view.x+(point.x-width/2)/view.zoom,y:view.y+(point.y-height/2)/view.zoom};
}
export function zoomAt(view,point,zoom,width,height) {
 const anchor=sourcePoint(point,view,width,height);
 return {zoom,x:anchor.x-(point.x-width/2)/zoom,y:anchor.y-(point.y-height/2)/zoom};
}
export function pointBox(point,size,width,height) {
 return {x:Math.max(0,point.x-size/2),y:Math.max(0,point.y-size/2),w:Math.min(width,point.x+size/2)-Math.max(0,point.x-size/2),h:Math.min(height,point.y+size/2)-Math.max(0,point.y-size/2)};
}
