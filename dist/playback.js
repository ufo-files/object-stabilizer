// Match the displayed source frame, not an interpolated position between frames.
export function sampleTrack(rows,time){
 if(time<=rows[0].time)return rows[0];
 let lo=0,hi=rows.length;
 while(lo+1<hi){const mid=(lo+hi)>>1;if(rows[mid].time<=time+.0001)lo=mid;else hi=mid;}
 return rows[lo];
}
