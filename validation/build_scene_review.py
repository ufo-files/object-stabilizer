"""Build a frame-matched review proxy from source and actual encoded export."""
from itertools import zip_longest
from pathlib import Path
import argparse,json
import cv2,numpy as np
from backend.tripod import frames


def build(source,export,report,output):
 report=json.loads(Path(report).read_text());output=Path(output);output.mkdir(parents=True,exist_ok=True)
 width,height=report['width'],report['height'];rate=report['source']['fps'];points=np.array(report['analysis']['points'])[:4]
 original=frames(source,report['source']['width'],report['source']['height']);encoded=frames(export,width,height)
 writer=cv2.VideoWriter(str(output/'scene-comparison.mp4'),cv2.VideoWriter_fourcc(*'avc1'),rate,(1920,780))
 contact=[];scenes=[];count=0
 try:
  for index,(before,after) in enumerate(zip_longest(original,encoded)):
   if before is None or after is None:raise ValueError('Source and export frame counts differ.')
   display=np.zeros((780,1920,3),np.uint8)
   for col,(frame,label) in enumerate(((before,'ORIGINAL'),(after,'AUTOMATIC SCENE'))):
    display[:540,col*960:(col+1)*960]=cv2.resize(frame,(960,540),interpolation=cv2.INTER_AREA)
    cv2.rectangle(display,(col*960,0),(col*960+959,27),(10,15,20),-1)
    cv2.putText(display,f'{label} - FRAME {index}',(col*960+12,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(140,245,180),1)
   for k,point in enumerate(points):
    x,y=np.rint(point).astype(int);origin=180+k*390
    for j,(frame,label) in enumerate(((before,'SOURCE'),(after,'EXPORT'))):
     crop=cv2.resize(frame[y-32:y+33,x-32:x+33],(195,195),interpolation=cv2.INTER_NEAREST)
     cx,cy=np.rint((point-[x,y]+32.5)*3).astype(int)
     for start,end in [((cx-12,cy),(cx-4,cy)),((cx+4,cy),(cx+12,cy)),((cx,cy-12),(cx,cy-4)),((cx,cy+4),(cx,cy+12))]:cv2.line(crop,start,end,(80,220,120),1)
     display[575:770,origin+j*195:origin+(j+1)*195]=crop
     cv2.putText(display,f'REF {k+1} {label}',(origin+j*195+6,564),cv2.FONT_HERSHEY_SIMPLEX,.4,(140,245,180),1)
   writer.write(display);count+=1
   if index in [0,35,39,187,255,263,283,287]:
    tile=display[540:780,180:1740].copy();cv2.putText(tile,f'FRAME {index}',(4,235),cv2.FONT_HERSHEY_SIMPLEX,.42,(140,245,180),1);contact.append(tile)
   if index in [0,263,348]:scenes.append(display[:540].copy())
 finally:
  writer.release();original.close();encoded.close()
 if count!=report['frames']:raise ValueError('Review proxy frame count differs from the export.')
 cv2.imwrite(str(output/'reference-frames.png'),np.concatenate(contact,axis=0))
 cv2.imwrite(str(output/'full-scene-frames.png'),np.concatenate(scenes,axis=0))
 (output/'review.json').write_text(json.dumps(dict(frames=count,proxy=True,frame_matched=True,reference_crops='Fixed frame-zero coordinates. Never follows output references.',zero_motion_passed=report['zero_motion_passed']),indent=2))
 print(json.dumps(dict(frames=count,output=str(output),zero_motion_passed=report['zero_motion_passed'])))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--export',required=True);p.add_argument('--report',required=True);p.add_argument('--output',required=True);a=p.parse_args();build(a.source,a.export,a.report,a.output)
