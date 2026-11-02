"""Fixed-coordinate before/after review and an independent rigid-transform check.

No crop in this review follows a detected point in the output. All three clips
are decoded sequentially. The small magnified video is a review proxy, not the
full-resolution stabilized export.
"""
from itertools import zip_longest
from pathlib import Path
import argparse,json
import cv2,numpy as np
from backend.point_lock import yuv_frames
from backend.light_center import translation_sample_error,profile_observations


def run(source,old,new,report,directory):
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
 report=json.loads(Path(report).read_text());window=np.array(report['window']);x,y=map(int,window)
 videos=[yuv_frames(p,report['source']['width'],report['source']['height']) for p in (source,old,new)]
 writer=cv2.VideoWriter(str(directory/'fixed-coordinate-comparison.mp4'),cv2.VideoWriter_fourcc(*'avc1'),report['source']['fps'],(1170,430))
 wanted=[0,35,39,187,255,263,283,287];panels=[];samples=[];source_shapes=[]
 try:
  for index,frames in enumerate(zip_longest(*videos)):
   if any(f is None for f in frames):raise ValueError('Comparison input frame counts differ.')
   row=report['rows'][index]
   error,mean,count=translation_sample_error(frames[0][0],frames[2][0],(row['dx'],row['dy']),window)
   samples.append(dict(frame=index,max_error=error,mean_error=mean,samples=count))
   # Shape observation is explicitly source-peak-centered, solely to measure
   # internal relative positions. It is never used to recenter review output.
   source=frames[0][0];sx,sy=np.rint([row['source_x'],row['source_y']]).astype(int)
   box=source[sy-32:sy+33,sx-32:sx+33];profiles=profile_observations(box,(32,32))
   delta=profiles['profile_center_2']-profiles['profile_center_8']
   source_shapes.append(dict(frame=index,separation_vector=delta.tolist(),separation_pixels=float(np.linalg.norm(delta))))
   display=np.zeros((430,1170,3),np.uint8)
   for col,(planes,label) in enumerate(zip(frames,['ORIGINAL FIXED REGION','PREVIOUS PEAK LOCK','NEW LIGHT-CENTER ALIGNMENT'])):
    patch=planes[0][y-32:y+33,x-32:x+33]
    crop=cv2.cvtColor(cv2.resize(patch,(390,390),interpolation=cv2.INTER_NEAREST),cv2.COLOR_GRAY2BGR)
    cx=(report['anchor'][0]-x+32.5)*6;cy=(report['anchor'][1]-y+32.5)*6
    for a,b in [((cx-18,cy),(cx-7,cy)),((cx+7,cy),(cx+18,cy)),((cx,cy-18),(cx,cy-7)),((cx,cy+7),(cx,cy+18))]:cv2.line(crop,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),(60,200,80),1)
    display[40:,col*390:(col+1)*390]=crop
    cv2.putText(display,label,(col*390+8,16),cv2.FONT_HERSHEY_SIMPLEX,.42,(80,240,120),1)
    cv2.putText(display,f'FRAME {index}',(col*390+8,33),cv2.FONT_HERSHEY_SIMPLEX,.42,(80,240,120),1)
   writer.write(display)
   if index in wanted:panels.append(display)
 finally:
  writer.release()
  for video in videos:video.close()
 cv2.imwrite(str(directory/'fixed-coordinate-frames.png'),np.concatenate(panels,axis=0))
 initial=source_shapes[0]['separation_pixels'];maximum_change=max(abs(row['separation_pixels']-initial) for row in source_shapes)
 result=dict(frames=len(samples),fixed_output_window=window.tolist(),full_frame_translation_samples=samples,
             max_translation_sample_error=max(row['max_error'] for row in samples),
             mean_translation_sample_error=float(np.mean([row['mean_error'] for row in samples])),
             source_internal_profile_observations=source_shapes,
             source_max_internal_separation_change_pixels=maximum_change,
             rigid_alignment_limit='A translation or rotation cannot change the separation of two measured features inside a light profile. These are image measurements, not a determination of the physical cause.',
             zero_motion_passed=report['zero_motion_passed'])
 (directory/'review.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('full_frame_translation_samples','source_internal_profile_observations')},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--old',required=True);p.add_argument('--new',required=True);p.add_argument('--report',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.source,a.old,a.new,a.report,a.output)
