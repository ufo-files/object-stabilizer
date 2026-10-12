"""Reproduce the sample benchmark; raw video/artifacts stay outside Git."""
from pathlib import Path
import argparse
import csv
import json
import math
import sys
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.processing import track_video,export_video

parser=argparse.ArgumentParser();parser.add_argument('video',type=Path);parser.add_argument('--output',type=Path,default=Path(__file__).parent/'artifacts');args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
track=track_video(args.video,dict(point=[1252,522],size=24,radius=100,mode='bright'))
(args.output/'tracking.json').write_text(json.dumps(track,indent=2))
rows={r['frame']:r for r in track['rows']}
manual=list(csv.DictReader((Path(__file__).parent/'manual-target.csv').open()))
comparisons=[]
for r in manual:
    frame=int(r['frame']);p=rows[frame];error=math.hypot(p['x']-float(r['x']),p['y']-float(r['y']))
    comparisons.append(dict(frame=frame,manual=[float(r['x']),float(r['y'])],tracked=[p['x'],p['y']],error_source_pixels=error))
export=export_video(args.video,track,args.output/'validated-stabilized-8x.mp4',8)
cap=cv2.VideoCapture(str(args.output/'validated-stabilized-8x.mp4'))
output=[];errors=[];frames=0
canvas=np.full((3*230,4*230,3),245,np.uint8)
selected=set(np.linspace(0,export['frames']-1,12,dtype=int))
while True:
    ok,f=cap.read()
    if not ok:break
    h,w=f.shape[:2];cx,cy=w//2,h//2;roi=cv2.cvtColor(f[cy-60:cy+60,cx-60:cx+60],cv2.COLOR_BGR2GRAY).astype(float)
    # Independent output measurement: intensity above local background, no tracker model.
    weights=np.maximum(roi-140,0);yy,xx=np.mgrid[:120,:120]
    if weights.sum():errors.append(math.hypot(float((xx*weights).sum()/weights.sum())-60,float((yy*weights).sum()/weights.sum())-60))
    if frames in selected:
        i=len(output);output.append(frames);x=(i%4)*230;y=(i//4)*230
        crop=f[cy-50:cy+50,cx-50:cx+50];crop=cv2.resize(crop,(200,200),interpolation=cv2.INTER_NEAREST)
        cv2.line(crop,(100,0),(100,199),(0,100,0),1);cv2.line(crop,(0,100),(199,100),(0,100,0),1)
        canvas[y+25:y+225,x+15:x+215]=crop
        cv2.putText(canvas,f'Output frame {frames}',(x+15,y+18),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
    frames+=1
cap.release();cv2.imwrite(str(args.output/'centered-contact-sheet.png'),canvas)
report=dict(source=args.video.name,target='User-confirmed bright point near the middle of the sky',mode='bright',decoded_source_frames=len(track['rows']),rejected_frames=track['rejected'],manual_review=dict(samples=len(comparisons),estimated_annotation_uncertainty_pixels=1.,mean_error_source_pixels=float(np.mean([r['error_source_pixels'] for r in comparisons])),max_error_source_pixels=max(r['error_source_pixels'] for r in comparisons),comparisons=comparisons),export=dict(zoom=8,frames=frames,fps=export['fps'],duration_seconds=export['duration'],original_duration_seconds=track['duration'],max_center_error_output_pixels=max(errors),mean_center_error_output_pixels=float(np.mean(errors))),limitations='One nearly stationary bright point in a short night clip. Manual positions are approximate (±1 source pixel). This does not validate moving objects, occlusion, or general scenes; those have synthetic regression checks.')
(Path(__file__).parent/'sample-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='manual_review'},indent=2));print('Manual sample maximum error:',report['manual_review']['max_error_source_pixels'])
assert report['rejected_frames']==0
assert report['manual_review']['max_error_source_pixels']<1
assert report['export']['max_center_error_output_pixels']<2
assert abs(export['duration']-track['duration'])<1/track['fps']
