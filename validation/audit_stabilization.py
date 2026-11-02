"""Audit decoded exports, independently of stored tracker coordinates.

Every frame/reference is compared with both frame zero and its previous frame.
An unmeasurable patch fails the audit; it is never counted as zero motion.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def measure(template, image):
    if np.ptp(template)<.01 or np.ptp(image)<.01:
        return dict(dx=None,dy=None,pixels=None,correlation=None,measurable=False)
    if np.array_equal(template,image):
        return dict(dx=0.,dy=0.,pixels=0.,correlation=1.,measurable=True)
    try:
        score,warp=cv2.findTransformECC(template,image,np.eye(2,3,dtype=np.float32),
            cv2.MOTION_TRANSLATION,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-6),None,3)
        delta=warp[:,2].astype(float)
        return dict(dx=float(delta[0]),dy=float(delta[1]),pixels=float(np.linalg.norm(delta)),
                    correlation=float(score),measurable=bool(score>=.85 and np.isfinite(delta).all()))
    except cv2.error:
        return dict(dx=None,dy=None,pixels=None,correlation=None,measurable=False)


def audit(video,points,output,tolerance=0.0,progress=lambda *_:None,cancel=lambda:False,save_contact_sheet=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise ValueError('Could not decode export.')
    fps=cap.get(cv2.CAP_PROP_FPS);initial=None;previous=None;rows=[];frame=0
    try:
        while True:
            if cancel():raise InterruptedError('Stability audit cancelled.')
            ok,image=cap.read()
            if not ok:break
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
            patches=[cv2.getRectSubPix(gray,(65,65),tuple(map(float,p))) for p in points]
            if initial is None:initial=patches;previous=patches
            for index,(first,last,current) in enumerate(zip(initial,previous,patches)):
                drift=measure(first,current);step=measure(last,current)
                passed=(drift['measurable'] and step['measurable'] and drift['pixels']<=tolerance and step['pixels']<=tolerance)
                rows.append(dict(frame=frame,time_seconds=frame/fps,reference=index+1,passed=passed,
                    **{'drift_'+k:v for k,v in drift.items()},**{'step_'+k:v for k,v in step.items()}))
            previous=patches;frame+=1
            if frame%8==0:progress(frame)
    finally:cap.release()
    if not rows:raise ValueError('Export contained no decodable frames.')
    with (output/'frames.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    references=[]
    for index in range(len(points)):
        group=[r for r in rows if r['reference']==index+1]
        measured_steps=[r for r in group if r['step_measurable']]
        measured_drifts=[r for r in group if r['drift_measurable']]
        worst=sorted(measured_steps,key=lambda r:r['step_pixels'],reverse=True)[:8]
        references.append(dict(reference=index+1,failed_frames=sum(not r['passed'] for r in group),
            unmeasurable_step_frames=sum(not r['step_measurable'] for r in group),
            unmeasurable_drift_frames=sum(not r['drift_measurable'] for r in group),
            maximum_measurable_step_pixels=max((r['step_pixels'] for r in measured_steps),default=None),
            maximum_measurable_drift_pixels=max((r['drift_pixels'] for r in measured_drifts),default=None),
            worst_step_frames=[dict(frame=r['frame'],pixels=r['step_pixels'],correlation=r['step_correlation']) for r in worst]))
    report=dict(video=str(Path(video).resolve()),decoded_frames=frame,fps=fps,tolerance_pixels=tolerance,
        passed=all(r['passed'] for r in rows),primary_passed=all(r['passed'] for r in rows if r['reference']==1),
        method='Independent translation-only image registration of fixed 65x65 patches in every decoded frame; score below 0.85 is unmeasurable and fails.',
        limitation='Image registration estimates apparent patch displacement. A low correlation cannot distinguish motion from other image differences; it cannot certify stability.',references=references)
    (output/'report.json').write_text(json.dumps(report,indent=2))
    if not save_contact_sheet:return report
    # Fixed-position, nearest-neighbour before/after crops, without recentering.
    worst=references[0]['worst_step_frames'][:4];wanted={n for r in worst for n in (max(0,r['frame']-1),r['frame'])};images={}
    cap=cv2.VideoCapture(str(video));n=0
    while True:
        ok,image=cap.read()
        if not ok:break
        if n in wanted:images[n]=image
        n+=1
    cap.release();panels=[]
    for item in worst:
        for n in (max(0,item['frame']-1),item['frame']):
            tiles=[]
            for i,point in enumerate(points):
                tile=cv2.resize(cv2.getRectSubPix(images[n],(65,65),tuple(map(float,point))),(260,260),interpolation=cv2.INTER_NEAREST)
                cv2.line(tile,(120,130),(140,130),(0,200,0),1);cv2.line(tile,(130,120),(130,140),(0,200,0),1)
                cv2.putText(tile,f'Frame {n}, ref {i+1}',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,0),1)
                tiles.append(tile)
            panels.append(np.concatenate(tiles,axis=1))
    if panels:cv2.imwrite(str(output/'worst-frame-pairs.png'),np.concatenate(panels,axis=0))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('video');parser.add_argument('--references',required=True,help='QA JSON containing referencePoints');parser.add_argument('--output',required=True);parser.add_argument('--tolerance',type=float,default=0.0);args=parser.parse_args()
    points=json.loads(Path(args.references).read_text())['referencePoints']
    report=audit(args.video,points,args.output,args.tolerance)
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['passed'] else 1)
