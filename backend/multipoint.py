"""Stabilize explicitly selected reference objects in their first-frame positions."""
import math
from contextlib import closing
import cv2
import numpy as np
from .tracking import BrightPointTracker,FlowTracker


def registration_matrix(current,reference,width,height):
    current=np.asarray(current,np.float64);reference=np.asarray(reference,np.float64)
    count=len(reference)
    if current.shape!=reference.shape or reference.shape!=(count,2) or not 2<=count<=4:
        raise ValueError('Select two to four reference objects.')
    if not np.isfinite(current).all() or not np.isfinite(reference).all():
        raise ValueError('Reference coordinates must be finite.')
    for points in (current,reference):
        if min(np.linalg.norm(points[i]-points[j]) for i in range(count) for j in range(i))<8:
            raise ValueError('Reference objects must be distinct and at least 8 pixels apart.')
    # Orthogonal Procrustes: one rotation and translation, never scale or shear.
    source_center=current[0];target_center=reference[0]
    u,_,vt=np.linalg.svd((current-source_center).T@(reference-target_center))
    correction=np.eye(2);correction[-1,-1]=np.linalg.det(vt.T@u.T)
    rotation=vt.T@correction@u.T
    matrix=np.eye(3);matrix[:2,:2]=rotation
    matrix[:2,2]=target_center-rotation@source_center
    registered=current@rotation.T+matrix[:2,2]
    error=float(np.linalg.norm(registered-reference,axis=1).max())
    return matrix,error


def rigid_trajectory(rows,reference,width,height,tolerance=.5):
    """Reject non-rigid reference disagreements before they become camera roll.

    Translation always follows the measured primary point, without smoothing.
    Only rotation is interpolated when the joint rigid fit is inconsistent.
    """
    fits=[registration_matrix(row['points'],reference,width,height) for row in rows]
    angles=np.unwrap([math.atan2(matrix[1,0],matrix[0,0]) for matrix,_ in fits])
    supported=np.array([error<=tolerance for _,error in fits])
    if not supported.any():raise ValueError('No consistent rigid orientation could be established.')
    times=np.array([row['time'] for row in rows],float)
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Reference frame times must be strictly increasing.')
    rotations=np.interp(times,times[supported],angles[supported])
    matrices=[]
    for row,angle in zip(rows,rotations):
        cosine=math.cos(angle);sine=math.sin(angle)
        matrix=np.array([[cosine,-sine,0.],[sine,cosine,0.],[0.,0.,1.]])
        matrix[:2,2]=np.asarray(reference[0])-matrix[:2,:2]@np.asarray(row['points'][0])
        matrices.append(matrix)
    return matrices,supported


def core_center(frame,point):
    x,y=point;h,w=frame.shape[:2];r=16
    left=max(0,int(round(x))-r);top=max(0,int(round(y))-r)
    patch=frame[top:min(h,int(round(y))+r+1),left:min(w,int(round(x))+r+1)]
    gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY).astype(float)
    border=np.concatenate([gray[0],gray[-1],gray[:,0],gray[:,-1]])
    background=float(np.median(border));peak=float(gray.max()-background)
    noise=max(64,float(np.median(np.abs(border-background))*1.4826))
    if peak<max(300,noise*5):raise ValueError('A reference object is no longer reliably visible. Choose a different object or shorter segment.')
    weights=np.maximum(gray-background-peak*.8,0);yy,xx=np.indices(gray.shape);mass=weights.sum()
    center=np.array([left+(xx*weights).sum()/mass,top+(yy*weights).sum()/mass])
    if np.linalg.norm(center-np.array(point))>12:raise ValueError('A reference object lost its lock. Choose a different reference or shorter segment.')
    return center


def refine_anchor(frame,matrix,reference):
    """Correct sampling error using translation alone; keep the first star locked."""
    matrix=matrix.copy();target=np.asarray(reference[0],float)
    left=int(round(target[0]))-16;top=int(round(target[1]))-16
    offset=np.array([[1.,0,-left],[0,1.,-top],[0,0,1.]])
    def measure(candidate):
        patch=cv2.warpAffine(frame,(offset@candidate)[:2],(33,33),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
        measured=core_center(patch,target-[left,top])+[left,top]
        delta=measured-target
        return float(np.linalg.norm(delta)),delta
    error,delta=measure(matrix)
    if error>2:raise ValueError('Primary reference failed its rendered-position check.')
    best=matrix.copy();best_error=error
    for _ in range(6):
        if error<.02:break
        matrix[:2,2]-=delta
        error,delta=measure(matrix)
        if error<best_error:best=matrix.copy();best_error=error
    # Sampling and a thresholded centroid are not smooth. Keep the best actual
    # rendered result instead of returning the last iterate in an oscillation.
    for step in (.25,.125,.0625,.03125):
        if best_error<.02:break
        origin=best.copy()
        for dx in (-2,-1,0,1,2):
            for dy in (-2,-1,0,1,2):
                candidate=origin.copy();candidate[:2,2]+=[dx*step,dy*step]
                error,_=measure(candidate)
                if error<best_error:best=candidate;best_error=error
    return best,best_error


def track_multiple_video(path,options,progress=lambda *_:None,cancel=lambda:False):
    from .processing import metadata,source_frames_16bit,Cancelled
    info=metadata(path);points=options['points'];requested=float(options.get('start',0))
    if not 2<=len(points)<=4:raise ValueError('Select two to four reference objects.')
    if not 0<=requested<info['duration']:raise ValueError('Start frame is outside the video.')
    for x,y in points:
        if not(0<=x<info['width'] and 0<=y<info['height']):raise ValueError('A reference is outside the video.')
    registration_matrix(points,points,info['width'],info['height'])
    crop=options.get('crop_box')
    if crop:
        x,y,w,h=crop
        if min(w,h)<6 or x<0 or y<0 or x+w>info['width']+.01 or y+h>info['height']+.01:raise ValueError('Crop box is outside the video.')
    cap=cv2.VideoCapture(str(path))
    if requested:cap.set(cv2.CAP_PROP_POS_MSEC,requested*1000)
    rows=[];mode=options.get('mode','bright')
    try:
        ok,frame=cap.read()
        if not ok:raise ValueError('Could not decode the selected frame.')
        start=max(0,cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
        engine=BrightPointTracker if mode=='bright' else FlowTracker
        trackers=[engine(frame,p,24 if mode=='bright' else 48,options.get('radius',100)) for p in points]
        rows.append(dict(frame=int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1,time=start,points=[t.initial.tolist() for t in trackers],accepted=True,confidence=1.))
        while True:
            if cancel():raise Cancelled()
            ok,frame=cap.read()
            if not ok:break
            time=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if time<=rows[-1]['time']:time=rows[-1]['time']+1/info['fps']
            observations=[tracker.update(frame) for tracker in trackers]
            for i,item in enumerate(observations):
                if not item.accepted:raise ValueError(f'Reference {i+1} lost lock at {time:.2f}s. Choose a different reference or shorter segment. No approximate stabilization was exported.')
            rows.append(dict(frame=int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1,time=time,points=[[o.x,o.y] for o in observations],accepted=True,confidence=min(o.confidence for o in observations)))
            progress(.55*(time-start)/max(.001,info['duration']-start),f'Tracking {len(points)} references: frame {len(rows)}')
    finally:cap.release()
    track=dict(**info,start=start,mode=mode,rows=rows,rejected=0,multiple_points=True,
               selected_points=points,crop_box=crop,center_selection=bool(crop),
               crop_width=crop[2] if crop else None,crop_height=crop[3] if crop else None,
               output_width=(options.get('output_width') or info['width']) if crop else info['width'],
               output_height=(options.get('output_height') or info['height']) if crop else info['height'])
    from .primary_motion import PrimaryImageMotion
    primary=PrimaryImageMotion()
    reference=None;maximum=0;aligned=0
    # Photometry first, then estimate a checked orientation over the complete
    # timeline. Inconsistent secondary points must never inject camera roll.
    with closing(source_frames_16bit(path,track,cancel)) as frames:
        for frame,row in zip(frames,rows):
            if cancel():raise Cancelled()
            measured=[core_center(frame,p).tolist() for p in row['points']] if mode=='bright' else row['points']
            row['detected_points']=row['points'];row['points']=measured
            primary.add(frame,measured[0])
            if reference is None:reference=measured
            aligned+=1
            progress(.55+.2*aligned/len(rows),f'Measuring references: frame {aligned} / {len(rows)}')
    if aligned!=len(rows):raise ValueError('Original decode ended before every selected reference could be measured.')
    correct_rotation=bool(options.get('correct_rotation',False))
    if correct_rotation:matrices,supported=rigid_trajectory(rows,reference,info['width'],info['height'])
    else:matrices=[np.eye(3) for _ in rows];supported=np.ones(len(rows),bool)
    primary_positions,image_scores=primary.positions([row['time'] for row in rows],reference[0])
    for i,(row,matrix,position,score) in enumerate(zip(rows,matrices,primary_positions,image_scores)):
        if cancel():raise Cancelled()
        row['core_points']=row['points']
        measured=np.asarray(row['points'],float).copy();measured[0]=position
        matrix[:2,2]=np.asarray(reference[0])-matrix[:2,:2]@position
        registered=measured@matrix[:2,:2].T+matrix[:2,2]
        error=float(np.linalg.norm(registered-np.asarray(reference),axis=1).max())
        row.update(points=measured.tolist(),camera_transform=matrix.tolist(),reference_error=error,
                   rotation_supported=bool(supported[i]),primary_image_score=float(score),
                   accepted=bool(score>=.85),confidence=min(row['confidence'],float(score)),
                   reason='' if score>=.85 else 'Primary image registration is uncertain',x=float(position[0]),y=float(position[1]))
        maximum=max(maximum,error)
        progress(.75+.24*(i+1)/len(rows),f'Preparing image-based stabilization: frame {i+1} / {len(rows)}')
    track.update(reference_points=reference,maximum_reference_error=maximum,
                 rotation_uncertain_frames=int((~supported).sum()),rotation_fit_tolerance=.5,
                 rejected=sum(not row['accepted'] for row in rows),
                 transform_model='tripod_image_pose_v6',primary_reference=0,correct_rotation=correct_rotation,
                 primary_lock='joint initial-image and consecutive-image registration',
                 stabilization='rigid image-patch stabilization' if correct_rotation else 'translation-only image-patch stabilization',reference_error_units='pixels')
    return track
