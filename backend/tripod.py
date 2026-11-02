"""Tripod-only frame registration. Every pose refers directly to frame zero."""
from pathlib import Path
from fractions import Fraction
import csv
import json
import subprocess
import cv2
import numpy as np


def info(path):
    data=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=r_frame_rate,color_space:format=duration','-of','json',str(path)]))
    cap=cv2.VideoCapture(str(path)); ok,frame=cap.read(); cap.release()
    if not ok: raise ValueError('Cannot decode the first frame.')
    rate=data['streams'][0]['r_frame_rate']
    return dict(color_space=(data['streams'][0].get('color_space') if data['streams'][0].get('color_space') in ('bt709','bt470bg','smpte170m','smpte240m','bt2020nc') else 'bt709'),width=frame.shape[1],height=frame.shape[0],rate=rate,fps=float(Fraction(rate)),duration=float(data['format']['duration']))


def frames(path,width,height):
    process=subprocess.Popen(['ffmpeg','-v','error','-nostdin','-i',str(path),'-map','0:v:0','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','bgr24','-'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    try:
        while True:
            data=process.stdout.read(width*height*3)
            if not data: break
            if len(data)!=width*height*3: raise ValueError('Incomplete decoded frame.')
            yield np.frombuffer(data,np.uint8).reshape(height,width,3).copy()
        if process.wait(): raise ValueError('Video decoding failed.')
    finally:
        process.stdout.close()
        if process.poll() is None: process.terminate();process.wait()


class Reference:
    def __init__(self,frame,box,search=48):
        x,y,w,h=map(lambda v:int(round(v)),box)
        if min(w,h)<12: raise ValueError('Draw a reference box at least 12 pixels wide and high.')
        if x<search or y<search or x+w+search>frame.shape[1] or y+h+search>frame.shape[0]:
            raise ValueError('Keep the reference at least 48 pixels from the frame edge.')
        self.x,self.y,self.w,self.h,self.search=x,y,w,h,search
        self.template=cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY).astype(np.float32)/255
        if self.template.std()<.004: raise ValueError('The reference box needs visible detail.')

    def locate(self,frame):
        x,y,w,h,s=self.x,self.y,self.w,self.h,self.search
        gray=cv2.cvtColor(frame[y-s:y+h+s,x-s:x+w+s],cv2.COLOR_BGR2GRAY).astype(np.float32)/255
        matches=cv2.matchTemplate(gray,self.template,cv2.TM_CCOEFF_NORMED)
        _,score,_,at=cv2.minMaxLoc(matches)
        if score<.5: raise ValueError(f'Reference lost (match {score:.2f}). Choose a larger box containing fixed detail.')
        # ECC aligns the original template to this frame, never to the previous frame.
        warp=np.float32([[1,0,at[0]],[0,1,at[1]]])
        try:
            score,warp=cv2.findTransformECC(self.template,gray,warp,cv2.MOTION_TRANSLATION,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-6),None,3)
        except cv2.error as exc: raise ValueError('Could not refine the reference position.') from exc
        displacement=warp[:,2].astype(float)-s
        if score<.5 or np.max(np.abs(displacement))>=s-2: raise ValueError('Reference moved outside the tripod search area.')
        return displacement,float(score)


def patch(frame,reference,position):
    x,y= np.rint(position).astype(int)+[reference.x,reference.y]
    return cv2.cvtColor(frame[y:y+reference.h,x:x+reference.w],cv2.COLOR_BGR2GRAY).astype(np.float32)/255, np.array([x,y])


def relative_motion(first,current):
    if np.array_equal(first,current): return np.zeros(2),1.
    score,warp=cv2.findTransformECC(first,current,np.eye(2,3,dtype=np.float32),cv2.MOTION_TRANSLATION,(3,100,1e-6),None,3)
    if not np.isfinite(warp).all() or score<.5: raise ValueError('Reference could not be registered between adjacent frames.')
    return warp[:,2].astype(float),float(score)


def solve_path(absolute,increments):
    """Retain measured frame-to-frame motion, with weak absolute drift anchors.

    This fits camera displacement. It does not smooth the video trajectory: known
    camera vibration satisfies both observations and is corrected in full.
    """
    absolute=np.asarray(absolute);increments=np.asarray(increments)
    count=len(absolute)
    if count==1:return absolute.copy()
    weight=.005
    diagonal=np.full(count,2.+weight);diagonal[-1]=1.+weight
    rhs=weight*absolute;rhs[1:]+=increments[1:];rhs[:-1]-=increments[1:]
    diagonal=diagonal[1:].copy();rhs=rhs[1:].copy()
    for i in range(1,len(diagonal)):
        multiplier=-1/diagonal[i-1]
        diagonal[i]+=multiplier;rhs[i]-=multiplier*rhs[i-1]
    result=np.zeros_like(absolute);result[-1]=rhs[-1]/diagonal[-1]
    for i in range(len(diagonal)-2,-1,-1):result[i+1]=(rhs[i]+result[i+2])/diagonal[i]
    return result


def audit_export(destination,metadata,box):
    width,height=metadata['width'],metadata['height']
    output=frames(destination,width,height);first=next(output)
    ref=Reference(first,box);first_patch,_=patch(first,ref,[0,0]);previous=first_patch
    rows=[dict(frame=0,step_pixels=0.,drift_pixels=0.,step_score=1.,drift_score=1.)]
    for index,frame in enumerate(output,1):
        current,_=patch(frame,ref,[0,0]);row=dict(frame=index)
        for name,template in [('step',previous),('drift',first_patch)]:
            try:
                shift,score=relative_motion(template,current)
                row[name+'_pixels']=float(np.linalg.norm(shift)) if score>=.85 else None
                row[name+'_score']=score
            except (ValueError,cv2.error):row[name+'_pixels']=None;row[name+'_score']=None
        rows.append(row);previous=current
    return rows


def stabilize(source,destination,box,progress=lambda *_:None):
    metadata=info(source); width,height=metadata['width'],metadata['height']
    # Pass 1: measure every source frame. No corrected or browser frames enter tracking.
    decoder=frames(source,width,height);first=next(decoder);reference=Reference(first,box)
    previous,previous_origin=patch(first,reference,[0,0])
    absolute=[np.zeros(2)];increments=[np.zeros(2)];scores=[1.]
    try:
        for index,frame in enumerate(decoder,1):
            position,score=reference.locate(frame)
            current,origin=patch(frame,reference,position)
            shift,pair_score=relative_motion(previous,current)
            absolute.append(position);increments.append(origin-previous_origin+shift);scores.append(min(score,pair_score))
            previous,previous_origin=current,origin
            progress(min(.4,index/metadata['fps']/metadata['duration']*.4),f'Measuring frame {index+1}')
    finally:decoder.close()
    positions=solve_path(absolute,increments)
    # Pass 2: the nth pose is applied only to the nth decoded source frame.
    encoder=subprocess.Popen(['ffmpeg','-v','error','-nostdin','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}','-r',metadata['rate'],'-i','-','-i',str(source),'-map','0:v:0','-map','1:a:0?','-c:a','aac','-map_metadata','-1','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-preset','fast','-crf','10','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    decoder=frames(source,width,height);rows=[]
    try:
        for index,frame in enumerate(decoder):
            if index>=len(positions):raise ValueError('Source frame count changed between decoding passes.')
            displacement=positions[index]
            corrected=cv2.warpAffine(frame,np.float32([[1,0,-displacement[0]],[0,1,-displacement[1]]]),(width,height),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
            encoder.stdin.write(corrected.tobytes())
            rows.append(dict(frame=index,dx=float(displacement[0]),dy=float(displacement[1]),score=scores[index]))
            progress(.4+.45*(index+1)/len(positions),f'Correcting frame {index+1} of {len(positions)}')
        if len(rows)!=len(positions):raise ValueError('Source frame count changed between decoding passes.')
        encoder.stdin.close();error=encoder.stderr.read().decode();code=encoder.wait()
        if code: raise ValueError(error[-1000:])
    finally:
        decoder.close()
        if encoder.poll() is None:encoder.kill();encoder.wait()
        encoder.stderr.close()
    progress(.9,'Checking every exported frame')
    # Padding only adds bottom/right pixels; source coordinates remain unchanged.
    audit_metadata=dict(metadata,width=width+width%2,height=height+height%2)
    audit=audit_export(destination,audit_metadata,box)
    measured_steps=[r['step_pixels'] for r in audit if r['step_pixels'] is not None]
    measured_drifts=[r['drift_pixels'] for r in audit if r['drift_pixels'] is not None]
    unmeasurable=sum(r['step_pixels'] is None or r['drift_pixels'] is None for r in audit)
    maximum_step=max(measured_steps,default=None);maximum_drift=max(measured_drifts,default=None)
    report=dict(engine='tripod-rebuild-1',frames=len(rows),output_frames=len(audit),source=metadata,box=list(box),rows=rows,audit=audit,max_step_pixels=maximum_step,max_residual_pixels=maximum_drift,unmeasurable_frames=unmeasurable,zero_motion_passed=unmeasurable==0 and maximum_step==0 and maximum_drift==0)
    if len(audit)!=len(rows):raise ValueError('Encoded frame count does not match the source.')
    Path(destination).with_suffix('.json').write_text(json.dumps(report))
    with Path(destination).with_suffix('.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=['frame','dx','dy','score']);writer.writeheader();writer.writerows(rows)
    return report
