from fractions import Fraction
from pathlib import Path
import json
import math
import shutil
import subprocess
import cv2
import numpy as np
from .tracking import BrightPointTracker, FlowTracker, repair_gaps

class Cancelled(Exception):pass

def metadata(path):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):raise ValueError('Install FFmpeg (including ffprobe) to process videos.')
    p=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,duration,nb_frames:format=duration','-of','json',str(path)],capture_output=True,text=True,check=True)
    data=json.loads(p.stdout)
    if not data.get('streams'):raise ValueError('No video stream found.')
    stream=data['streams'][0];rate=stream['r_frame_rate'];fps=float(Fraction(rate))
    duration=float(stream.get('duration') or data['format']['duration'])
    if not (0<fps<=240 and 0<duration<=86400):raise ValueError('Unsupported frame rate or duration.')
    cap=cv2.VideoCapture(str(path));ok,frame=cap.read();cap.release()
    if not ok:raise ValueError('OpenCV could not decode this video.')
    return dict(width=frame.shape[1],height=frame.shape[0],fps=fps,rate=rate,duration=duration,frames=int(stream.get('nb_frames') or math.ceil(duration*fps)))

def track_video(path,options,progress=lambda *_:None,cancel=lambda:False):
    info=metadata(path);cap=cv2.VideoCapture(str(path));requested=float(options.get('start',0))
    if not 0<=requested<info['duration']:raise ValueError('Start frame is outside the video.')
    point=options['point']
    if not(0<=point[0]<info['width'] and 0<=point[1]<info['height']):raise ValueError('Target is outside the video frame.')
    if requested:cap.set(cv2.CAP_PROP_POS_MSEC,requested*1000)
    try:
        ok,frame=cap.read()
        if not ok:raise ValueError('Could not decode the selected start frame.')
        start=max(0,cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
        engine=BrightPointTracker if options.get('mode','bright')=='bright' else FlowTracker
        tracker=engine(frame,point,options.get('size',24),options.get('radius',100))
        rows=[dict(frame=int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1,time=start,x=float(tracker.initial[0]),y=float(tracker.initial[1]),confidence=1.,accepted=True)]
        while True:
            if cancel():raise Cancelled()
            ok,frame=cap.read()
            if not ok:break
            t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if t<=rows[-1]['time']:t=rows[-1]['time']+1/info['fps']
            observation=tracker.update(frame)
            rows.append(dict(frame=int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1,time=t,**observation.row()))
            progress(min(.99,(t-start)/max(.001,info['duration']-start)),f'Tracking source frame {len(rows)}')
        rows=repair_gaps(rows)
        return dict(**info,start=start,mode=options.get('mode','bright'),rows=rows,rejected=sum(not r['accepted'] for r in rows))
    finally:cap.release()

def transform(frame,point,zoom):
    h,w=frame.shape[:2]
    matrix=np.array([[zoom,0,w/2-zoom*point[0]],[0,zoom,h/2-zoom*point[1]]],np.float64)
    return cv2.warpAffine(frame,matrix,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)

def export_video(path,track,destination,zoom=1,progress=lambda *_:None,cancel=lambda:False):
    if not(1<=zoom<=16):raise ValueError('Zoom must be between 1 and 16.')
    destination=Path(destination);temporary=destination.with_name(destination.stem+'.partial.mp4')
    cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,track['rows'][0]['frame'])
    width,height=track['width'],track['height'];start=track['start'];fps=track['fps']
    total=math.ceil((track['duration']-start)*fps-1e-6)
    args=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}','-r',track['rate'],'-i','pipe:0','-ss',str(start),'-i',str(path),'-map','0:v:0','-map','1:a?','-c:v','libx264','-threads','2','-preset','fast','-crf','18','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-pix_fmt','yuv420p','-c:a','aac','-shortest','-movflags','+faststart',str(temporary)]
    # File-backed stderr cannot fill a pipe while frames are streamed.
    with temporary.with_suffix('.log').open('w+b') as log:
        proc=subprocess.Popen(args,stdin=subprocess.PIPE,stderr=log,stdout=subprocess.DEVNULL)
        count=0
        try:
            previous=None;previous_row=None
            for row in track['rows']:
                if cancel():raise Cancelled()
                ok,frame=cap.read()
                if not ok:raise ValueError('Source decode ended before the recorded track.')
                while previous is not None and count<total and start+count/fps<row['time']-1e-6:
                    if cancel():raise Cancelled()
                    rendered=transform(previous,(previous_row['x'],previous_row['y']),zoom)
                    proc.stdin.write(rendered.tobytes());count+=1
                    progress(count/total,f'Encoding frame {count} / {total}')
                previous=frame;previous_row=row
            while previous is not None and count<total:
                if cancel():raise Cancelled()
                proc.stdin.write(transform(previous,(previous_row['x'],previous_row['y']),zoom).tobytes());count+=1
                progress(count/total,f'Encoding frame {count} / {total}')
            proc.stdin.close();code=proc.wait(timeout=120)
            if code:
                log.seek(0);raise ValueError('FFmpeg export failed: '+log.read().decode(errors='replace')[-1000:])
            temporary.replace(destination)
        except Exception:
            if proc.poll() is None:proc.kill();proc.wait()
            if temporary.exists():temporary.unlink()
            raise
        finally:
            cap.release()
            if proc.stdin and not proc.stdin.closed:proc.stdin.close()
    temporary.with_suffix('.log').unlink(missing_ok=True)
    return dict(frames=count,fps=fps,duration=count/fps,path=str(destination))
