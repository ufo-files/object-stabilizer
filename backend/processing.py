from fractions import Fraction
from pathlib import Path
import json
import math
import shutil
import subprocess
import zipfile
import tempfile
from contextlib import closing
import cv2
import numpy as np
from .tracking import BrightPointTracker, FlowTracker, repair_gaps

class Cancelled(Exception):pass

def prepare_video(path):
    """Create a browser-compatible proxy for preview and tracking only."""
    destination=Path(path).with_name('prepared.mp4')
    filters='pad=ceil(iw/2)*2:ceil(ih/2)*2'
    result=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y',
        '-i',str(path),'-map','0:v:0','-map','0:a:0?',
        '-c:v','libx264','-threads','2','-preset','fast','-crf','18',
        '-vf',filters,'-pix_fmt','yuv420p','-map_metadata','-1',
        '-fps_mode','passthrough','-c:a','aac','-movflags','+faststart',
        str(destination)],capture_output=True,text=True)
    if result.returncode:
        destination.unlink(missing_ok=True)
        raise ValueError('Could not prepare this video: '+result.stderr[-1000:])
    return destination

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
    if options.get('points'):
        from .multipoint import track_multiple_video
        return track_multiple_video(path,options,progress,cancel)
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
            progress(min(.69,.7*(t-start)/max(.001,info['duration']-start)),f'Tracking source frame {len(rows)}')
        rows=repair_gaps(rows)
        result=dict(**info,output_width=options.get('output_width') or info['width'],output_height=options.get('output_height') or info['height'],center_selection=options.get('center_selection',True),crop_width=options.get('crop_width'),crop_height=options.get('crop_height'),reference_size=options.get('size',24),start=start,mode=options.get('mode','bright'),rows=rows,rejected=sum(not r['accepted'] for r in rows))
        if options.get('mode','bright')=='bright' and ((options.get('crop_width') and options.get('crop_height')) or not result['center_selection']):
            refine_visible_centers(path,result,progress,cancel)
        return result
    finally:cap.release()

def transform(frame,point,zoom,output_size=None):
    h,w=frame.shape[:2]
    if output_size:w,h=output_size
    if zoom>4:
        return precise_crop(frame,point,zoom,w,h)
    matrix=np.array([[zoom,0,w/2-zoom*point[0]],[0,zoom,h/2-zoom*point[1]]],np.float64)
    return cv2.warpAffine(frame,matrix,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)

def precise_crop(frame,point,zoom,width,height):
    """Separable cubic sampling without warpAffine's 1/32-pixel lookup grid."""
    def coordinates(length,center):
        positions=(np.arange(length,dtype=np.float64)-length/2)/zoom+center
        base=np.floor(positions).astype(int)
        indices=base[:,None]+np.arange(-1,3)[None,:]
        distance=np.abs(positions[:,None]-indices)
        a=-.75
        weights=np.where(distance<=1,(a+2)*distance**3-(a+3)*distance**2+1,
            np.where(distance<2,a*distance**3-5*a*distance**2+8*a*distance-4*a,0))
        return indices,weights.astype(np.float32)
    xs,wx=coordinates(width,point[0]);ys,wy=coordinates(height,point[1])
    source_height,source_width=frame.shape[:2]
    top=max(0,int(ys.min()));bottom=min(source_height,int(ys.max())+1)
    if bottom<=top:return np.zeros((height,width,*frame.shape[2:]),dtype=frame.dtype)
    source=frame[top:bottom]
    horizontal=np.zeros((bottom-top,width,*frame.shape[2:]),np.float32)
    trailing=(None,)*(frame.ndim-2)
    for tap in range(4):
        valid=(xs[:,tap]>=0)&(xs[:,tap]<source_width)
        weight=(wx[:,tap]*valid)[(None,slice(None),*trailing)]
        horizontal+=source[:,np.clip(xs[:,tap],0,source_width-1)].astype(np.float32)*weight
    output=np.zeros((height,width,*frame.shape[2:]),np.float32)
    for tap in range(4):
        index=ys[:,tap]-top;valid=(index>=0)&(index<bottom-top)
        weight=(wy[:,tap]*valid)[(slice(None),None,*trailing)]
        output+=horizontal[np.clip(index,0,bottom-top-1)]*weight
    if np.issubdtype(frame.dtype,np.integer):output=np.rint(np.clip(output,0,np.iinfo(frame.dtype).max))
    return output.astype(frame.dtype)

def output_geometry(track,zoom):
    if track.get('multiple_points'):
        if track.get('crop_box'):
            return track['output_width'],track['output_height'],min(track['output_width']/track['crop_box'][2],track['output_height']/track['crop_box'][3])
        return track['width'],track['height'],1
    if not track.get('center_selection',True):return track['width'],track['height'],1
    width,height=track.get('output_width',track['width']),track.get('output_height',track['height'])
    if track.get('crop_width') and track.get('crop_height'):
        zoom*=min(width/track['crop_width'],height/track['crop_height'])
    return width,height,zoom

def original_source(path):
    path=Path(path)
    original=path.with_name('source.video')
    return original if path.name=='prepared.mp4' and original.is_file() else path

def source_color(path):
    result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
        'stream=color_primaries,color_transfer,pix_fmt','-of','json',str(original_source(path))],capture_output=True,text=True,check=True)
    return json.loads(result.stdout).get('streams',[{}])[0]

def source_frames_16bit(path,track,cancel):
    """Decode the original, with its display rotation, without the 8-bit proxy."""
    source=original_source(path)
    size=track['width']*track['height']*3*2
    args=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-threads','2','-i',str(source),
        '-map','0:v:0','-an','-sn','-dn','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-fps_mode','passthrough','-pix_fmt','bgr48le','-f','rawvideo','pipe:1']
    with tempfile.TemporaryFile() as log:
        proc=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=log)
        index=0
        try:
            while True:
                if cancel():raise Cancelled()
                data=proc.stdout.read(size)
                if not data:break
                if len(data)!=size:raise ValueError('Original video decode returned an incomplete frame.')
                if index>=track['rows'][0]['frame']:
                    yield np.frombuffer(data,dtype='<u2').reshape(track['height'],track['width'],3)
                index+=1
            if proc.wait(timeout=30):
                log.seek(0);raise ValueError('Original video decode failed: '+log.read().decode(errors='replace')[-1000:])
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
            proc.stdout.close()

def refine_visible_centers(path,track,progress=lambda *_:None,cancel=lambda:False):
    """Center the rendered body, not only the detector's band-pass signal.

    Measuring after cubic sampling matters when a tiny source crop is enlarged
    to HD. This has no temporal smoothing or motion lag. Low-confidence frames
    remain flagged, and refinement cannot move outside the selected footprint.
    """
    full_frame=not track.get('center_selection',True)
    cw,ch=(min(32,max(24,track.get('reference_size',24))),)*2 if full_frame else (track['crop_width'],track['crop_height'])
    resolution=192 if full_frame else min(768,max(192,track.get('output_width',192),track.get('output_height',192)))
    scale=min(resolution/cw,resolution/ch);width=max(8,round(cw*scale));height=max(8,round(ch*scale))
    yy,xx=np.indices((height,width));color=source_color(path)
    with closing(source_frames_16bit(path,track,cancel)) as frames:
        for index,(frame,row) in enumerate(zip(frames,track['rows'])):
            if cancel():raise Cancelled()
            if row['accepted'] or row.get('reconstructed'):
                origin=np.array([row['x'],row['y']]);center=origin.copy()
                for _ in range(4):
                    image=messaging_frame(transform(frame,center,scale,(width,height)),color)
                    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
                    border=np.concatenate([gray[0],gray[-1],gray[:,0],gray[:,-1]])
                    background=float(np.median(border));peak=float(gray.max()-background)
                    noise=max(128,float(np.median(np.abs(border-background))*1.4826))
                    if peak<max(300,noise*5):break
                    weights=np.maximum(gray-background-peak*(.8 if full_frame else .2),0);mass=float(weights.sum())
                    if mass<=0:break
                    delta=np.array([(xx*weights).sum()/mass-width/2,(yy*weights).sum()/mass-height/2])/scale
                    if np.linalg.norm(center+delta-origin)>min(cw,ch)*.4:break
                    center+=delta
                    if np.linalg.norm(delta)<.001:break
                row['detected_x'],row['detected_y']=map(float,origin)
                row['x'],row['y']=map(float,center)
            progress(.7+.29*(index+1)/len(track['rows']),f'Refining visible center {index+1} / {len(track["rows"])}')
    track['centering']='bright reference core' if full_frame else 'rendered bright body'
    track['centering_version']=3 if full_frame else 2

def rendered_frames(path,track,zoom=1,cancel=lambda:False):
    """The same native-rate timeline and crop for MP4 and image sequences."""
    if not(1<=zoom<=16):raise ValueError('Zoom must be between 1 and 16.')
    if track.get('multiple_points') and track.get('transform_model')!='tripod_image_pose_v6':
        # Saved centroid-based tracks must be remeasured from actual source images.
        from .multipoint import track_multiple_video
        points=track.get('selected_points') or track['reference_points']
        options=dict(points=points,point=points[0],start=track['start'],mode=track.get('mode','bright'),
                     radius=100,crop_box=track.get('crop_box'),output_width=track.get('output_width'),output_height=track.get('output_height'))
        track=track_multiple_video(path,options,cancel=cancel)
    width,height,zoom=output_geometry(track,zoom)
    start=track['start'];fps=track['fps']
    total=math.ceil((track['duration']-start)*fps-fps*1e-6)
    frames=source_frames_16bit(path,track,cancel)
    count=0;previous=None;previous_row=None
    reference=track['rows'][0]
    def render(frame,row):
        if row.get('camera_transform') is not None:
            matrix=np.array(row['camera_transform'],np.float64)
            if (matrix.shape!=(3,3) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[2],[0,0,1],atol=1e-8)
                or not np.allclose(matrix[:2,:2].T@matrix[:2,:2],np.eye(2),atol=1e-8)
                or not np.isclose(np.linalg.det(matrix[:2,:2]),1,atol=1e-8)):
                raise ValueError('This saved transform distorts the image. Track the references again.')
            if track.get('crop_box'):
                x,y,_,_=track['crop_box'];matrix=np.array([[zoom,0,-x*zoom],[0,zoom,-y*zoom],[0,0,1.]])@matrix
            return cv2.warpAffine(frame,matrix[:2],(width,height),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
        if track.get('center_selection',True):point=(row['x'],row['y'])
        else:point=(width/2+row['x']-reference['x'],height/2+row['y']-reference['y'])
        return transform(frame,point,zoom,(width,height))
    try:
        for row in track['rows']:
            if cancel():raise Cancelled()
            frame=next(frames,None)
            if frame is None:raise ValueError('Original source decode ended before the recorded track.')
            rendered=None
            while previous is not None and count<total and start+count/fps<row['time']-1e-6:
                if cancel():raise Cancelled()
                if rendered is None:rendered=render(previous,previous_row)
                yield rendered
                count+=1
            previous=frame;previous_row=row
        rendered=None
        while previous is not None and count<total:
            if cancel():raise Cancelled()
            if rendered is None:rendered=render(previous,previous_row)
            yield rendered
            count+=1
    finally:frames.close()

def export_frames(path,track,destination,zoom=1,progress=lambda *_:None,cancel=lambda:False):
    """Losslessly compressed 16-bit PNGs and exact playback cadence."""
    destination=Path(destination);temporary=destination.with_suffix('.partial.zip')
    width,height,_=output_geometry(track,zoom);fps=track['fps']
    total=math.ceil((track['duration']-track['start'])*fps-fps*1e-6);count=0
    try:
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
            with closing(rendered_frames(path,track,zoom,cancel)) as frames:
                for count,frame in enumerate(frames,1):
                    ok,image=cv2.imencode('.png',frame,[cv2.IMWRITE_PNG_COMPRESSION,6])
                    if not ok:raise ValueError('Could not encode PNG frame.')
                    archive.writestr(f'frames/frame_{count:06d}.png',image.tobytes())
                    progress(count/total,f'Saving frame {count} / {total}')
            if cancel():raise Cancelled()
            archive.writestr('timing.json',json.dumps(dict(fps=fps,rate=track['rate'],frames=count,
                width=width,height=height,source_start_seconds=track['start'],duration_seconds=count/fps,color=source_color(path),
                pattern='frames/frame_%06d.png',first_frame=1,bit_depth=16,compression='lossless PNG',source='original',
                timing='Frame N starts at (N - 1) / fps seconds. Matches the video export timeline.'),indent=2))
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return dict(frames=count,fps=fps,rate=track['rate'],duration=count/fps,path=str(destination))

def messaging_frame(frame,color):
    """Map HDR into SDR for broad H.264 playback; leave SDR transfer unchanged."""
    transfer=color.get('color_transfer')
    if transfer not in ('arib-std-b67','smpte2084'):
        return frame
    rgb=frame[:,:,::-1].astype(np.float32)/65535
    if transfer=='arib-std-b67':
        a=.17883277;b=1-4*a;c=.5-a*np.log(4*a)
        linear=np.where(rgb<=.5,rgb*rgb/3,(np.exp((rgb-c)/a)+b)/12)
        luminance=np.maximum(linear@np.array([.2627,.6780,.0593],np.float32),1e-9)
        linear*=np.power(luminance,.2)[:,:,None]*10
        white=10
    else:
        m1=2610/16384;m2=2523/32;c1=3424/4096;c2=2413/128;c3=2392/128
        power=rgb**(1/m2)
        linear=(np.maximum(power-c1,0)/np.maximum(c2-c3*power,1e-9))**(1/m1)*100
        white=100
    if color.get('color_primaries')=='bt2020':
        matrix=np.array([[1.660491,-.587641,-.072850],[-.124550,1.132900,-.008350],[-.018151,-.100579,1.118730]],np.float32)
        linear=linear@matrix.T
    linear=np.maximum(linear,0)
    luminance=np.maximum(linear@np.array([.2126,.7152,.0722],np.float32),1e-9)
    mapped=luminance*(1+luminance/(white*white))/(1+luminance)
    linear*= (mapped/luminance)[:,:,None]
    rgb=np.where(linear<.018,4.5*linear,1.099*np.power(linear,.45)-.099)
    return np.ascontiguousarray(np.rint(np.clip(rgb[:,:,::-1],0,1)*65535).astype(np.uint16))

def export_video(path,track,destination,zoom=1,progress=lambda *_:None,cancel=lambda:False,lossless=True):
    if not(1<=zoom<=16):raise ValueError('Zoom must be between 1 and 16.')
    destination=Path(destination);temporary=destination.with_name(destination.stem+('.partial.mkv' if lossless else '.partial.mp4'))
    width,height,_=output_geometry(track,zoom);start=track['start'];fps=track['fps']
    total=math.ceil((track['duration']-start)*fps-fps*1e-6)
    args=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y','-f','rawvideo','-pix_fmt','bgr48le','-s',f'{width}x{height}','-r',track['rate'],'-i','pipe:0','-ss',str(start),'-i',str(original_source(path)),'-map','0:v:0','-map','1:a:0?','-c:v','ffv1','-level','3','-coder','1','-context','1','-slicecrc','1','-threads','2','-pix_fmt','gbrp16le','-c:a','flac','-sample_fmt','s32','-shortest',str(temporary)]
    if not lossless:
        codec=args.index('-c:v')
        args[codec:]=['-c:v','libx264','-preset','medium','-crf','10' if track.get('multiple_points') else '16','-threads','2',
            '-profile:v','high','-level:v','4.2','-pix_fmt','yuv420p','-tag:v','avc1',
            '-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart',str(temporary)]
    color=source_color(path)
    output_color=color if lossless else dict(color_primaries='bt709',color_transfer='bt709')
    parameters=['range=full','colorspace=gbr']
    for key,parameter in [('color_primaries','color_primaries'),('color_transfer','color_trc')]:
        if output_color.get(key) not in (None,'unknown','unspecified'):parameters.append(f'{parameter}={output_color[key]}')
    filters='setparams='+':'.join(parameters)
    if not lossless:filters+=',scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,setparams=range=limited:colorspace=bt709:color_primaries=bt709:color_trc=bt709'
    args[-1:-1]=['-vf',filters]
    # File-backed stderr cannot fill a pipe while frames are streamed.
    with temporary.with_suffix('.log').open('w+b') as log:
        proc=subprocess.Popen(args,stdin=subprocess.PIPE,stderr=log,stdout=subprocess.DEVNULL)
        count=0
        try:
            with closing(rendered_frames(path,track,zoom,cancel)) as frames:
                for count,rendered in enumerate(frames,1):
                    proc.stdin.write((rendered if lossless else messaging_frame(rendered,color)).tobytes())
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
            if proc.stdin and not proc.stdin.closed:proc.stdin.close()
    temporary.with_suffix('.log').unlink(missing_ok=True)
    return dict(frames=count,fps=fps,duration=count/fps,path=str(destination))
