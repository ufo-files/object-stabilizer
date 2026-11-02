"""Local Real-ESRGAN derivatives; originals and stabilization audits stay immutable.

SRVGG architecture follows Xintao Wang's Real-ESRGAN implementation (BSD-3).
See vendor/Real-ESRGAN-LICENSE. Model is framewise, not a temporal VSR model.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import tempfile

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODEL='realesr-general-x4v3'
ENGINE='realesrgan-derivative-1'
WEIGHTS={
 'realesr-general-x4v3.pth':'8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292',
 'realesr-general-wdn-x4v3.pth':'1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca',
}


def status():
    return dict(model=MODEL,available=all((ROOT/'.models/realesrgan'/name).is_file() for name in WEIGHTS),
                framewise=True,scales=[1,2],source='https://github.com/xinntao/Real-ESRGAN')


def model(denoise=.5,device=None):
    import torch
    from torch import nn
    from torch.nn import functional as F
    class SRVGG(nn.Module):
        def __init__(self):
            super().__init__()
            self.body=nn.ModuleList([nn.Conv2d(3,64,3,1,1),nn.PReLU(64)])
            for _ in range(32):self.body.extend([nn.Conv2d(64,64,3,1,1),nn.PReLU(64)])
            self.body.append(nn.Conv2d(64,48,3,1,1))
            self.upsampler=nn.PixelShuffle(4)
        def forward(self,x):
            out=x
            for layer in self.body:out=layer(out)
            return self.upsampler(out)+F.interpolate(x,scale_factor=4,mode='nearest')
    if not 0<=denoise<=1:raise ValueError('Denoise strength must be between zero and one.')
    params=[]
    for name,digest in WEIGHTS.items():
        path=ROOT/'.models/realesrgan'/name
        if not path.is_file():raise ValueError('Real-ESRGAN model is not installed. Run scripts/setup_enhancement.py.')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Real-ESRGAN model checksum does not match.')
        params.append(torch.load(path,map_location='cpu',weights_only=True)['params'])
    weights={key:denoise*params[0][key]+(1-denoise)*params[1][key] for key in params[0]}
    torch.set_num_threads(4)
    device=device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    net=SRVGG().eval();net.load_state_dict(weights,strict=True)
    return net.to(device),device


def enhance_image(rgb,net,device,scale=2,tile=256):
    """Overlap exceeds the 34-pixel receptive radius; discard overlap at seams."""
    import torch
    if scale not in (1,2):raise ValueError('Enhancement scale must be 1 or 2.')
    height,width=rgb.shape[:2]
    result=np.empty((height*scale,width*scale,3),np.uint8)
    pad=40
    with torch.inference_mode():
        for y in range(0,height,tile):
            for x in range(0,width,tile):
                x1,y1=min(width,x+tile),min(height,y+tile)
                ax,ay=max(0,x-pad),max(0,y-pad)
                bx,by=min(width,x1+pad),min(height,y1+pad)
                tensor=torch.from_numpy(np.ascontiguousarray(rgb[ay:by,ax:bx].transpose(2,0,1))).unsqueeze(0).to(device,dtype=torch.float32)/255
                output=net(tensor).clamp(0,1).squeeze(0).permute(1,2,0).cpu().numpy()
                output=np.rint(output*255).astype(np.uint8)
                output=cv2.resize(output,((bx-ax)*scale,(by-ay)*scale),interpolation=cv2.INTER_LANCZOS4)
                result[y*scale:y1*scale,x*scale:x1*scale]=output[(y-ay)*scale:(y1-ay)*scale,(x-ax)*scale:(x1-ax)*scale]
    return result


def probe(video):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
        'stream=width,height,avg_frame_rate:frame=best_effort_timestamp_time','-of','json',str(video)]))


def light_audit(video,parent,width,height):
    if parent.get('engine') not in ('light-center-1','moving-light-segment-2') or not parent.get('anchor'):
        return dict(available=False,reason='No selected-light reference in the parent report.')
    from .point_lock import yuv_frames
    from .light_segment import moment,patch_at
    native_width=parent['output_width'];native_height=parent['output_height']
    factors=np.array([width/native_width,height/native_height])
    anchor=(np.array(parent['anchor'])+.5)*factors-.5
    radius=max(16,min(128,round(16*max(factors))))
    visibility=parent.get('audit',[])
    observations=[];previous=None;first=None
    for index,planes in enumerate(yuv_frames(video,width,height)):
        if index<len(visibility) and not visibility[index].get('visible',True):continue
        row=dict(frame=index,measurable=False)
        try:
            center=moment(patch_at(planes[0],anchor,radius))+np.rint(anchor)-radius
            if first is None:first=center
            row.update(measurable=True,center_x=float(center[0]),center_y=float(center[1]),
                       drift_pixels=float(np.linalg.norm(center-first)),
                       step_pixels=None if previous is None else float(np.linalg.norm(center-previous)))
            previous=center
        except ValueError:previous=None
        observations.append(row)
    measured=[row for row in observations if row['measurable']]
    return dict(available=True,reference='First measurable enhanced center, not proof of object structure.',
                measured_frames=len(measured),unmeasurable_frames=len(observations)-len(measured),
                max_center_drift_pixels=max((row['drift_pixels'] for row in measured),default=None),
                max_center_step_pixels=max((row['step_pixels'] for row in measured if row['step_pixels'] is not None),default=None),
                zero_motion_passed=bool(measured) and len(measured)==len(observations) and all(row['drift_pixels']==0 for row in measured),
                observations=observations)


def file_sha256(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def build(video,report_path,destination,scale=2,denoise=.5,progress=lambda *_:None,min_long_edge=0):
    video,destination=Path(video),Path(destination)
    parent=json.loads(Path(report_path).read_text())
    before=probe(video);stream=before['streams'][0]
    width,height=stream['width'],stream['height'];count=len(before['frames'])
    if count!=parent['output_frames']:raise ValueError('Stabilized frame count does not match its report.')
    out_width,out_height=width*scale,height*scale
    if min_long_edge and max(out_width,out_height)<min_long_edge:
        factor=min_long_edge/max(out_width,out_height)
        out_width=max(2,round(out_width*factor/2)*2);out_height=max(2,round(out_height*factor/2)*2)
    progress(.01,'Loading verified Real-ESRGAN model')
    net,device=model(denoise)
    with tempfile.TemporaryDirectory(prefix='ai-export-',dir=destination.parent) as directory:
        temporary=Path(directory)/'enhanced.mp4'
        decoder=subprocess.Popen(['ffmpeg','-v','error','-nostdin','-i',str(video),'-map','0:v:0',
            '-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        # Audio is copied from the stabilized segment. No synthesis/interpolation.
        encoder=subprocess.Popen(['ffmpeg','-v','error','-nostdin','-y','-f','rawvideo','-pix_fmt','rgb24',
            '-s',f'{out_width}x{out_height}','-r',stream['avg_frame_rate'],'-i','-',
            '-i',str(video),'-map','0:v:0','-map','1:a:0?','-c:a','copy','-c:v','libx264','-qp','1',
            '-preset','fast','-pix_fmt','yuv420p','-threads','4','-map_metadata','-1',
            '-metadata',f'comment=AI enhanced derivative; {MODEL}; denoise={denoise}; scale={scale}; source={video.name}',
            '-movflags','+faststart',str(temporary)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        observed=0
        try:
            while True:
                raw=decoder.stdout.read(width*height*3)
                if not raw:break
                if len(raw)!=width*height*3:raise ValueError('Incomplete stabilized frame.')
                rgb=np.frombuffer(raw,np.uint8).reshape(height,width,3)
                enhanced=enhance_image(rgb,net,device,scale,tile=768 if device=='mps' else 256)
                if (out_width,out_height)!=(width*scale,height*scale):
                    enhanced=cv2.resize(enhanced,(out_width,out_height),interpolation=cv2.INTER_LANCZOS4)
                encoder.stdin.write(enhanced.tobytes());observed+=1
                progress(.03+.87*observed/count,f'AI enhancing frame {observed} of {count}')
            if decoder.wait():raise ValueError('Cannot decode stabilized video.')
            encoder.stdin.close();error=encoder.stderr.read().decode()
            if encoder.wait():raise ValueError(error[-1000:])
        finally:
            decoder.stdout.close()
            if decoder.poll() is None:decoder.kill();decoder.wait()
            if encoder.poll() is None:encoder.kill();encoder.wait()
            encoder.stdin.close();encoder.stderr.close()
        progress(.92,'Checking enhanced frame count, resolution and timestamps')
        after=probe(temporary)
        if observed!=count or len(after['frames'])!=count:raise ValueError('AI export lost or duplicated frames.')
        if (after['streams'][0]['width'],after['streams'][0]['height'])!=(out_width,out_height):raise ValueError('AI export dimensions differ from requested scale.')
        timestamps=[float(row['best_effort_timestamp_time']) for row in before['frames']]
        if any(abs(float(row['best_effort_timestamp_time'])-t)>1e-5 for row,t in zip(after['frames'],timestamps)):
            raise ValueError('AI export changed frame timestamps.')
        report=dict(engine=ENGINE,variant='ai-enhanced',parent_export=video.stem,model=MODEL,
                    model_source='https://github.com/xinntao/Real-ESRGAN',source_export_sha256=file_sha256(video),model_sha256=WEIGHTS,
                    scale=scale,denoise_strength=denoise,device=device,framewise=True,
                    output_width=out_width,output_height=out_height,
                    model_output_width=width*scale,model_output_height=height*scale,
                    presentation_resize='Lanczos; adds no captured detail' if (out_width,out_height)!=(width*scale,height*scale) else None,
                    source_width=width,source_height=height,frames=count,output_frames=count,
                    start_frame=parent.get('start_frame',0),end_frame=parent.get('end_frame',count-1),
                    fps=stream['avg_frame_rate'],extension='mp4',
                    frame_count_passed=True,timestamps_passed=True,
                    note='AI-generated visual enhancement. Estimated detail is not additional captured evidence. '
                         'The parent stabilization audit does not certify the enhanced object shape or its residual motion.')
        progress(.97,'Measuring enhanced light-center movement')
        report['motion_audit']=light_audit(temporary,parent,out_width,out_height)
        temporary.replace(destination)
        destination.with_suffix('.json').write_text(json.dumps(report,indent=2))
    progress(1,'AI-enhanced video ready')
    return report
