"""Local tripod stabilizer: a single decode/register/render pipeline."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import uuid
import os
import json
from typing import Literal
import cv2
from fastapi import FastAPI,File,UploadFile,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,FiniteFloat,Field
from starlette.concurrency import run_in_threadpool
from .tripod import info
from . import light_segment, frame_archive, enhancement, export_bundle
from .light_center import stabilize as stabilize_light,reference
from .scene_motion import inspect as inspect_scene
from .scene_pipeline import ENGINE,stabilize as stabilize_scene

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'.runs'/'tripod-rebuild';RUNS.mkdir(parents=True,exist_ok=True)
app=FastAPI(title='Tripod Stabilizer')
executor=ThreadPoolExecutor(max_workers=1)
enhancement_executor=ThreadPoolExecutor(max_workers=1)
jobs={}
bundle_jobs={}

def folder(identifier):
    if len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier): raise HTTPException(404)
    path=RUNS/identifier
    if not path.is_dir(): raise HTTPException(404)
    return path

@app.middleware('http')
async def fresh(request,call_next):
    response=await call_next(request);response.headers['Cache-Control']='no-store';return response

@app.get('/api/health')
def health(): return dict(engine=ENGINE,version=19,default_mode='scene',features=['automatic_scene','sparse_light_fields','distributed_background_features','rigid_camera_correction','encoded_preview','independent_export_audit','selected_light_crop','bidirectional_moving_light','scrub_reference_frame','visible_segment_export','complete_export_zip','local_ai_enhancement','three_denoise_presets'])

@app.post('/api/videos')
async def upload(file:UploadFile=File(...)):
    identifier=uuid.uuid4().hex;path=RUNS/identifier;path.mkdir()
    with (path/'source.video').open('wb') as handle:
        total=0
        while chunk:=await file.read(1024*1024):
            total+=len(chunk)
            if total>2*1024**3: raise HTTPException(413,'Maximum file size is 2 GB.')
            handle.write(chunk)
    try:
        metadata=await run_in_threadpool(light_segment.describe,path/'source.video')
        metadata['original_filename']=Path(file.filename or 'original-video.video').name
        cap=cv2.VideoCapture(str(path/'source.video'));ok,frame=cap.read();cap.release()
        if not ok: raise ValueError('Cannot decode video.')
        cv2.imwrite(str(path/'first.png'),frame)
        analysis=await run_in_threadpool(inspect_scene,path/'source.video')
        (path/'metadata.json').write_text(json.dumps(metadata))
    except Exception as exc: raise HTTPException(400,str(exc))
    return dict(id=identifier,first=f'/api/videos/{identifier}/first',analysis=analysis,**metadata)

@app.get('/api/videos/{identifier}/original')
def original(identifier:str):
    path=folder(identifier)
    metadata=json.loads((path/'metadata.json').read_text()) if (path/'metadata.json').exists() else {}
    return FileResponse(path/'source.video',filename=metadata.get('original_filename','original-video.video'))

@app.get('/api/videos/{identifier}/first')
def first(identifier:str): return FileResponse(folder(identifier)/'first.png')

@app.get('/api/videos/{identifier}/frames/{index}')
def selected_frame(identifier:str,index:int):
    path=folder(identifier)
    metadata=json.loads((path/'metadata.json').read_text()) if (path/'metadata.json').exists() else light_segment.describe(path/'source.video')
    if not 0 <= index < metadata['frame_count']:raise HTTPException(404,'Frame is outside the video.')
    image=path/f'frame-{index}.png'
    if not image.exists():
        try:
            temporary=path/f'frame-{index}-{uuid.uuid4().hex}.tmp'
            temporary.write_bytes(light_segment.frame_at(path/'source.video',index));temporary.replace(image)
        except ValueError as exc:raise HTTPException(400,str(exc))
    return FileResponse(image,media_type='image/png')

class Request(BaseModel):
    video_id:str
    mode:Literal['scene','light']|None=None
    box:tuple[FiniteFloat,FiniteFloat,FiniteFloat,FiniteFloat]|None=None
    crop_to_selection:bool=False
    reference_frame:int=Field(default=0,ge=0,strict=True)
    track_visible_segment:bool=False

@app.post('/api/reference')
def inspect_reference(options:Request):
    if options.box is None:raise HTTPException(422,'A reference box is required for selected-light mode.')
    try:
        if options.track_visible_segment or options.reference_frame:
            result=light_segment.analyse(folder(options.video_id)/'source.video',options.box,options.reference_frame)
            return dict(anchor=result['anchor'],window=[round(v) for v in result['anchor']],width=result['width'],height=result['height'],
                        **{key:result[key] for key in ('reference_frame','first_visible_frame','last_visible_frame','start_frame','end_frame','frame_count','boundary_reasons')})
        _,width,height,anchor,window,_=reference(folder(options.video_id)/'source.video',options.box)
        return dict(anchor=anchor.tolist(),window=window.tolist(),width=width,height=height)
    except ValueError as exc:raise HTTPException(400,str(exc))

@app.post('/api/stabilize')
def start(options:Request):
    mode=options.mode or ('light' if options.box is not None else 'scene')
    if mode=='light' and options.box is None:raise HTTPException(422,'A reference box is required for selected-light mode.')
    if mode!='light' and (options.reference_frame or options.track_visible_segment):raise HTTPException(422,'Reference frames and visible-segment tracking are available only in selected-light mode.')
    if mode!='light' and options.crop_to_selection:raise HTTPException(422,'Crop to selection is available only in selected-light mode.')
    path=folder(options.video_id);identifier=uuid.uuid4().hex
    job=dict(id=identifier,status='queued',progress=0.,message='Queued')
    jobs[identifier]=job
    def update(value,message): job.update(progress=value,message=message)
    def run():
        job['status']='running'
        try:
            output=path/f'{identifier}.mp4'
            report=(stabilize_scene(path/'source.video',output,update) if mode=='scene'
                    else light_segment.stabilize(path/'source.video',output,options.box,options.reference_frame,update,crop_to_selection=options.crop_to_selection)
                    if options.track_visible_segment or options.reference_frame
                    else stabilize_light(path/'source.video',output,options.box,update,crop_to_selection=options.crop_to_selection))
            fields=('extension','anchor','window','frames','output_frames','max_residual_pixels','max_step_pixels','zero_motion_passed','unmeasurable_frames')
            summary={key:report[key] for key in fields}
            if mode=='scene':summary.update({key:report[key] for key in ('analysis','motion_model','rigid_geometry_passed')})
            else:summary.update({key:report[key] for key in ('center_lock_passed','profile_metrics','crop_to_selection','crop_box','output_width','output_height','source_anchor')})
            if 'reference_frame' in report:summary.update({key:report[key] for key in ('reference_frame','first_visible_frame','last_visible_frame','start_frame','end_frame','source_frame_count','boundary_reasons')})
            encoded=info(output)
            summary.update(mode=mode,engine=report['engine'],output_width=encoded['width'],output_height=encoded['height'],fps=encoded['fps'],rate=encoded['rate'])
            job.update(status='complete',progress=1,message='Export rendered; inspect frame audit',result=dict(bundle_request=f'/api/results/{options.video_id}/{identifier}/bundle',enhancement_request=f'/api/results/{options.video_id}/{identifier}/enhance',frames_request=f'/api/results/{options.video_id}/{identifier}/frames',url=f'/api/results/{options.video_id}/{identifier}/mp4',report=f'/api/results/{options.video_id}/{identifier}/json',measurements=f'/api/results/{options.video_id}/{identifier}/csv',**summary))
        except Exception as exc: job.update(status='failed',message=str(exc))
    executor.submit(run)
    return dict(id=identifier)

@app.get('/api/jobs/{identifier}')
def status(identifier:str):
    if identifier not in jobs: raise HTTPException(404)
    return dict(jobs[identifier])

class FrameExportOptions(BaseModel):
    long_edge:Literal[0,1920,3840]=0

class EnhancementOptions(BaseModel):
    scale:Literal[1,2]=2
    denoise:Literal[.25,.5,.75]=.5

@app.get('/api/enhancement')
def enhancement_status():return enhancement.status()

def bundle_root():
    config=ROOT/'.export-storage.json'
    configured=os.environ.get('STABILIZER_EXPORT_ROOT')
    if not configured and config.exists():configured=json.loads(config.read_text()).get('root')
    path=Path(configured) if configured else RUNS/'bundles'
    if configured and not path.is_dir():raise ValueError('Export storage is unavailable. Reconnect the export drive and retry.')
    return path

@app.post('/api/results/{video}/{identifier}/bundle')
def complete_bundle(video:str,identifier:str):
    if len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier):raise HTTPException(404)
    path=folder(video)
    if not (path/f'{identifier}.mp4').is_file() or not (path/f'{identifier}.json').is_file():raise HTTPException(404)
    key=(video,identifier)
    previous=bundle_jobs.get(key)
    if previous in jobs and jobs[previous]['status'] in ('queued','running','complete'):return dict(id=previous)
    if not enhancement.status()['available']:raise HTTPException(503,'AI model is not installed. Run scripts/setup_enhancement.py.')
    job_id=uuid.uuid4().hex
    job=dict(id=job_id,status='queued',progress=0.,message='Queued complete export ZIP')
    jobs[job_id]=job;bundle_jobs[key]=job_id
    def update(value,message):job.update(progress=value,message=message)
    def run():
        job['status']='running'
        try:
            target=bundle_root()/video/f'{identifier}.bundle.zip'
            target.parent.mkdir(parents=True,exist_ok=True)
            details={}
            if not target.exists():
                metadata=json.loads((path/'metadata.json').read_text()) if (path/'metadata.json').exists() else {}
                details=export_bundle.build(path/'source.video',path/f'{identifier}.mp4',path/f'{identifier}.json',target,
                                            original_name=metadata.get('original_filename'),progress=update)
            job.update(status='complete',progress=1,message='Complete export ZIP ready',result=dict(url=f'/api/results/{video}/{identifier}/bundle.zip',**details))
        except Exception as exc:job.update(status='failed',message=str(exc))
    enhancement_executor.submit(run)
    return dict(id=job_id)

@app.post('/api/results/{video}/{identifier}/enhance')
def enhance_export(video:str,identifier:str,options:EnhancementOptions):
    if len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier):raise HTTPException(404)
    path=folder(video)
    if not (path/f'{identifier}.mp4').is_file() or not (path/f'{identifier}.json').is_file():raise HTTPException(404)
    if json.loads((path/f'{identifier}.json').read_text()).get('variant')=='ai-enhanced':raise HTTPException(422,'Choose the stabilized export as the enhancement source.')
    if not enhancement.status()['available']:raise HTTPException(503,'AI model is not installed. Run scripts/setup_enhancement.py.')
    export_id=uuid.uuid4().hex;job_id=uuid.uuid4().hex
    job=dict(id=job_id,status='queued',progress=0.,message='Queued local AI enhancement');jobs[job_id]=job
    def update(value,message):job.update(progress=value,message=message)
    def run():
        job['status']='running'
        try:
            report=enhancement.build(path/f'{identifier}.mp4',path/f'{identifier}.json',path/f'{export_id}.mp4',
                                     scale=options.scale,denoise=options.denoise,progress=update)
            job.update(status='complete',progress=1,message='AI-enhanced derivative ready',result=dict(
                url=f'/api/results/{video}/{export_id}/mp4',report=f'/api/results/{video}/{export_id}/json',
                frames_request=f'/api/results/{video}/{export_id}/frames',**report))
        except Exception as exc:job.update(status='failed',message=str(exc))
    enhancement_executor.submit(run)
    return dict(id=job_id)

@app.post('/api/results/{video}/{identifier}/frames')
def export_frames(video:str,identifier:str,options:FrameExportOptions|None=None):
    if len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier):raise HTTPException(404)
    path=folder(video)
    if not (path/f'{identifier}.mp4').is_file() or not (path/f'{identifier}.json').is_file():raise HTTPException(404)
    edge=options.long_edge if options else 0
    extension=f'frames-{edge}.zip' if edge else 'zip'
    job_id=uuid.uuid4().hex
    job=dict(id=job_id,status='queued',progress=0.,message='Queued frame export')
    jobs[job_id]=job
    def update(value,message):job.update(progress=value,message=message)
    def run():
        job['status']='running'
        try:
            archive=path/f'{identifier}.{extension}'
            if not archive.exists():
                details=frame_archive.build(path/f'{identifier}.mp4',path/f'{identifier}.json',archive,update,long_edge=edge)
            else:details={}
            job.update(status='complete',progress=1,message='PNG frames ready',result=dict(url=f'/api/results/{video}/{identifier}/{extension}',**details))
        except Exception as exc:job.update(status='failed',message=str(exc))
    executor.submit(run)
    return dict(id=job_id)

@app.post('/api/videos/{video}/original-frames')
def original_frames(video:str):
    path=folder(video);job_id=uuid.uuid4().hex
    job=dict(id=job_id,status='queued',progress=0.,message='Queued original frames');jobs[job_id]=job
    def update(value,message):job.update(progress=value,message=message)
    def run():
        job['status']='running'
        try:
            identifier='0'*32;archive=path/f'{identifier}.zip'
            if not archive.exists():
                metadata=light_segment.describe(path/'source.video')
                report=path/'original-frames.json'
                report.write_text(json.dumps(dict(output_frames=metadata['frame_count'],variant='original',start_frame=0)))
                frame_archive.build(path/'source.video',report,archive,update)
            job.update(status='complete',progress=1,message='Original PNG frames ready',result=dict(url=f'/api/results/{video}/{"0"*32}/zip'))
        except Exception as exc:job.update(status='failed',message=str(exc))
    executor.submit(run)
    return dict(id=job_id)

@app.get('/api/results/{video}/{identifier}/{extension}')
def result(video:str,identifier:str,extension:str):
    if len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier) or extension not in ('mp4','webm','json','csv','zip','frames-1920.zip','frames-3840.zip','bundle.zip'): raise HTTPException(404)
    path=folder(video)/f'{identifier}.{extension}'
    if extension=='bundle.zip':
        try:path=bundle_root()/video/f'{identifier}.bundle.zip'
        except ValueError as exc:raise HTTPException(503,str(exc))
    if not path.is_file(): raise HTTPException(404)
    return FileResponse(path,filename='ufo-video-exports.zip' if extension=='bundle.zip' else f'tripod-stabilized.{extension}')

app.mount('/',StaticFiles(directory=ROOT/'dist',html=True),name='ui')
