"""Run: python -m backend.server. Local UI and API share one origin."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event,Lock
import json
import uuid
from fastapi import FastAPI,File,HTTPException,UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from .processing import metadata,track_video,export_video,Cancelled

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'.runs'
RUNS.mkdir(exist_ok=True)
app=FastAPI(title='Local Object Stabilizer')
executor=ThreadPoolExecutor(max_workers=1)
lock=Lock();jobs={};uploads={}

class TrackOptions(BaseModel):
    video_id:str
    point:tuple[float,float]
    start:float=Field(default=0,ge=0)
    size:int=Field(default=24,ge=8,le=128)
    radius:int=Field(default=100,ge=5,le=500)
    mode:str='bright'
class ExportOptions(BaseModel):
    zoom:float=Field(default=1,ge=1,le=16)

def job_record(job_id):
    with lock:record=jobs.get(job_id)
    if record is None:raise HTTPException(404,'Job not found')
    return record

def submit(kind,work):
    job_id=uuid.uuid4().hex;cancel=Event()
    record=dict(id=job_id,kind=kind,status='queued',progress=0,message='Queued',result=None,error=None,cancel=cancel)
    with lock:jobs[job_id]=record
    def update(progress,message):
        with lock:record.update(progress=progress,message=message)
    def run():
        try:
            with lock:record['status']='running'
            result=work(update,cancel.is_set,job_id)
            if cancel.is_set():raise Cancelled()
            with lock:record.update(status='complete',progress=1,message='Complete',result=result)
        except Cancelled:
            with lock:record.update(status='cancelled',message='Cancelled')
        except Exception as exc:
            with lock:record.update(status='failed',error=str(exc),message=str(exc))
    executor.submit(run)
    return dict(id=job_id)

@app.get('/api/health')
def health():return dict(engine='opencv',version=1)

@app.post('/api/videos')
async def upload_video(file:UploadFile=File(...)):
    video_id=uuid.uuid4().hex;folder=RUNS/video_id;folder.mkdir()
    path=folder/'source.video'
    try:
        total=0
        with path.open('wb') as out:
            while chunk:=await file.read(1024*1024):
                total+=len(chunk)
                if total>2*1024**3:raise ValueError('Maximum video size is 2 GB.')
                out.write(chunk)
        info=metadata(path)
        with lock:uploads[video_id]=path
        return dict(id=video_id,**info)
    except Exception as exc:
        path.unlink(missing_ok=True);folder.rmdir();raise HTTPException(400,str(exc))
    finally:await file.close()

@app.post('/api/track')
def track(options:TrackOptions):
    with lock:path=uploads.get(options.video_id)
    if path is None:raise HTTPException(404,'Upload this video again after restarting the service.')
    if options.mode not in ('bright','flow'):raise HTTPException(400,'Unknown tracker mode')
    def work(progress,cancel,job_id):
        result=track_video(path,options.model_dump(),progress,cancel)
        result['video_id']=options.video_id
        (path.parent/f'{job_id}.json').write_text(json.dumps(result,indent=2))
        return result
    return submit('track',work)

@app.get('/api/jobs/{job_id}')
def get_job(job_id:str):
    record=job_record(job_id)
    with lock:return {k:v for k,v in record.items() if k!='cancel'}

@app.post('/api/jobs/{job_id}/cancel')
def cancel_job(job_id:str):
    job_record(job_id)['cancel'].set();return dict(cancelled=True)

@app.post('/api/jobs/{job_id}/export')
def export(job_id:str,options:ExportOptions):
    record=job_record(job_id)
    if record['kind']!='track' or record['status']!='complete':raise HTTPException(409,'Finish tracking before export.')
    track=record['result']
    with lock:path=uploads.get(track['video_id'])
    if path is None:raise HTTPException(404,'Source video unavailable')
    def work(progress,cancel,export_id):
        destination=path.parent/f'{export_id}.mp4'
        result=export_video(path,track,destination,options.zoom,progress,cancel)
        return dict(frames=result['frames'],fps=result['fps'],duration=result['duration'],download=f'/api/jobs/{export_id}/video')
    return submit('export',work)

@app.get('/api/jobs/{job_id}/video')
def video(job_id:str):
    record=job_record(job_id)
    if record['kind']!='export' or record['status']!='complete':raise HTTPException(404,'Export not ready')
    # Resolve only an internal generated export ID, never user-provided paths.
    paths=list(RUNS.glob(f'*/{record["id"]}.mp4'))
    if not paths:raise HTTPException(404,'Export missing')
    return FileResponse(paths[0],media_type='video/mp4',filename='stabilized.mp4')

app.mount('/',StaticFiles(directory=ROOT/'dist',html=True),name='ui')
if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=8080)
