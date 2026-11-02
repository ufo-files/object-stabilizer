"""One ZIP with original, stabilized, and every AI denoise preset plus frames."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile

from . import enhancement,frame_archive

PRESETS={'mild':.25,'standard':.5,'strong':.75}
ENGINE='complete-export-bundle-1'


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def build(original,stabilized,report_path,destination,original_name=None,progress=lambda *_:None):
    original,stabilized,report_path,destination=map(Path,(original,stabilized,report_path,destination))
    # The stable export id is immutable. Cache completed derivatives for retries.
    work=destination.with_suffix('.contents');work.mkdir(exist_ok=True)
    parent=json.loads(report_path.read_text());source_hash=digest(stabilized)
    name=Path(original_name or '').name.replace('\\','_')
    if not name or name in ('.','..') or not Path(name).suffix:
        fmt=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=format_name','-of','json',str(original)]))['format']['format_name']
        extension='mp4' if 'mp4' in fmt or 'mov' in fmt else ('avi' if 'avi' in fmt else ('webm' if 'webm' in fmt else 'mkv'))
        name=f'original.{extension}'
    (work/'original').mkdir(exist_ok=True);shutil.copyfile(original,work/'original'/name)
    (work/'stabilized').mkdir(exist_ok=True);shutil.copyfile(stabilized,work/'stabilized/video.mp4')
    shutil.copyfile(report_path,work/'stabilized/audit.json')
    if report_path.with_suffix('.csv').is_file():shutil.copyfile(report_path.with_suffix('.csv'),work/'stabilized/motion.csv')
    progress(.01,'Original and stabilized videos preserved; preparing AI copies')
    variants={}
    for index,(preset,strength) in enumerate(PRESETS.items()):
        target=work/'ai'/preset/'video.mp4';target.parent.mkdir(parents=True,exist_ok=True)
        cached=None
        if target.is_file() and target.with_suffix('.json').is_file():
            cached=json.loads(target.with_suffix('.json').read_text())
            if not (cached.get('source_export_sha256')==source_hash and cached.get('engine')==enhancement.ENGINE
                    and cached.get('denoise_strength')==strength and cached.get('scale')==2
                    and max(cached.get('output_width',0),cached.get('output_height',0))>=1920):cached=None
        if cached is None:
            def update(value,message,i=index,p=preset):progress(.01+.69*(i+value)/3,f'{p.title()} AI: {message}')
            cached=enhancement.build(stabilized,report_path,target,scale=2,denoise=strength,min_long_edge=1920,progress=update)
        variants[preset]=cached
    # Native frames exactly match each video. Add large viewing copies only for
    # small stabilized crops; keep those separately labeled from native pixels.
    frame_sets=[('stabilized',stabilized,report_path,0)]
    if max(parent.get('output_width',1920),parent.get('output_height',1080))<1920:
        frame_sets.append(('stabilized-viewing-1920',stabilized,report_path,1920))
    frame_sets += [(f'ai-{name}',work/'ai'/name/'video.mp4',work/'ai'/name/'video.json',0) for name in PRESETS]
    for index,(label,video,report,edge) in enumerate(frame_sets):
        target=work/'frames'/label;target.parent.mkdir(exist_ok=True)
        if not target.is_dir():
            with tempfile.TemporaryDirectory(prefix='bundle-frames-',dir=work) as temporary:
                temporary=Path(temporary);archive=temporary/'frames.zip'
                def update(value,message,i=index,n=label):progress(.7+.25*(i+value)/len(frame_sets),f'{n}: {message}')
                frame_archive.build(video,report,archive,update,long_edge=edge)
                unpack=temporary/'unpacked';unpack.mkdir()
                with zipfile.ZipFile(archive) as zipped:zipped.extractall(unpack)
                unpack.replace(target)
    manifest=dict(engine=ENGINE,original_file=f'original/{name}',original_sha256=digest(original),
                  stabilized_sha256=source_hash,frames=parent['output_frames'],
                  frame_rate='Preserved from the stabilized video; no interpolated frames.',
                  ai=dict(model=enhancement.MODEL,presets=PRESETS,model_scale=2,minimum_viewing_long_edge=1920,
                          note='AI estimated detail. Small crops are enlarged with Lanczos after AI processing; resizing is recorded in each report.'),
                  variants=variants)
    (work/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (work/'README.txt').write_text(
        'ORIGINAL: unchanged uploaded video.\n'
        'STABILIZED: the same MP4 shown in the app, plus its spatial audit.\n'
        'AI: mild (0.25), standard (0.50), strong (0.75) Real-ESRGAN denoising, each saved separately.\n'
        'FRAMES: one PNG per frame, with CSV timestamps/source frame numbers. Native sets exactly match the decoded video pixels.\n'
        'Small stabilized crops also include a separate 1920-pixel viewing set.\n'
        'AI videos use 2x model inference and a minimum 1920-pixel long edge for viewing. Extra enlargement adds no recorded detail.\n'
        'AI processing is framewise. It may alter shape, brightness or apparent motion; parent stabilization measurements do not certify AI detail.\n'
        'See manifest.json and each AI video.json for settings, model hashes and provenance.\n')
    files=sorted(p for p in work.rglob('*') if p.is_file())
    temporary=destination.with_suffix('.zip.partial')
    try:
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as zipped:
            for index,path in enumerate(files):
                zipped.write(path,path.relative_to(work).as_posix())
                progress(.95+.05*(index+1)/len(files),f'Packaging file {index+1} of {len(files)}')
        temporary.replace(destination)
    finally:temporary.unlink(missing_ok=True)
    return dict(frames=parent['output_frames'],presets=list(PRESETS),files=len(files),bytes=destination.stat().st_size)
