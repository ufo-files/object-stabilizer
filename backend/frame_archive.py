"""Export each decoded MP4 frame once as a full-resolution lossless PNG."""
from pathlib import Path
import csv
import io
import json
import subprocess
import tempfile
import zipfile


def build(video, report_path, destination, progress=lambda *_: None, long_edge=0):
    video, destination = Path(video), Path(destination)
    report = json.loads(Path(report_path).read_text())
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate:frame=best_effort_timestamp_time,pkt_duration_time',
        '-of', 'json', str(video)]))
    stream, timing = probe['streams'][0], probe['frames']
    if long_edge not in (0,1920,3840):raise ValueError('Unsupported PNG export size.')
    source_width, source_height = stream['width'], stream['height']
    scale=[]
    if long_edge:
        scale=['-vf',f'scale={long_edge}:-1:flags=lanczos' if source_width>=source_height else f'scale=-1:{long_edge}:flags=lanczos']
    expected = report['output_frames']
    if len(timing) != expected:
        raise ValueError('Video frame count does not match its audit.')
    with tempfile.TemporaryDirectory(prefix='frame-export-', dir=destination.parent) as directory:
        directory = Path(directory)
        progress(.05, 'Extracting full-resolution PNG frames')
        subprocess.run([
            'ffmpeg', '-v', 'error', '-nostdin', '-i', str(video), '-map', '0:v:0',
            *scale, '-fps_mode', 'passthrough', '-start_number', '1', '-threads', '2',
            str(directory/'frame-%08d.png')], check=True, capture_output=True)
        frames = sorted(directory.glob('frame-*.png'))
        if len(frames) != expected:
            raise ValueError('Frame extraction dropped or duplicated video frames.')
        # PNG IHDR records the actual encoded dimensions, including rounding.
        import struct
        with frames[0].open('rb') as first_png:
            first_png.seek(16)
            width,height=struct.unpack('>II',first_png.read(8))
        manifest = io.StringIO()
        writer = csv.writer(manifest)
        writer.writerow(['file', 'export_frame', 'source_frame', 'timestamp_seconds'])
        first = report.get('start_frame', 0)
        for index, (frame, timestamp) in enumerate(zip(frames, timing)):
            writer.writerow([frame.name, index+1, first+index+1, timestamp['best_effort_timestamp_time']])
        metadata = dict(variant=report.get('variant','stabilized'),model=report.get('model'),parent_export=report.get('parent_export'),width=width, height=height, source_width=source_width, source_height=source_height, resize='Lanczos enlargement/resizing; no AI or new captured detail' if long_edge else 'Native video pixels',
                        fps=stream['avg_frame_rate'], frames=expected,
                        frame_numbering='1-based',
                        description='One lossless PNG per decoded video frame, at the selected dimensions. No frame-rate conversion. Cropping follows the video export; optional resizing is recorded separately.')
        temporary = directory/'frames.zip'
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            archive.writestr('metadata.json', json.dumps(metadata, indent=2))
            if report.get('variant')=='ai-enhanced':archive.writestr('enhancement.json',json.dumps(report,indent=2))
            archive.writestr('frames.csv', manifest.getvalue())
            for index, frame in enumerate(frames):
                archive.write(frame, frame.name)
                progress(.5+.5*(index+1)/expected, f'Packing frame {index+1} of {expected}')
        temporary.replace(destination)
    return dict(frames=expected, width=width, height=height, fps=stream['avg_frame_rate'])
