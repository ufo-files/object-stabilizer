"""Independently verify every native PNG against its actual encoded video."""
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import cv2
import numpy as np


def audit(archive,original):
    counts={}
    with zipfile.ZipFile(archive) as zipped,tempfile.TemporaryDirectory() as directory:
        manifest=json.loads(zipped.read('manifest.json'))
        assert zipped.read(manifest['original_file'])==Path(original).read_bytes()
        assert manifest['ai']['presets']==dict(mild=.25,standard=.5,strong=.75)
        variants=[('stabilized','stabilized/video.mp4')]+[(f'ai-{p}',f'ai/{p}/video.mp4') for p in ('mild','standard','strong')]
        for name,video_path in variants:
            metadata=json.loads(zipped.read(f'frames/{name}/metadata.json'))
            prefix=f'frames/{name}/'
            names=sorted(n for n in zipped.namelist() if n.startswith(prefix) and n.endswith('.png'))
            count=manifest['frames'];assert len(names)==count
            width,height=metadata['width'],metadata['height']
            if name.startswith('ai-'):
                report=json.loads(zipped.read(f'ai/{name[3:]}/video.json'))
                assert report['timestamps_passed'] and report['frame_count_passed']
                assert report['denoise_strength']==manifest['ai']['presets'][name[3:]]
                assert max(width,height)>=1920
                assert report['output_width']==width and report['output_height']==height
            rows=list(csv.DictReader(io.StringIO(zipped.read(prefix+'frames.csv').decode())))
            assert len(rows)==count
            video=Path(directory)/'video.mp4';video.write_bytes(zipped.read(video_path))
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(video)]))
            assert len(probe['frames'])==count
            assert all(abs(float(r['timestamp_seconds'])-float(t['best_effort_timestamp_time']))<1e-6 for r,t in zip(rows,probe['frames']))
            process=subprocess.Popen(['ffmpeg','-v','error','-i',str(video),'-map','0:v:0','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
            try:
                for png in names:
                    raw=process.stdout.read(width*height*3);assert len(raw)==width*height*3
                    image=cv2.imdecode(np.frombuffer(zipped.read(png),np.uint8),1)[:,:,::-1]
                    assert np.array_equal(image,np.frombuffer(raw,np.uint8).reshape(height,width,3)),png
                assert not process.stdout.read(1);assert process.wait()==0
            finally:
                process.stdout.close()
                if process.poll() is None:process.kill();process.wait()
            counts[name]=dict(frames=count,width=width,height=height,all_pngs_match_video=True)
    return counts

if __name__=='__main__':print(json.dumps(audit(sys.argv[1],sys.argv[2])))
