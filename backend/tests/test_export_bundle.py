from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
import zipfile

import cv2
import numpy as np
from backend import enhancement,export_bundle


@unittest.skipUnless(importlib.util.find_spec('torch') and enhancement.status()['available'],'Install optional AI model')
class BundleTests(unittest.TestCase):
    def test_all_presets_native_frames_and_original_bytes_are_in_one_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);video=root/'stable.mp4';report=root/'stable.json'
            a=np.full((2,48,64,3),8,np.uint8);a[:,20:28,28:36]=180
            subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','64x48','-r','24','-i','-','-c:v','libx264','-qp','1','-pix_fmt','yuv420p',str(video)],input=a.tobytes(),check=True,capture_output=True)
            report.write_text(json.dumps(dict(output_frames=2,start_frame=9,end_frame=10,output_width=64,output_height=48)))
            before=hashlib.sha256(video.read_bytes()).hexdigest();archive=root/'complete.zip'
            result=export_bundle.build(video,video,report,archive,original_name='original.mp4')
            self.assertEqual(result['presets'],['mild','standard','strong'])
            self.assertEqual(before,hashlib.sha256(video.read_bytes()).hexdigest())
            with zipfile.ZipFile(archive) as zipped:
                self.assertEqual(zipped.read('original/original.mp4'),video.read_bytes())
                self.assertEqual(zipped.read('stabilized/video.mp4'),video.read_bytes())
                for name,strength in export_bundle.PRESETS.items():
                    settings=json.loads(zipped.read(f'ai/{name}/video.json'))
                    self.assertEqual(settings['denoise_strength'],strength)
                    self.assertEqual(settings['output_width'],1920)
                    self.assertEqual(settings['output_height'],1440)
                    names=[n for n in zipped.namelist() if n.startswith(f'frames/ai-{name}/') and n.endswith('.png')]
                    self.assertEqual(len(names),2)
                    image=cv2.imdecode(np.frombuffer(zipped.read(names[0]),np.uint8),1)
                    self.assertEqual(image.shape[:2],(1440,1920))
                for prefix in ['stabilized','stabilized-viewing-1920']:
                    self.assertEqual(len([n for n in zipped.namelist() if n.startswith(f'frames/{prefix}/') and n.endswith('.png')]),2)
            # A retry must reuse completed AI files, not change existing derivatives.
            cached=root/'complete.contents/ai/standard/video.mp4';mtime=cached.stat().st_mtime_ns
            export_bundle.build(video,video,report,root/'complete.zip',original_name='original.mp4')
            self.assertEqual(mtime,cached.stat().st_mtime_ns)
