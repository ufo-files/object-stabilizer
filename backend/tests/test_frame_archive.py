import csv
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

import cv2
import numpy as np
from backend.frame_archive import build


class FrameArchiveTests(unittest.TestCase):
    def test_requested_sizes_preserve_all_frames_and_fractional_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);video=root/'video.mp4';report=root/'video.json'
            frames=np.zeros((3,32,48,3),np.uint8)
            frames[:,8:24,12:36,:]=np.array([40,100,180])[:,None,None,None]
            subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','48x32','-r','60000/1001','-i','-','-c:v','libx264','-pix_fmt','yuv420p',str(video)],input=frames.tobytes(),check=True,capture_output=True)
            report.write_text(json.dumps(dict(output_frames=3,start_frame=9)))
            for edge,dimensions in ((0,(48,32)),(1920,(1920,1280)),(3840,(3840,2560))):
                archive=root/f'frames-{edge}.zip'
                result=build(video,report,archive,long_edge=edge)
                self.assertEqual((result['width'],result['height']),dimensions)
                with zipfile.ZipFile(archive) as zipped:
                    metadata=json.loads(zipped.read('metadata.json'))
                    self.assertEqual(metadata['fps'],'60000/1001')
                    rows=list(csv.DictReader(io.StringIO(zipped.read('frames.csv').decode())))
                    self.assertEqual([int(row['source_frame']) for row in rows],[10,11,12])
                    self.assertAlmostEqual(float(rows[-1]['timestamp_seconds']),2*1001/60000,places=5)
                    for row in rows:
                        image=cv2.imdecode(np.frombuffer(zipped.read(row['file']),np.uint8),1)
                        self.assertEqual((image.shape[1],image.shape[0]),dimensions)
