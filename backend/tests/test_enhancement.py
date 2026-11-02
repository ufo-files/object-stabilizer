"""Acceptance with real official weights, never a mocked enhancement model."""
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
from backend import enhancement,frame_archive


@unittest.skipUnless(importlib.util.find_spec('torch') and enhancement.status()['available'],'Install optional AI requirements and weights')
class EnhancementTests(unittest.TestCase):
    def test_actual_model_tiling_has_no_visible_seam_and_is_not_plain_resizing(self):
        yy,xx=np.mgrid[:81,:97]
        a=np.stack([xx*2,yy*2,(xx+yy)],axis=2).clip(0,255).astype(np.uint8)
        a[35:45,40:50]=220
        net,device=enhancement.model(.5,device='cpu')
        for scale in (1,2):
            full=enhancement.enhance_image(a,net,device,scale,tile=256)
            tiled=enhancement.enhance_image(a,net,device,scale,tile=32)
            self.assertEqual(full.shape,(81*scale,97*scale,3))
            self.assertLessEqual(np.abs(full.astype(int)-tiled.astype(int)).max(),1)
            baseline=cv2.resize(a,(97*scale,81*scale),interpolation=cv2.INTER_LANCZOS4)
            self.assertGreater(np.abs(full.astype(float)-baseline).mean(),.1)

    def test_video_preserves_source_audio_timing_and_pngs_match_encoded_enhanced_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.mp4';report=root/'source.json'
            yy,xx=np.mgrid[:48,:64];image=np.rint(8+180*np.exp(-((xx-32)**2+(yy-24)**2)/5)).astype(np.uint8)
            images=np.repeat(np.stack((image,image,image),axis=2)[None],6,axis=0)
            subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','64x48','-r','60000/1001','-i','-',
                '-f','lavfi','-i','sine=frequency=440:sample_rate=48000','-shortest','-c:v','libx264','-qp','1','-pix_fmt','yuv420p','-c:a','aac',str(source)],input=images.tobytes(),check=True,capture_output=True)
            report.write_text(json.dumps(dict(output_frames=6,start_frame=9,end_frame=14)))
            hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,report)]
            output=root/'enhanced.mp4'
            result=enhancement.build(source,report,output,scale=2,denoise=.25)
            self.assertEqual(result['variant'],'ai-enhanced')
            self.assertEqual((result['output_width'],result['output_height']),(128,96))
            self.assertEqual(result['output_frames'],6)
            self.assertTrue(result['timestamps_passed'])
            self.assertEqual(hashes,[hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,report)])
            audio=lambda path:subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-map','0:a:0','-c:a','copy','-f','adts','-'])
            self.assertEqual(audio(source),audio(output),'Enhanced audio must be copied without re-encoding')
            archive=root/'frames.zip';frame_archive.build(output,output.with_suffix('.json'),archive)
            raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(output),'-map','0:v:0','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','-'])
            decoded=np.frombuffer(raw,np.uint8).reshape(6,96,128,3)
            with zipfile.ZipFile(archive) as zipped:
                metadata=json.loads(zipped.read('metadata.json'))
                self.assertEqual(metadata['variant'],'ai-enhanced')
                self.assertEqual(metadata['model'],enhancement.MODEL)
                self.assertEqual(json.loads(zipped.read('enhancement.json'))['scale'],2)
                names=sorted(n for n in zipped.namelist() if n.endswith('.png'));self.assertEqual(len(names),6)
                for index,name in enumerate(names):
                    png=cv2.imdecode(np.frombuffer(zipped.read(name),np.uint8),1)[:,:,::-1]
                    np.testing.assert_array_equal(png,decoded[index])
