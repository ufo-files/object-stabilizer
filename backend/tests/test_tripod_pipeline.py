"""End-to-end camera-vibration regression; compare output pixels, not tracker data."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from backend.processing import track_video,rendered_frames


class TripodPipelineTests(unittest.TestCase):
    def test_real_decode_track_render_cancels_vibration_on_each_matching_frame(self):
        rng=np.random.default_rng(718)
        yy,xx=np.mgrid[:240,:320]
        gray=rng.integers(7,18,(240,320)).astype(float)
        for x,y in [(260,80),(70,110)]:gray+=200*np.exp(-((xx-x)**2+(yy-y)**2)/5)
        scene=cv2.cvtColor(np.clip(gray,0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
        offsets=[(0,0),(2,-1),(-3,2),(1,3),(-2,-2),(7,-4),(-6,5),(0,0)]
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'tripod.avi'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'FFV1'),30,(320,240))
            self.assertTrue(writer.isOpened())
            for dx,dy in offsets:writer.write(cv2.warpAffine(scene,np.array([[1.,0,dx],[0,1.,dy]]),(320,240),flags=cv2.INTER_NEAREST))
            writer.release()
            track=track_video(source,dict(point=[260,80],points=[[260,80],[70,110]],center_selection=False,mode='bright'))
            self.assertFalse(track['correct_rotation'])
            output=list(rendered_frames(source,track))
            self.assertEqual(len(output),len(offsets))
            for index,frame in enumerate(output):
                np.testing.assert_array_equal(frame[40:-40,40:-40],output[0][40:-40,40:-40],err_msg=f'camera vibration remains in output frame {index}')
                np.testing.assert_array_equal(np.array(track['rows'][index]['camera_transform'])[:2,:2],np.eye(2))
