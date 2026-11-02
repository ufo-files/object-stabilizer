import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from validation.audit_stabilization import audit,measure


class ExportAuditTests(unittest.TestCase):
    def test_image_registration_detects_motion_independently_of_centroid(self):
        yy,xx=np.mgrid[:65,:65]
        first=np.exp(-((xx-32.)**2+(yy-32.)**2)/12).astype(np.float32)
        shifted=np.exp(-((xx-35.)**2+(yy-30.)**2)/12).astype(np.float32)
        measured=measure(first,shifted)
        self.assertTrue(measured['measurable'])
        self.assertAlmostEqual(measured['dx'],3,delta=.05)
        self.assertAlmostEqual(measured['dy'],-2,delta=.05)

    def test_blank_patches_cannot_pass_as_zero_motion(self):
        self.assertFalse(measure(np.zeros((65,65),np.float32),np.zeros((65,65),np.float32))['measurable'])

    def test_audit_reads_every_exported_frame_and_fails_a_single_jump(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);video=folder/'jump.avi'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),30,(128,96))
            yy,xx=np.mgrid[:96,:128]
            for dx in (0,0,3,0):
                gray=(15+220*np.exp(-((xx-64-dx)**2+(yy-48)**2)/12)).astype(np.uint8)
                writer.write(np.repeat(gray[:,:,None],3,axis=2))
            writer.release()
            report=audit(video,[[64,48]],folder/'report',.1)
            self.assertEqual(report['decoded_frames'],4)
            self.assertFalse(report['primary_passed'])
            self.assertGreater(report['references'][0]['maximum_measurable_step_pixels'],2.8)
            self.assertEqual(len((folder/'report'/'frames.csv').read_text().splitlines()),5)
            self.assertTrue((folder/'report'/'worst-frame-pairs.png').is_file())

    def test_identical_observable_images_pass_strict_zero_tolerance(self):
        yy,xx=np.mgrid[:65,:65]
        image=np.exp(-((xx-32.)**2+(yy-32.)**2)/12).astype(np.float32)
        result=measure(image,image.copy())
        self.assertTrue(result['measurable'])
        self.assertEqual(result['pixels'],0)
