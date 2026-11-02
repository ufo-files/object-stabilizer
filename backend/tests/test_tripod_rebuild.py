"""Black-box checks for the replacement tripod pipeline."""
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from backend.tripod import Reference,stabilize,frames

class TripodRebuildTests(unittest.TestCase):
    def test_absolute_registration_does_not_accumulate_or_rotate(self):
        rng=np.random.default_rng(17)
        image=rng.integers(0,255,(240,320,3),dtype=np.uint8)
        tracker=Reference(image,(120,80,64,64))
        for dx,dy in [(0,0),(5,-3),(-8,4),(0,0),(5,-3)]:
            moved=cv2.warpAffine(image,np.float32([[1,0,dx],[0,1,dy]]),(320,240))
            displacement,score=tracker.locate(moved)
            np.testing.assert_allclose(displacement,[dx,dy],atol=.08)
            self.assertGreater(score,.99)

    def test_real_decode_encode_frame_count_and_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source.avi';destination=Path(directory)/'result.mp4'
            rng=np.random.default_rng(21)
            image=rng.integers(20,220,(240,320,3),dtype=np.uint8)
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'FFV1'),24,(320,240))
            motion=[(0,0),(7,-4),(-6,5),(4,3),(0,0)]
            for dx,dy in motion:writer.write(cv2.warpAffine(image,np.float32([[1,0,dx],[0,1,dy]]),(320,240)))
            writer.release()
            result=stabilize(source,destination,(120,80,64,64))
            self.assertEqual(result['frames'],len(motion));self.assertEqual(result['output_frames'],len(motion))
            for row,(dx,dy) in zip(result['rows'],motion):np.testing.assert_allclose([row['dx'],row['dy']],[dx,dy],atol=.08)
            decoded=list(frames(destination,320,240))
            for frame in decoded[1:]:
                self.assertLess(np.abs(frame[30:-30,30:-30].astype(float)-decoded[0][30:-30,30:-30]).mean(),3.)
            self.assertLess(result['max_residual_pixels'],.15)

    def test_blank_reference_fails(self):
        with self.assertRaises(ValueError):Reference(np.zeros((240,320,3),np.uint8),(120,80,64,64))

if __name__=='__main__':unittest.main()
