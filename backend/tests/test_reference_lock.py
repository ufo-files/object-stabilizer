"""A stationary star removes camera shake without recentering the scene."""
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from backend.processing import rendered_frames,refine_visible_centers
from backend.tracking import BrightPointTracker


class ReferenceLockTests(unittest.TestCase):
    def test_shaking_camera_locks_star_at_original_position_preserving_relative_motion(self):
        yy,xx=np.mgrid[:120,:200]
        rng=np.random.default_rng(42)
        frames=[];rows=[];tracker=None;expected=[]
        shifts=[(0,0), (3.4,-2.7), (-4.2,3.1), (1.6,4.3), (-2.5,-3.2), (4.1,1.7)]
        for i,(dx,dy) in enumerate(shifts):
            image=12+rng.normal(0,.8,xx.shape)
            image+=(160+20*np.sin(i))*np.exp(-((xx-45-dx)**2+(yy-35-dy)**2)/3.4)
            image+=200*np.exp(-((xx-140-dx-i*1.5)**2+(yy-70-dy)**2)/3.4)
            gray=np.clip(image,0,255).astype(np.uint8)
            if tracker is None:
                tracker=BrightPointTracker(gray,(45,35),16,20)
                x,y=tracker.initial
            else:
                observation=tracker.update(gray)
                self.assertTrue(observation.accepted)
                x,y=observation.x,observation.y
            rows.append(dict(time=i/30,frame=i,x=x+.4*np.sin(i),y=y+.3*np.cos(i),accepted=True))
            frames.append(cv2.cvtColor(gray.astype(np.uint16)*257,cv2.COLOR_GRAY2BGR))
            expected.append(140+i*1.5)
        track=dict(width=200,height=120,start=0,fps=30,duration=len(frames)/30,
                   center_selection=False,rows=rows)
        with patch('backend.processing.source_frames_16bit',side_effect=lambda *_:iter_frames(frames)),patch('backend.processing.source_color',return_value={}):
            refine_visible_centers('unused',track)
            self.assertIn('detected_x',track['rows'][0])
            output=list(rendered_frames('unused',track))
        self.assertEqual(len(output),len(frames))
        errors=[]
        for i,frame in enumerate(output):
            self.assertEqual(frame.shape,(120,200,3))
            gray=frame[:,:,0].astype(float)
            def centroid(x,y):
                region=gray[y-8:y+9,x-8:x+9]
                weights=np.maximum(region-np.median(region)-3000,0)
                sy,sx=np.indices(region.shape)
                return np.array([(sx*weights).sum()/weights.sum()+x-8,(sy*weights).sum()/weights.sum()+y-8])
            star=centroid(45,35);moving=centroid(round(expected[i]),70)
            errors.append(float(np.linalg.norm(star-[45,35])))
            self.assertLess(np.linalg.norm(moving-[expected[i],70]),.25)
        self.assertLess(max(errors),.2)
        np.testing.assert_array_equal(output[0],frames[0])
        print('Reference-lock maximum star residual:',round(max(errors),4),'source pixels')


def iter_frames(frames):
    yield from frames
