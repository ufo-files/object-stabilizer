import unittest
import cv2
import numpy as np
from backend.tracking import BrightPointTracker,FlowTracker,repair_gaps
from backend.processing import transform

def point_image(x,y,brightness=180,seed=0,distractor=None):
    yy,xx=np.mgrid[:160,:240];rng=np.random.default_rng(seed)
    image=40+xx*.05+rng.normal(0,2,xx.shape)+brightness*np.exp(-((xx-x)**2+(yy-y)**2)/(2*1.3**2))
    if distractor:image+=210*np.exp(-((xx-distractor[0])**2+(yy-distractor[1])**2)/(2*1.3**2))
    return np.clip(image,0,255).astype(np.uint8)

class TrackingTests(unittest.TestCase):
    def test_subpixel_moving_point_with_noise_and_exposure_change(self):
        t=BrightPointTracker(point_image(50.3,60.7),(50,61),24,40)
        errors=[]
        for i in range(1,30):
            x,y=50.3+i*1.2,60.7+i*.7
            r=t.update(point_image(x,y,brightness=120+i,seed=i))
            self.assertTrue(r.accepted);errors.append(np.hypot(r.x-x,r.y-y))
        self.assertLess(max(errors),.3)
    def test_nearby_brighter_point_does_not_steal_lock(self):
        t=BrightPointTracker(point_image(80,80),(80,80),24,50)
        r=t.update(point_image(82,80,distractor=(105,80)))
        self.assertTrue(r.accepted);self.assertLess(abs(r.x-82),.3)
    def test_missing_target_holds_and_recovers(self):
        t=BrightPointTracker(point_image(80,80),(80,80),24,50)
        lost=t.update(np.full((160,240),40,np.uint8))
        self.assertFalse(lost.accepted);self.assertLess(abs(lost.x-80),.3)
        found=t.update(point_image(83,82,seed=3));self.assertTrue(found.accepted)
        self.assertLess(np.hypot(found.x-83,found.y-82),.3)
    def test_distant_distractor_rejected(self):
        t=BrightPointTracker(point_image(80,80),(80,80),12,100)
        r=t.update(point_image(140,80));self.assertFalse(r.accepted)
    def test_blank_initial_point_rejected(self):
        with self.assertRaises(ValueError):BrightPointTracker(np.full((160,240),40,np.uint8),(80,80))
    def test_flow_tracks_texture_without_following_background(self):
        rng=np.random.default_rng(5);patch=rng.integers(0,255,(40,40),dtype=np.uint8)
        first=np.full((160,240),30,np.uint8);first[60:100,80:120]=patch
        t=FlowTracker(first,(100,80),40,30)
        for i in range(1,8):
            f=np.full_like(first,30);f[60+i:100+i,80+2*i:120+2*i]=patch
            r=t.update(f);self.assertTrue(r.accepted)
            self.assertLess(np.hypot(r.x-(100+2*i),r.y-(80+i)),.5)
    def test_affine_transform_centers_exact_point_at_zoom(self):
        f=point_image(70,55,seed=1);out=transform(f,(70,55),8)
        _,_,_,peak=cv2.minMaxLoc(cv2.GaussianBlur(out,(0,0),1))
        self.assertLess(np.hypot(peak[0]-120,peak[1]-80),2)
    def test_only_short_bounded_gaps_are_reconstructed(self):
        rows=[dict(time=i/15,x=0 if i<3 else 6,y=0,accepted=i in (0,3),confidence=1) for i in range(4)]
        repaired=repair_gaps(rows);self.assertAlmostEqual(repaired[1]['x'],2);self.assertAlmostEqual(repaired[2]['x'],4)
        rows[-1]['time']=2;self.assertEqual(repair_gaps(rows)[1]['x'],0)
if __name__=='__main__':unittest.main()
