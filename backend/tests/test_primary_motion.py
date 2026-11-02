import unittest
import cv2
import numpy as np
from backend.primary_motion import PrimaryImageMotion,displacement


class PrimaryImageMotionTests(unittest.TestCase):
    def test_abrupt_camera_motion_is_not_smoothed_or_delayed(self):
        yy,xx=np.mgrid[:128,:192];motion=PrimaryImageMotion()
        offsets=[[0.,0.],[13.,-9.],[-12.,11.],[8.,7.],[0.,0.]]
        for i,(dx,dy) in enumerate(offsets):
            signal=1000+18000*(1+i*.05)*np.exp(-((xx-80-dx)**2+(yy-60-dy)**2)/14)
            image=np.repeat(signal.astype(np.uint16)[:,:,None],3,axis=2)
            motion.add(image,[80+dx,60+dy])
        positions,scores=motion.positions(np.arange(5)/30,[80.,60.])
        np.testing.assert_allclose(positions,np.array(offsets)+[80,60],atol=.05)
        self.assertTrue((scores>.98).all())

    def test_constant_image_remains_exactly_stationary(self):
        yy,xx=np.mgrid[:96,:128]
        frame=np.repeat((1000+20000*np.exp(-((xx-64)**2+(yy-48)**2)/10)).astype(np.uint16)[:,:,None],3,axis=2)
        motion=PrimaryImageMotion()
        for _ in range(4):motion.add(frame,[64,48])
        positions,_=motion.positions([0,.03,.08,.11],[64,48])
        np.testing.assert_array_equal(positions,[[64,48]]*4)

    def test_blank_primary_is_rejected_instead_of_being_held(self):
        with self.assertRaises(ValueError):PrimaryImageMotion().add(np.zeros((96,128,3),np.uint16),[64,48])

    def test_failed_absolute_match_does_not_replace_measured_camera_motion(self):
        motion=PrimaryImageMotion()
        motion.sequential=[np.array([0.,0.]),np.array([12.,-8.]),np.array([1.,2.])]
        motion.absolute=motion.sequential.copy();motion.absolute_scores=[1.,0.,1.];motion.scores=[1.,.9,.95]
        positions,_=motion.positions([0,.03,.07],[64,48])
        np.testing.assert_array_equal(positions,[[64,48],[76,40],[65,50]])

    def test_initial_image_matches_remove_accumulated_drift_without_smoothing_camera_jumps(self):
        camera=np.array([[0.,0.],[12.,-8.],[-10.,9.],[7.,3.],[1.,-2.]])
        motion=PrimaryImageMotion()
        motion.sequential=list(camera+np.arange(5)[:,None]*[.2,-.2])
        motion.absolute=list(camera);motion.absolute_scores=[1.]*5;motion.scores=[1.]*5
        positions,_=motion.positions(np.arange(5)/30,[64.,48.])
        self.assertLess(np.linalg.norm(positions-camera-[64,48],axis=1).max(),.4)
        np.testing.assert_array_equal(positions[0],[64,48])
