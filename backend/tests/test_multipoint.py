import unittest
from unittest.mock import patch
import cv2
import numpy as np
from backend.multipoint import registration_matrix,core_center,refine_anchor,rigid_trajectory
from backend.processing import rendered_frames


class MultiplePointTests(unittest.TestCase):
    def test_two_three_and_four_references_and_independent_motion(self):
        reference=np.array([[40.,40.],[270,45],[260,190],[50,180]])
        angle=.03
        motion=np.array([[np.cos(angle),-np.sin(angle),4.],[np.sin(angle),np.cos(angle),-3.],[0,0,1]])
        matrices=[motion]*3
        for count,motion in zip([2,3,4],matrices):
            current=cv2.perspectiveTransform(reference[None],motion)[0]
            inverse,error=registration_matrix(current[:count],reference[:count],320,240)
            self.assertLess(error,.001)
            np.testing.assert_allclose(cv2.perspectiveTransform(current[None],inverse)[0],reference,atol=.001)
            moving=np.array([[[155.,110.]]]);observed=cv2.perspectiveTransform(moving,motion)
            np.testing.assert_allclose(cv2.perspectiveTransform(observed,inverse),moving,atol=.001)

    def test_bad_geometry_is_rejected(self):
        with self.assertRaises(ValueError):registration_matrix([[10,10],[10,10]],[[10,10],[10,10]],320,240)
        matrix,error=registration_matrix([[10,10],[20,20],[30,30]],[[10,10],[20,20],[30,30]],320,240)
        np.testing.assert_allclose(matrix,np.eye(3),atol=1e-12)

    def test_disagreeing_references_cannot_stretch_any_part_of_frame(self):
        reference=np.array([[40.,40.],[270,45],[260,190],[50,180]])
        for count in (2,3,4):
            current=reference[:count]@np.array([[1.1,.03],[.02,.95]])+[4,-3]
            matrix,error=registration_matrix(current,reference[:count],1920,1080)
            self.assertGreater(error,1)
            np.testing.assert_allclose(matrix[:2,:2]@current[0]+matrix[:2,2],reference[0],atol=1e-10)
            np.testing.assert_allclose(matrix[:2,:2].T@matrix[:2,:2],np.eye(2),atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(matrix[:2,:2]),1)
            corners=np.array([[0.,0.],[1920,0],[1920,1080],[0,1080]])
            mapped=corners@matrix[:2,:2].T+matrix[:2,2]
            np.testing.assert_allclose(np.linalg.norm(mapped[:,None]-mapped,axis=2),np.linalg.norm(corners[:,None]-corners,axis=2),atol=1e-9)

    def test_full_frame_and_crop_share_the_same_reference_warp(self):
        yy,xx=np.mgrid[:64,:80]
        first=np.repeat(np.rint(1000+40000*np.exp(-((xx-30.)**2+(yy-25.)**2)/3.4)).astype(np.uint16)[:,:,None],3,axis=2)
        second=cv2.warpAffine(first,np.array([[1,0,5.],[0,1,-3.]]),(80,64))
        rows=[dict(frame=0,time=0,x=30,y=25,points=[[30,25],[50,25]],camera_transform=np.eye(3).tolist()),
              dict(frame=1,time=1/30,x=35,y=22,points=[[35,22],[55,22]],camera_transform=[[1,0,-5],[0,1,3],[0,0,1]])]
        track=dict(width=80,height=64,start=0,fps=30,duration=2/30,rows=rows,multiple_points=True,transform_model='tripod_image_pose_v6',reference_points=[[30,25],[50,25]],center_selection=False)
        with patch('backend.processing.source_frames_16bit',side_effect=lambda *_:(f for f in [first,second])):
            result=list(rendered_frames('unused',track))
        np.testing.assert_array_equal(result[0][10:45,10:65],result[1][10:45,10:65])
        track.update(crop_box=[20,15,30,30],output_width=60,output_height=60)
        with patch('backend.processing.source_frames_16bit',side_effect=lambda *_:(f for f in [first,second])):
            cropped=list(rendered_frames('unused',track))
        self.assertEqual(cropped[0].shape,(60,60,3))
        np.testing.assert_array_equal(cropped[0],cropped[1])

    def test_rendered_primary_stays_locked_when_other_references_disagree(self):
        yy,xx=np.mgrid[:128,:192]
        def star(x,y):
            gray=np.rint(1000+48000*np.exp(-((xx-x)**2+(yy-y)**2)/4)).astype(np.uint16)
            return np.repeat(gray[:,:,None],3,axis=2)
        first=star(60.25,50.35);anchor=core_center(first,(60,50))
        reference=np.array([anchor,[150.,30.],[140.,110.]])
        for dx,dy in ((.4,-.7),(12.3,8.2),(-7.2,3.6)):
            frame=star(60.25+dx,50.35+dy)
            points=reference+[dx,dy];points[0]=core_center(frame,points[0]);points[1]+=[0,4];points[2]+=[0,-3]
            matrix,_=registration_matrix(points,reference,192,128)
            matrix,error=refine_anchor(frame,matrix,reference)
            output=cv2.warpAffine(frame,matrix[:2],(192,128),flags=cv2.INTER_CUBIC)
            self.assertLess(np.linalg.norm(core_center(output,anchor)-anchor),.05)
            np.testing.assert_allclose(matrix[:2,:2].T@matrix[:2,:2],np.eye(2),atol=1e-12)

    def test_secondary_outlier_cannot_inject_roll_or_smooth_primary_motion(self):
        reference=np.array([[80.,60.],[900.,65.],[500.,400.]])
        times=[0.,.03,.09,.14,.2];rows=[]
        for i,time in enumerate(times):
            angle=.2*time;c=np.cos(angle);s=np.sin(angle)
            motion=np.array([[c,-s],[s,c]])
            points=reference@motion.T+[(-1)**i*14,i*7.]
            if i==2:points[1]+=[0,12.]
            rows.append(dict(time=time,points=points.tolist()))
        matrices,supported=rigid_trajectory(rows,reference,1920,1080)
        self.assertEqual(supported.tolist(),[True,True,False,True,True])
        for time,row,matrix in zip(times,rows,matrices):
            self.assertAlmostEqual(np.arctan2(matrix[1,0],matrix[0,0]),-.2*time,places=10)
            np.testing.assert_allclose(matrix[:2,:2]@row['points'][0]+matrix[:2,2],reference[0],atol=1e-10)
            np.testing.assert_allclose(matrix[:2,:2].T@matrix[:2,:2],np.eye(2),atol=1e-12)

    def test_coherent_abrupt_camera_roll_is_preserved(self):
        reference=np.array([[80.,60.],[900.,65.],[500.,400.]])
        angles=[0.,.02,-.03,.01];rows=[]
        for i,angle in enumerate(angles):
            c=np.cos(angle);s=np.sin(angle);rotation=np.array([[c,-s],[s,c]])
            rows.append(dict(time=i/30,points=(reference@rotation.T+[i*3.,0.]).tolist()))
        matrices,supported=rigid_trajectory(rows,reference,1920,1080)
        self.assertTrue(supported.all())
        np.testing.assert_allclose([np.arctan2(m[1,0],m[0,0]) for m in matrices],-np.array(angles),atol=1e-12)

    def test_missing_reference_stops_instead_of_holding_or_interpolating(self):
        with self.assertRaises(ValueError):core_center(np.zeros((64,80,3),np.uint16),(30,25))
