import unittest
from unittest.mock import patch
import cv2
import numpy as np
from backend.processing import refine_visible_centers,rendered_frames


class ReferenceCoreTests(unittest.TestCase):
    def test_varying_offset_halo_does_not_pull_the_visible_reference_core(self):
        yy,xx=np.mgrid[:64,:80];frames=[];rows=[]
        offsets=[0,5,-5,4,-4,3]
        for i,tail in enumerate(offsets):
            x,y=37.2+i*.7,30.1-i*.4
            image=1000+45000*np.exp(-((xx-x)**2+(yy-y)**2)/2.8)
            if tail:image+=30000*np.exp(-((xx-x)**2+(yy-y-tail)**2)/2.8)
            frames.append(np.repeat(np.rint(image).astype(np.uint16)[:,:,None],3,axis=2))
            rows.append(dict(frame=i,time=i/30,x=x,y=y+tail*.4,accepted=True))
        track=dict(width=80,height=64,start=0,fps=30,duration=len(frames)/30,
                   center_selection=False,reference_size=24,rows=rows)
        with patch('backend.processing.source_frames_16bit',side_effect=lambda *_:(frame for frame in frames)),patch('backend.processing.source_color',return_value={}):
            refine_visible_centers('unused',track)
            output=list(rendered_frames('unused',track))
        self.assertEqual(track['centering'],'bright reference core')
        centers=[]
        for frame in output:
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(float)
            weights=np.maximum(gray-1000-(gray.max()-1000)*.8,0)
            centers.append([(xx*weights).sum()/weights.sum(),(yy*weights).sum()/weights.sum()])
        errors=np.linalg.norm(np.array(centers)-centers[0],axis=1)
        self.assertLess(max(errors),.2,'measure the visible core instead of averaging the changing halo')
