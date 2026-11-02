import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np

from backend.processing import export_video,track_video,transform,refine_visible_centers


class VisibleCenteringTests(unittest.TestCase):
    def test_faint_high_zoom_frames_keep_subpixel_rendered_centers(self):
        yy,xx=np.mgrid[:64,:80]
        frames=[];rows=[]
        for index in range(6):
            x,y=37.25+index*.13,30.3-index*.17
            gray=40+(1100-index*70)*np.exp(-((xx-x)**2+(yy-y)**2)/3.4)
            frames.append(np.repeat(np.rint(gray).astype(np.uint16)[:,:,None],3,axis=2))
            rows.append(dict(x=x+.2,y=y-.15,frame=index,time=index/25,accepted=True))
        track=dict(rows=rows,crop_width=18.4,crop_height=17.8,output_width=1124,output_height=1080)
        with patch('backend.processing.source_frames_16bit',return_value=(frame for frame in frames)),patch('backend.processing.source_color',return_value={}):
            refine_visible_centers(Path('synthetic.video'),track)
        self.assertEqual(track['centering_version'],2)
        for frame,row in zip(frames,rows):
            image=transform(frame,(row['x'],row['y']),min(1124/18.4,1080/17.8),(1124,1080))
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(float)
            background=np.median(np.concatenate([gray[0],gray[-1],gray[:,0],gray[:,-1]]))
            weights=np.maximum(gray-background-(gray.max()-background)*.2,0)
            ys,xs=np.indices(gray.shape)
            error=np.hypot((xs*weights).sum()/weights.sum()-562,(ys*weights).sum()/weights.sum()-540)
            self.assertLess(error,.5,'measure the actual HD render, not only detector coordinates')

    def test_high_zoom_sampling_moves_continuously_at_subpixel_offsets(self):
        yy,xx=np.mgrid[:40,:48];frame=(1000+xx*400+yy*100).astype(np.float32)
        samples=[float(transform(frame,(24+offset,20),64,(33,31))[15,16]) for offset in np.arange(0,.02,.001)]
        differences=np.diff(samples)
        self.assertTrue(np.all(differences>0),'tiny source movements must not stick then jump at 1/32 pixel')
        self.assertLess(float(differences.max()),1)

    def test_encoded_enlarged_crop_stays_centered_as_halo_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);source=folder/'source.avi'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),15,(80,64))
            yy,xx=np.mgrid[:64,:80]
            for i in range(8):
                x,y=37+i*.6,30+i*.2
                image=15+170*np.exp(-((xx-x)**2+(yy-y)**2)/3.4)
                image+=(i%3)*25*np.exp(-((xx-x-2.5)**2+(yy-y+1.5)**2)/18)
                writer.write(cv2.cvtColor(np.clip(image,0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR))
            writer.release()
            track=track_video(source,dict(point=[37,30],mode='bright',size=20,radius=30,
                crop_width=16,crop_height=20,output_width=480,output_height=600))
            self.assertEqual(track['centering'],'rendered bright body')
            output=folder/'output.mp4';export_video(source,track,output,lossless=False)
            cap=cv2.VideoCapture(str(output));errors=[]
            while True:
                ok,frame=cap.read()
                if not ok:break
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(float)
                border=np.concatenate([gray[0],gray[-1],gray[:,0],gray[:,-1]])
                background=np.median(border)
                weights=np.maximum(gray-background-(gray.max()-background)*.2,0);ys,xs=np.indices(gray.shape)
                errors.append(np.hypot((xs*weights).sum()/weights.sum()-240,(ys*weights).sum()/weights.sum()-300))
            cap.release()
            self.assertEqual(len(errors),8)
            self.assertLess(max(errors),2.5,'measure actual encoded frames, not only tracking coordinates')
