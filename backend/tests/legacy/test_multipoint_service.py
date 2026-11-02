from pathlib import Path
import tempfile
import time
import unittest
import cv2
import numpy as np
from fastapi.testclient import TestClient
from backend.server import app


class MultiplePointServiceTests(unittest.TestCase):
    def wait(self,client,job):
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            record=client.get('/api/jobs/'+job).json()
            if record['status'] in ('failed','cancelled'):self.fail(record.get('error') or record['status'])
            if record['status']=='complete':return record['result']
            time.sleep(.02)
        self.fail('Job timed out')

    def test_multiple_reference_upload_track_and_mp4_export(self):
        with tempfile.TemporaryDirectory() as directory,TestClient(app) as client:
            folder=Path(directory);source=folder/'references.avi'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),15,(320,240))
            yy,xx=np.mgrid[:240,:320];points=[[50,40],[270,45],[90,190]]
            for i in range(5):
                image=np.full(xx.shape,20.)
                for x,y in points:image+=180*np.exp(-((xx-x-i*1.2)**2+(yy-y+i*.5)**2)/3.4)
                writer.write(cv2.cvtColor(np.clip(image,0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR))
            writer.release()
            response=client.post('/api/videos',files={'file':('references.avi',source.read_bytes(),'video/x-msvideo')})
            self.assertEqual(response.status_code,200,response.text);video=response.json()['id']
            bad=client.post('/api/track',json=dict(video_id=video,point=points[0],points=points[:1]))
            self.assertEqual(bad.status_code,422)
            response=client.post('/api/track',json=dict(video_id=video,point=points[0],points=points,mode='bright'))
            self.assertEqual(response.status_code,200,response.text);job=response.json()['id']
            result=self.wait(client,job)
            self.assertTrue(result['multiple_points']);self.assertEqual(result['rejected'],0)
            self.assertEqual(result['transform_model'],'tripod_image_pose_v6')
            self.assertIn('rotation_uncertain_frames',result)
            self.assertFalse(result['correct_rotation'])
            for row in result['rows']:np.testing.assert_array_equal(np.array(row['camera_transform'])[:2,:2],np.eye(2))
            self.assertEqual(len(result['reference_points']),3);self.assertEqual(len(result['rows']),5)
            self.assertTrue(all('camera_transform' in row for row in result['rows']))
            export=client.post('/api/jobs/'+job+'/export',json={'lossless':False}).json()['id']
            rendered=self.wait(client,export)
            self.assertEqual(rendered['stability_qa']['decoded_frames'],5)
            self.assertIn(rendered['stability_qa']['status'],('passed','failed'))
            response=client.get(rendered['download'])
            self.assertEqual(response.status_code,200);output=folder/'output.mp4';output.write_bytes(response.content)
            cap=cv2.VideoCapture(str(output));self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),320)
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),240)
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),5);cap.release()
