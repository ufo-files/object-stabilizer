from pathlib import Path
import tempfile
import time
import unittest
import cv2
import numpy as np
from fastapi.testclient import TestClient
from backend.server import app

class ServiceTests(unittest.TestCase):
    def test_local_upload_track_and_export(self):
        with tempfile.TemporaryDirectory() as folder,TestClient(app) as client:
            path=Path(folder)/'input.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),15,(240,160))
            yy,xx=np.mgrid[:160,:240]
            for i in range(12):
                image=np.clip(35+180*np.exp(-((xx-80-i)**2+(yy-70)**2)/4),0,255).astype(np.uint8)
                writer.write(cv2.cvtColor(image,cv2.COLOR_GRAY2BGR))
            writer.release()
            self.assertEqual(client.get('/api/health').json()['engine'],'opencv')
            self.assertIn('Object Stabilizer',client.get('/').text)
            response=client.post('/api/videos',files={'file':('input.avi',path.read_bytes(),'video/x-msvideo')})
            self.assertEqual(response.status_code,200,response.text);video_id=response.json()['id']
            job=client.post('/api/track',json=dict(video_id=video_id,point=[80,70],mode='bright')).json()['id']
            result=self.wait(client,job)
            self.assertEqual(len(result['rows']),12);self.assertEqual(result['rejected'],0)
            export=client.post(f'/api/jobs/{job}/export',json={'zoom':4}).json()['id']
            output=self.wait(client,export);self.assertEqual(output['frames'],12)
            download=client.get(output['download']);self.assertEqual(download.status_code,200)
            exported=Path(folder)/'output.mp4';exported.write_bytes(download.content)
            cap=cv2.VideoCapture(str(exported));self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),12);cap.release()
    def wait(self,client,job):
        for _ in range(300):
            result=client.get(f'/api/jobs/{job}').json()
            if result['status']=='complete':return result['result']
            if result['status']=='failed':self.fail(result['error'])
            time.sleep(.03)
        self.fail('Job did not complete')
    def test_invalid_requests_rejected(self):
        with TestClient(app) as client:
            self.assertEqual(client.post('/api/track',json=dict(video_id='missing',point=[0,0])).status_code,404)
            self.assertEqual(client.post('/api/track',json=dict(video_id='missing',point=[0,0],radius=-1)).status_code,422)
            self.assertEqual(client.get('/api/jobs/not-a-job/video').status_code,404)
            self.assertEqual(client.post('/api/videos',files={'file':('bad.mp4',b'bad data','video/mp4')}).status_code,400)
if __name__=='__main__':unittest.main()
