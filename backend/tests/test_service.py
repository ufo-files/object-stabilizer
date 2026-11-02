"""API acceptance for the replacement app; old API tests are in the snapshot."""
from pathlib import Path
import tempfile,time,unittest,io,zipfile,json,csv
import cv2,numpy as np
from fastapi.testclient import TestClient
from backend.server import app

class ServiceTests(unittest.TestCase):
    def test_upload_select_stabilize_preview_and_audit(self):
        with tempfile.TemporaryDirectory() as directory,TestClient(app) as client:
            source=Path(directory)/'source.avi';writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'FFV1'),15,(320,240))
            yy,xx=np.mgrid[:240,:320]
            gray=np.rint(8+185*np.exp(-((xx-160)**2+(yy-120)**2)/5)).astype(np.uint8)
            scene=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
            for dx,dy in [(0,0),(3,2),(-5,4),(0,0)]:writer.write(cv2.warpAffine(scene,np.float32([[1,0,dx],[0,1,dy]]),(320,240)))
            writer.release()
            self.assertEqual(client.get('/api/health').json()['engine'],'auto-scene-1')
            self.assertEqual(client.get('/api/health').json()['default_mode'],'scene')
            self.assertIn('Tripod Stabilizer',client.get('/').text)
            response=client.post('/api/videos',files={'file':('source.avi',source.read_bytes(),'video/x-msvideo')});self.assertEqual(response.status_code,200,response.text)
            video=response.json();self.assertEqual(client.get(video['first']).status_code,200)
            original=client.get('/api/videos/'+video['id']+'/original')
            self.assertEqual(original.content,source.read_bytes())
            self.assertIn('source.avi',original.headers['content-disposition'])
            reference=client.post('/api/reference',json=dict(video_id=video['id'],box=[120,80,64,64]));self.assertEqual(reference.status_code,200,reference.text);np.testing.assert_allclose(reference.json()['anchor'],[160,120],atol=.01)
            response=client.post('/api/stabilize',json=dict(video_id=video['id'],box=[120,80,64,64]));self.assertEqual(response.status_code,200,response.text)
            for _ in range(300):
                job=client.get('/api/jobs/'+response.json()['id']).json()
                if job['status'] not in ('queued','running'):break
                time.sleep(.02)
            self.assertEqual(job['status'],'complete',job)
            result=job['result'];self.assertEqual(result['frames'],4);self.assertEqual(result['output_frames'],4)
            download=client.get(result['url']);self.assertEqual(download.status_code,200)
            self.assertEqual(download.headers['cache-control'],'no-store')
            partial=client.get(result['url'],headers={'Range':'bytes=0-99'});self.assertEqual(partial.status_code,206);self.assertEqual(len(partial.content),100)
            report=client.get(result['report']).json();self.assertEqual(len(report['audit']),4)
            self.assertIn('center_step_pixels',report['audit'][1]);self.assertIn('center_drift_pixels',report['audit'][1])
            self.assertTrue(report['center_lock_passed']);self.assertLess(report['max_residual_pixels'],.05)
            self.assertEqual(result['extension'],'mp4')
            self.assertIn('profile_center_2_step_pixels',report['profile_metrics'])
            self.assertFalse(report['pixels_unchanged'])
            self.assertEqual(client.get(result['measurements']).status_code,200)
            cropped=client.post('/api/stabilize',json=dict(video_id=video['id'],mode='light',box=[137,97,53,47],crop_to_selection=True))
            for _ in range(300):
                crop_job=client.get('/api/jobs/'+cropped.json()['id']).json()
                if crop_job['status'] not in ('queued','running'):break
                time.sleep(.02)
            self.assertEqual(crop_job['status'],'complete',crop_job)
            crop=crop_job['result'];self.assertTrue(crop['crop_to_selection'])
            self.assertEqual((crop['output_width'],crop['output_height']),(54,48))
            self.assertEqual(client.get(crop['url']).status_code,200)
            for export,dimensions in ((result,(320,240)),(crop,(54,48))):
                response=client.post(export['frames_request'])
                self.assertEqual(response.status_code,200,response.text)
                for _ in range(300):
                    frames_job=client.get('/api/jobs/'+response.json()['id']).json()
                    if frames_job['status'] not in ('queued','running'):break
                    time.sleep(.02)
                self.assertEqual(frames_job['status'],'complete',frames_job)
                archive=client.get(frames_job['result']['url'])
                self.assertEqual(archive.status_code,200)
                with zipfile.ZipFile(io.BytesIO(archive.content)) as zipped:
                    names=sorted(name for name in zipped.namelist() if name.endswith('.png'))
                    self.assertEqual(names,[f'frame-{i:08d}.png' for i in range(1,5)])
                    metadata=json.loads(zipped.read('metadata.json'))
                    self.assertEqual((metadata['width'],metadata['height']),dimensions)
                    self.assertEqual(metadata['fps'],'15/1')
                    rows=list(csv.DictReader(io.StringIO(zipped.read('frames.csv').decode())))
                    self.assertEqual([int(r['source_frame']) for r in rows],[1,2,3,4])
                    self.assertAlmostEqual(float(rows[-1]['timestamp_seconds']),.2,places=5)
                    for name in names:
                        image=cv2.imdecode(np.frombuffer(zipped.read(name),np.uint8),cv2.IMREAD_COLOR)
                        self.assertEqual((image.shape[1],image.shape[0]),dimensions)


    def test_automatic_scene_requires_no_box_or_selected_point(self):
        with tempfile.TemporaryDirectory() as directory,TestClient(app) as client:
            source=Path(directory)/'scene.avi'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'FFV1'),24,(640,360))
            yy,xx=np.mgrid[:360,:640];image=np.full((360,640),8.,float)
            for x,y in ((100,80),(300,100),(520,220),(180,270)):
                image+=180*np.exp(-((xx-x)**2+(yy-y)**2)/8)
            scene=cv2.cvtColor(np.rint(image).astype(np.uint8),cv2.COLOR_GRAY2BGR)
            for dx,dy in ((0,0),(3,2),(-5,4),(0,0)):
                writer.write(cv2.warpAffine(scene,np.float32([[1,0,dx],[0,1,dy]]),(640,360)))
            writer.release()
            uploaded=client.post('/api/videos',files={'file':('scene.avi',source.read_bytes(),'video/x-msvideo')})
            self.assertEqual(uploaded.status_code,200,uploaded.text)
            video=uploaded.json();self.assertEqual(video['analysis']['method'],'sparse_lights')
            self.assertEqual(video['analysis']['reference_count'],4)
            response=client.post('/api/stabilize',json={'video_id':video['id']})
            self.assertEqual(response.status_code,200,response.text)
            for _ in range(400):
                job=client.get('/api/jobs/'+response.json()['id']).json()
                if job['status'] not in ('queued','running'):break
                time.sleep(.02)
            self.assertEqual(job['status'],'complete',job)
            self.assertEqual(job['result']['mode'],'scene')
            report=client.get(job['result']['report']).json()
            self.assertEqual(report['motion_model'],'translation')
            self.assertTrue(report['rigid_geometry_passed'])
            self.assertEqual(report['frames'],4)
            self.assertEqual(report['unmeasurable_frames'],0)
            self.assertLess(report['max_residual_pixels'],.1)

    def test_invalid_inputs(self):
        with TestClient(app) as client:
            self.assertEqual(client.post('/api/stabilize',json={'video_id':'missing','box':[0,0,64,64]}).status_code,404)
            self.assertEqual(client.post('/api/stabilize',json={'video_id':'missing','box':[0,0]}).status_code,422)
            self.assertEqual(client.get('/api/jobs/missing').status_code,404)
            self.assertEqual(client.get('/api/results/missing/missing/mp4').status_code,404)
            self.assertEqual(client.post('/api/videos',files={'file':('bad.mp4',b'bad data','video/mp4')}).status_code,400)
            self.assertEqual(client.post('/api/stabilize',json={'video_id':'missing','mode':'light'}).status_code,422)
            self.assertEqual(client.post('/api/stabilize',json={'video_id':'missing','mode':'affine'}).status_code,422)
            self.assertEqual(client.post('/api/stabilize',json={'video_id':'missing','mode':'scene','crop_to_selection':True}).status_code,422)

if __name__=='__main__':unittest.main()
