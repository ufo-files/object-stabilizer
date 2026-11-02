from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from backend.light_segment import analyse, stabilize, frame_at, selected, candidates
from backend.point_lock import yuv_frames


def moving_video(path, frames=121, first=9, last=99, faint=8, reappear=False):
    width,height=480,240
    yy,xx=np.mgrid[:height,:width]
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'FFV1'),24,(width,height))
    for frame in range(frames):
        image=np.full((height,width),8.,float)
        for x,y,b in ((400,180,150),(350,45,220)):
            image+=b*np.exp(-((xx-x)**2+(yy-y)**2)/5)
        if first <= frame <= last or (reappear and frame >= 110):
            x=90+2.1*(frame-first)
            y=110+8*np.sin((frame-first)/30)
            brightness=faint+180*max(0,np.sin(np.pi*(frame-first)/max(1,last-first)))
            image+=brightness*np.exp(-((xx-x)**2+(yy-y)**2)/5)
        image=np.rint(image).clip(0,255).astype(np.uint8)
        image[0:4,0:4]=frame
        writer.write(cv2.cvtColor(image,cv2.COLOR_GRAY2BGR))
    writer.release()
    seed=(first+last)//2
    x=90+2.1*(seed-first);y=110+8*np.sin((seed-first)/30)
    return seed,(round(x)-12,round(y)-12,25,25)


class SegmentTests(unittest.TestCase):
    def test_separate_noise_islands_beside_one_core_are_one_light(self):
        image=np.full((80,80),8,np.uint8)
        image[40,40]=208
        image[46,46]=28  # A disconnected island whose patch includes the core.
        target=selected(image,(34,34,12,12))
        np.testing.assert_allclose(target['point'],(40,40),atol=.01)
        self.assertEqual(len(candidates(image)),1)

    def test_faint_noise_does_not_block_selection_but_two_real_lights_do(self):
        image=np.full((80,80),8,np.uint8)
        image[40,40]=208;image[40,60]=28
        target=selected(image,(32,32,40,20))
        np.testing.assert_allclose(target['point'],(40,40),atol=.01)
        image[40,60]=188
        with self.assertRaisesRegex(ValueError,'Two distinct lights'):
            selected(image,(32,32,40,20))

    def test_bright_middle_frame_finds_faint_initial_appearance_and_disappearance(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'moving.avi'
            seed,box=moving_video(source,reappear=True)
            track=analyse(source,box,seed)
            self.assertEqual(track['first_visible_frame'],9)
            self.assertEqual(track['last_visible_frame'],99)
            self.assertEqual((track['start_frame'],track['end_frame']),(8,100))
            self.assertEqual(len(track['positions']),91)
            self.assertGreater(track['positions']['99'][0]-track['positions']['9'][0],180)
            # Seeding the same light near either end must find the same interval.
            for alternate in (10,98):
                x=90+2.1*(alternate-9);y=110+8*np.sin((alternate-9)/30)
                result=analyse(source,(round(x)-12,round(y)-12,25,25),alternate)
                self.assertEqual((result['start_frame'],result['end_frame']),(8,100))

    def test_actual_encoded_segment_has_exact_padding_crop_and_frame_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'moving.avi';output=Path(directory)/'output.mp4'
            seed,box=moving_video(source,faint=20)
            report=stabilize(source,output,box,seed,crop_to_selection=True)
            self.assertEqual((report['start_frame'],report['end_frame']),(8,100))
            self.assertEqual(report['frames'],93);self.assertEqual(report['output_frames'],93)
            self.assertEqual((report['output_width'],report['output_height']),(26,26))
            self.assertEqual(sum(row['visible'] for row in report['audit']),91)
            self.assertIsNone(report['audit'][0]['center_x']);self.assertIsNone(report['audit'][-1]['center_x'])
            self.assertEqual(report['rows'][0]['dx'],report['rows'][1]['dx'])
            self.assertEqual(report['rows'][-1]['dx'],report['rows'][-2]['dx'])
            self.assertTrue(report['rigid_translation_samples_passed'])
            self.assertLess(report['max_residual_pixels'],.15)
            self.assertEqual([row['source_frame'] for row in report['audit']],list(range(8,101)))
            # Decode every frame again, independently of the tracking/audit pass.
            frames=list(yuv_frames(output,26,26));self.assertEqual(len(frames),93)
            self.assertLess(int(frames[0][0][:25,:25].max()),30)
            self.assertLess(int(frames[-1][0][:25,:25].max()),30)
            for frame in frames[1:-1]:self.assertGreater(int(frame[0].max()),25)

    def test_frame_selection_is_exact_and_blank_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'moving.avi';seed,box=moving_video(source)
            image=cv2.imdecode(np.frombuffer(frame_at(source,seed),np.uint8),cv2.IMREAD_GRAYSCALE)
            self.assertEqual(int(image[1,1]),seed)
            with self.assertRaises(ValueError):analyse(source,box,0)
            with self.assertRaises(ValueError):frame_at(source,121)

    def test_video_boundaries_are_clamped_without_inventing_padding_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'edge.avi';seed,box=moving_video(source,frames=11,first=0,last=10,faint=30)
            result=analyse(source,box,seed)
            self.assertEqual((result['start_frame'],result['end_frame']),(0,10))
            self.assertEqual(result['boundary_reasons'],{'before':'video-boundary','after':'video-boundary'})

if __name__=='__main__':unittest.main()
