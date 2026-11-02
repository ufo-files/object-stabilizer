import json
import tempfile
import unittest
import zipfile
import subprocess
from pathlib import Path

import cv2
import numpy as np

from backend.processing import Cancelled,export_frames,export_video,messaging_frame


class FrameExportTests(unittest.TestCase):
    def test_hdr_to_sdr_keeps_black_and_white_and_monotonic_grays(self):
        levels=np.linspace(0,65535,256).astype(np.uint16)
        frame=np.broadcast_to(levels[None,:,None],(2,256,3)).copy()
        for transfer in ('arib-std-b67','smpte2084'):
            output=messaging_frame(frame,dict(color_transfer=transfer,color_primaries='bt2020'))
            self.assertEqual(output.dtype,np.uint16)
            self.assertEqual(output[0,0].max(),0)
            self.assertGreater(output[0,-1].min(),65000)
            self.assertTrue(np.all(np.diff(output[0,:,1].astype(int))>=0))

    def test_lossless_exports_preserve_original_16bit_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            original=folder/'source.video';proxy=folder/'prepared.mp4'
            proxy.write_bytes(b'This playback proxy must never be used for export.')
            values=(np.arange(24*32*3,dtype=np.uint16).reshape(24,32,3)*11+1000)
            subprocess.run(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr48le',
                '-s','32x24','-r','24','-i','pipe:0','-c:v','rawvideo','-pix_fmt','rgb48be',
                '-vf','setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=gbr',
                '-movflags','+write_colr','-f','mov',str(original)],input=values.tobytes()*2,check=True)
            track=dict(width=32,height=24,start=0,fps=24,rate='24/1',duration=2/24,
                center_selection=False,rows=[dict(frame=i,time=i/24,x=3,y=5) for i in range(2)])
            movie=folder/'export.mkv';export_video(proxy,track,movie)
            info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                '-show_entries','stream=codec_name,pix_fmt,color_primaries,color_transfer','-of','json',str(movie)]))['streams'][0]
            self.assertEqual(info['codec_name'],'ffv1');self.assertEqual(info['pix_fmt'],'gbrp16le')
            self.assertEqual(info['color_primaries'],'bt2020');self.assertEqual(info['color_transfer'],'arib-std-b67')
            decoded=subprocess.check_output(['ffmpeg','-v','error','-i',str(movie),'-f','rawvideo','-pix_fmt','bgr48le','pipe:1'])
            self.assertEqual(decoded,values.tobytes()*2)
            share=folder/'share.mp4';export_video(proxy,track,share,lossless=False)
            stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                '-show_entries','stream=codec_name,pix_fmt,color_primaries,color_transfer,color_space,color_range',
                '-of','json',str(share)]))['streams'][0]
            self.assertEqual(stream['codec_name'],'h264');self.assertEqual(stream['pix_fmt'],'yuv420p')
            self.assertEqual(stream['color_primaries'],'bt709');self.assertEqual(stream['color_transfer'],'bt709')
            self.assertEqual(stream['color_space'],'bt709');self.assertEqual(stream['color_range'],'tv')
            data=share.read_bytes();self.assertLess(data.index(b'moov'),data.index(b'mdat'))
            archive=folder/'frames.zip';export_frames(proxy,track,archive)
            with zipfile.ZipFile(archive) as z:
                image=cv2.imdecode(np.frombuffer(z.read('frames/frame_000001.png'),np.uint8),cv2.IMREAD_UNCHANGED)
                np.testing.assert_array_equal(image,values)
                self.assertEqual(z.getinfo('frames/frame_000001.png').compress_type,zipfile.ZIP_STORED)
                self.assertEqual(json.loads(z.read('timing.json'))['compression'],'lossless PNG')

    def test_fractional_native_rate_and_timestamp_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);source=folder/'source.avi'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),24,(32,24))
            for value in (30,100,180):writer.write(np.full((24,32,3),value,np.uint8))
            writer.release()
            fps=24000/1001
            track=dict(width=32,height=24,start=0,fps=fps,rate='24000/1001',duration=4/fps,
                rows=[dict(frame=i,time=time/fps,x=16,y=12) for i,time in enumerate((0,2,3))])
            destination=folder/'frames.zip'
            result=export_frames(source,track,destination)
            self.assertEqual(result['frames'],4)
            with zipfile.ZipFile(destination) as archive:
                timing=json.loads(archive.read('timing.json'))
                self.assertEqual(timing['rate'],'24000/1001');self.assertEqual(timing['fps'],fps)
                values=[]
                for i in range(1,5):
                    frame=cv2.imdecode(np.frombuffer(archive.read(f'frames/frame_{i:06d}.png'),np.uint8),cv2.IMREAD_UNCHANGED)
                    self.assertEqual(frame.dtype,np.uint16)
                    values.append(round(float(frame.mean())/257))
                self.assertTrue(np.allclose(values,[30,30,100,180],atol=1),values)
            cancelled=folder/'cancelled.zip'
            with self.assertRaises(Cancelled):export_frames(source,track,cancelled,cancel=lambda:True)
            self.assertFalse(cancelled.exists());self.assertFalse(cancelled.with_suffix('.partial.zip').exists())
