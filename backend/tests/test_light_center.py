"""Known camera motion, source geometry, and the actual encoded result."""
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from backend.light_center import center, refine, stabilize, translation_sample_error
from backend.point_lock import translate, yuv_frames


class LightCenterTests(unittest.TestCase):
    def test_subpixel_motion_is_not_rounded_to_a_peak_pixel(self):
        yy, xx = np.mgrid[:240, :320]
        scene = np.rint(8 + 180 * np.exp(-((xx - 160.2)**2 / 8 + (yy - 120.3)**2 / 16))).astype(np.uint8)
        window = np.array([160, 120])
        target = center(scene, window)
        for dx, dy in ((.35, -.6), (5.4, -3.2), (-7.25, 4.75)):
            # A distinct renderer creates the source motion, avoiding an inverse
            # test that merely reuses the production interpolation coefficients.
            moved = cv2.warpAffine(scene, np.float32([[1, 0, dx], [0, 1, dy]]), (320, 240), flags=cv2.INTER_LANCZOS4)
            position = np.array([160 + round(dx), 120 + round(dy)])
            displacement, error = refine(moved, window, target, position)
            self.assertLess(error, .02)
            corrected = translate(moved, displacement)
            self.assertLess(np.linalg.norm(center(corrected, window) - target), .02)
            self.assertLess(np.linalg.norm(displacement - [dx, dy]), .15)

    def test_encoded_video_has_no_accumulating_motion_or_local_warp(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.avi'
            destination = Path(directory) / 'output.mp4'
            yy, xx = np.mgrid[:240, :320]
            motions = [(0, 0), (.4, -.65), (5.4, -3.2), (-7.25, 4.75), (0, 0)]
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'FFV1'), 24, (320, 240))
            for dx, dy in motions:
                # Generate analytically sampled, translated lights. The second
                # light is outside the selection and never enters tracking.
                image = 8 + 180 * np.exp(-((xx - 160.2 - dx)**2 / 8 + (yy - 120.3 - dy)**2 / 16))
                image += 120 * np.exp(-((xx - 240.2 - dx)**2 / 8 + (yy - 160.3 - dy)**2 / 16))
                writer.write(cv2.cvtColor(np.rint(image).astype(np.uint8), cv2.COLOR_GRAY2BGR))
            writer.release()
            report = stabilize(source, destination, (140, 100, 40, 40))
            self.assertEqual(report['frames'], len(motions))
            self.assertEqual(report['output_frames'], len(motions))
            self.assertLess(report['max_residual_pixels'], .05)
            self.assertTrue(report['center_lock_passed'])
            self.assertTrue(report['rigid_translation_samples_passed'])
            decoded = list(yuv_frames(destination, 320, 240))
            reference_separation = center(decoded[0][0], (240, 160)) - center(decoded[0][0], (160, 120))
            for frame in decoded:
                separation = center(frame[0], (240, 160)) - center(frame[0], (160, 120))
                self.assertLess(np.linalg.norm(separation - reference_separation), .12)
            self.assertFalse(report['pixels_unchanged'])
            self.assertFalse(report['zero_motion_passed'])
            self.assertIn('profile_center_1_drift_pixels', report['profile_metrics'])
            self.assertTrue(destination.with_suffix('.csv').is_file())

    def test_blank_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            center(np.zeros((128, 128), np.uint8), (64, 64))

    def test_crop_uses_drawn_box_and_audits_actual_encoded_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.avi'
            output = Path(directory) / 'crop.mp4'
            yy, xx = np.mgrid[:240, :320]
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'FFV1'), 24, (320, 240))
            for dx, dy in ((0, 0), (5.4, -3.2), (-7.25, 4.75)):
                image = np.rint(8 + 180 * np.exp(-((xx - 160.2 - dx)**2 / 8 + (yy - 120.3 - dy)**2 / 16))).astype(np.uint8)
                writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
            writer.release()
            # Odd origin and dimensions, with the light intentionally off-center.
            box = (137, 97, 53, 47)
            report = stabilize(source, output, box, crop_to_selection=True)
            self.assertEqual(report['crop_box'], list(box))
            self.assertEqual((report['output_width'], report['output_height']), (54, 48))
            np.testing.assert_allclose(report['anchor'], np.array(report['source_anchor']) - [137, 97])
            self.assertLess(abs(report['anchor'][0] - 23.2), .1)
            self.assertTrue(report['rigid_translation_samples_passed'])
            self.assertEqual(report['output_frames'], 3)
            decoded = list(yuv_frames(output, 54, 48))
            source_frames = yuv_frames(source, 320, 240)
            original = next(source_frames)[0]; source_frames.close()
            difference = abs(decoded[0][0][:47, :53].astype(float) - original[97:144, 137:190])
            self.assertLessEqual(difference.max(), 2)
            self.assertLessEqual(decoded[0][0][-1].max(), 2)
            self.assertLessEqual(decoded[0][0][:, -1].max(), 2)
            self.assertLess(report['max_residual_pixels'], .05)

    def test_independent_samples_detect_an_unrequested_local_warp(self):
        yy, xx = np.mgrid[:240, :320]
        scene = ((xx + yy * 2) % 256).astype(np.uint8)
        displacement = (3.37, -2.56)
        output = translate(scene, displacement)
        error, _, _ = translation_sample_error(scene, output, displacement, (160, 120))
        self.assertLessEqual(error, 1)
        output[110:130, 150:170] = 0
        error, _, _ = translation_sample_error(scene, output, displacement, (160, 120))
        self.assertGreater(error, 40)


if __name__ == '__main__':
    unittest.main()
