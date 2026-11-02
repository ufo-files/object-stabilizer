"""Acceptance tests for native point locking, geometry, and the encoded file."""
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from backend.point_lock import peak, source_overlap_equal, stabilize, translate, yuv_frames


class PointLockTests(unittest.TestCase):
    def test_translation_sign_and_exact_integer_pixels(self):
        source = np.arange(96 * 128, dtype=np.uint8).reshape(96, 128)
        for displacement in [(0, 0), (5, -3), (-6, 4), (-4, -5)]:
            output = translate(source, displacement)
            self.assertTrue(source_overlap_equal(source, output, displacement))
            dx, dy = displacement
            self.assertEqual(output[40 - dy, 60 - dx], source[40, 60])
        broken = translate(source, (5, -3)); broken[30, 30] ^= 1
        self.assertFalse(source_overlap_equal(source, broken, (5, -3)))

    def test_fractional_chroma_uses_half_the_luma_displacement(self):
        yy, xx = np.mgrid[:48, :64]
        source = (xx * 2 + yy * 2).astype(np.uint8)
        shifted = translate(source, (1.5, -.5))
        np.testing.assert_array_equal(shifted[5:-5, 5:-5], source[5:-5, 5:-5] + 2)

    def test_blank_point_is_not_reported_as_locked(self):
        with self.assertRaises(ValueError):
            peak(np.zeros((96, 128), dtype=np.uint8), (20, 20, 32, 32))

    def test_lossless_export_locks_peak_without_hiding_appearance_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.avi'
            destination = Path(directory) / 'result.webm'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'FFV1'), 24, (320, 240))
            yy, xx = np.mgrid[:240, :320]
            motion = [(0, 0), (7, -4), (-6, 5), (4, 3), (0, 0)]
            for index, (dx, dy) in enumerate(motion):
                # The peak is known independently; the surrounding light varies.
                image = 8 + 185 * np.exp(-((xx - 160 - dx)**2 + (yy - 120 - dy)**2) / 5)
                image += (index % 3) * 55 * np.exp(-((xx - 160 - dx)**2 + (yy - 129 - dy)**2) / 12)
                image += 125 * np.exp(-((xx - 240 - dx)**2 + (yy - 160 - dy)**2) / 5)
                image = np.rint(np.clip(image, 0, 255)).astype(np.uint8)
                writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
            writer.release()
            result = stabilize(source, destination, (144, 104, 32, 32))
            self.assertEqual(result['frames'], len(motion))
            self.assertEqual(result['output_frames'], len(motion))
            self.assertEqual(result['anchor'], [160, 120])
            self.assertTrue(result['peak_lock_passed'])
            self.assertTrue(result['source_luma_preserved'])
            self.assertEqual(result['max_peak_offset_pixels'], 0)
            for row, expected in zip(result['rows'], motion):
                self.assertEqual((row['dx'], row['dy']), expected)
            decoded = list(yuv_frames(destination, 320, 240))
            for y, _, _ in decoded:
                self.assertEqual(tuple(peak(y, (128, 88, 65, 65))), (160, 120))
                # A separate fixed light retains its relative geometry.
                self.assertEqual(tuple(peak(y, (224, 144, 33, 33))), (240, 160))
            self.assertGreater(result['max_step_pixels'], 0)
            self.assertFalse(result['zero_motion_passed'], 'A fixed peak must not certify a frozen light distribution.')
            self.assertTrue(destination.with_suffix('.csv').is_file())


if __name__ == '__main__':
    unittest.main()
