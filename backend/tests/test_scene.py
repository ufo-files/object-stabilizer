"""Known camera movement, unrelated foreground motion, and rigid geometry."""
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from backend.scene_motion import inspect_frame, project, robust_pose, SceneTracker
from backend.scene_pipeline import render, stabilize


def texture():
    random = np.random.default_rng(73)
    image = cv2.GaussianBlur(random.integers(0, 256, (360, 640), dtype=np.uint8), (0, 0), .9)
    image = np.clip((image.astype(float) - 128) * 2 + 128, 0, 255).astype(np.uint8)
    for y in (50, 100, 270, 310):
        cv2.line(image, (20, y), (620, y), 25, 2)
    return image


class SceneTests(unittest.TestCase):
    def test_translation_rejects_a_moving_foreground(self):
        yy, xx = np.mgrid[60:301:80, 80:561:120]
        reference = np.column_stack((xx.ravel(), yy.ravel())).astype(float)
        current = reference + [5.3, -3.7]
        current[[6, 7, 8]] += [30, -14]
        pose, inliers, _ = robust_pose(reference, current, 640, 360)
        np.testing.assert_allclose(pose[:2, 2], [5.3, -3.7], atol=1e-9)
        self.assertEqual(inliers.sum(), len(reference) - 3)

    def test_render_rejects_scale_or_shear(self):
        planes = (np.zeros((120, 160), np.uint8), np.full((60, 80), 128, np.uint8), np.full((60, 80), 128, np.uint8))
        for bad in (np.array([[1.01, 0, 0], [0, 1, 0], [0, 0, 1]]),
                    np.array([[1, .01, 0], [0, 1, 0], [0, 0, 1]])):
            with self.assertRaises(ValueError):
                render(planes, bad)

    def test_static_scene_does_not_receive_a_tracking_correction(self):
        first = texture()
        analysis = inspect_frame(first)
        self.assertEqual(analysis['method'], 'scene_features')
        tracker = SceneTracker(first, analysis)
        np.testing.assert_array_equal(tracker.locate(first.copy()), tracker.reference)

    def test_sparse_lights_are_detected_without_a_selected_box(self):
        yy, xx = np.mgrid[:360, :640]
        image = np.full((360, 640), 8., float)
        for x, y in ((100, 80), (300, 100), (520, 220), (180, 270)):
            image += 180 * np.exp(-((xx - x)**2 + (yy - y)**2) / 8)
        analysis = inspect_frame(np.rint(image).astype(np.uint8))
        self.assertEqual(analysis['method'], 'sparse_lights')
        self.assertEqual(analysis['reference_count'], 4)
        self.assertTrue(analysis['ready'])

    def test_textured_video_estimates_camera_motion_and_ignores_subject(self):
        self.check_encoded_scene(rotation=False)

    def test_rotation_is_selected_only_when_it_improves_background_prediction(self):
        self.check_encoded_scene(rotation=True)

    def check_encoded_scene(self, rotation):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'source.avi', Path(directory) / 'output.mp4'
            background = texture()
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'FFV1'), 24, (640, 360))
            motions = [(0, 0), (.35, -.65), (4.4, -3.2), (-5.2, 2.3), (0, 0)]
            expected = []
            for index, (dx, dy) in enumerate(motions):
                scene = background.copy()
                # High-contrast moving foreground: its features must not define
                # the camera path even though the app receives no subject box.
                start = 200 + index * 12
                scene[130:220, start:start + 110] = np.tile(np.arange(110, dtype=np.uint8) % 2 * 240, (90, 1))
                angle = [0, .25, -.4, .35, 0][index] if rotation else 0
                matrix = cv2.getRotationMatrix2D((320, 180), angle, 1)
                matrix[:, 2] += [dx, dy]
                pose = np.eye(3); pose[:2] = matrix; expected.append(pose)
                frame = cv2.warpAffine(scene, matrix, (640, 360), flags=cv2.INTER_LANCZOS4)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
            writer.release()
            report = stabilize(source, output)
            self.assertEqual(report['analysis']['method'], 'scene_features')
            self.assertEqual(report['motion_model'], 'rigid' if rotation else 'translation')
            self.assertEqual(report['frames'], len(motions))
            self.assertEqual(report['output_frames'], len(motions))
            self.assertTrue(report['rigid_geometry_passed'])
            check_points = np.array([[80, 60], [560, 60], [560, 300], [80, 300]])
            for row, truth in zip(report['rows'], expected):
                pose = np.array(row['pose'])
                np.testing.assert_allclose(pose[:2, :2].T @ pose[:2, :2], np.eye(2), atol=1e-10)
                self.assertAlmostEqual(np.linalg.det(pose[:2, :2]), 1, places=10)
                self.assertLess(np.max(np.linalg.norm(project(check_points, pose) - project(check_points, truth), axis=1)), .3)
            self.assertEqual(report['unmeasurable_frames'], 0)
            self.assertLess(report['max_residual_pixels'], .2)


if __name__ == '__main__':
    unittest.main()
