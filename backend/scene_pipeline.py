"""Measure source camera motion, render one rigid correction, audit the export."""
from itertools import zip_longest
from pathlib import Path
import csv
import json
import subprocess

import cv2
import numpy as np

from .light_center import translation_sample_error
from .point_lock import translate, yuv_frames
from .scene_motion import measure, project, robust_pose


ENGINE = 'auto-scene-1'


def render(planes, pose):
    pose = np.asarray(pose, float)
    orientation = pose[:2, :2]
    if not np.allclose(orientation.T @ orientation, np.eye(2), atol=1e-9) or np.linalg.det(orientation) < 0:
        raise ValueError('The camera correction must preserve rigid geometry.')
    if np.allclose(orientation, np.eye(2), atol=1e-12):
        delta = pose[:2, 2]
        return (translate(planes[0], delta), translate(planes[1], delta / 2, 128),
                translate(planes[2], delta / 2, 128))
    # pose maps reference/output coordinates to current/source coordinates.
    # Only rotation and translation are applied, once, to the original frame.
    height, width = planes[0].shape
    luma = cv2.warpAffine(planes[0], pose[:2], (width, height),
                          flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    chroma_pose = pose[:2].copy()
    chroma_origin = np.array([0., .5])  # left-sited 4:2:0
    chroma_pose[:, 2] = (pose[:2, 2] + orientation @ chroma_origin - chroma_origin) / 2
    chroma = [cv2.warpAffine(plane, chroma_pose, (width // 2, height // 2),
                             flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=128) for plane in planes[1:]]
    return luma, *chroma


def rigid_sample_error(source, output, pose, window):
    if np.allclose(pose[:2, :2], np.eye(2), atol=1e-12):
        return translation_sample_error(source, output, pose[:2, 2], window)
    height, width = source.shape
    yy, xx = np.mgrid[5:height - 5:16, 5:width - 5:16]
    destination = np.column_stack((xx.ravel(), yy.ravel()))
    origin = project(destination, pose)
    inside = ((origin[:, 0] >= 4) & (origin[:, 0] < width - 4)
              & (origin[:, 1] >= 4) & (origin[:, 1] < height - 4))
    destination, origin = destination[inside], origin[inside]
    values = cv2.remap(source, origin[:, 0].astype(np.float32).reshape(-1, 1),
                       origin[:, 1].astype(np.float32).reshape(-1, 1), cv2.INTER_LANCZOS4).ravel()
    errors = abs(values.astype(float) - output[destination[:, 1], destination[:, 0]].astype(float))
    return float(errors.max()), float(errors.mean()), len(errors)


def audit_points(analysis, width, height):
    points = np.asarray(analysis['audit_points'], float)
    if len(points) <= 16:
        return points
    selected = []
    cells = set()
    for point in points:
        cell = (int(point[0] * 4 / width), int(point[1] * 3 / height))
        if cell not in cells:
            selected.append(point)
            cells.add(cell)
    return np.asarray(selected)


def image_observations(first, current, points):
    observed = np.full_like(points, np.nan)
    scores = []
    for index, point in enumerate(points):
        x, y = np.rint(point).astype(int)
        template = first[y - 24:y + 25, x - 24:x + 25].astype(np.float32) / 255
        image = current[y - 24:y + 25, x - 24:x + 25].astype(np.float32) / 255
        if np.array_equal(template, image):
            observed[index] = point
            scores.append(1.)
            continue
        try:
            score, warp = cv2.findTransformECC(template, image, np.eye(2, 3, dtype=np.float32),
                                               cv2.MOTION_TRANSLATION, (3, 80, 1e-6), None, 3)
            scores.append(float(score))
            if score >= .5 and np.linalg.norm(warp[:, 2]) < 12:
                observed[index] = point + warp[:, 2]
        except cv2.error:
            scores.append(None)
    return observed, scores


def stabilize(source, destination, progress=lambda *_: None):
    measurements = measure(source, progress)
    metadata = measurements['source']
    width, height = measurements['width'], measurements['height']
    rows = measurements['rows']
    destination = Path(destination)
    if destination.suffix != '.mp4':
        raise ValueError('Automatic scene stabilization requires an MP4 destination.')
    command = [
        'ffmpeg', '-v', 'error', '-nostdin', '-y', '-f', 'rawvideo', '-pix_fmt', 'yuv420p',
        '-s', f'{width}x{height}', '-r', metadata['rate'], '-color_range', 'pc',
        '-colorspace', metadata.get('color_space', 'bt709'), '-i', '-', '-i', str(source),
        '-map', '0:v:0', '-map', '1:a:0?', '-map_metadata', '-1', '-c:a', 'aac',
        '-c:v', 'libx264', '-qp', '1', '-preset', 'fast', '-profile:v', 'high',
        '-threads', '4', '-pix_fmt', 'yuv420p', '-color_range', 'pc',
        '-colorspace', metadata.get('color_space', 'bt709'), '-movflags', '+faststart', str(destination),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    decoder = yuv_frames(source, width, height)
    count = 0
    try:
        for index, planes in enumerate(decoder):
            if index >= len(rows):
                raise ValueError('Source frame count changed between measurement and rendering.')
            for plane in render(planes, np.asarray(rows[index]['pose'])):
                encoder.stdin.write(plane.tobytes())
            count += 1
            progress(.4 + .4 * count / len(rows), f'Correcting camera motion in frame {count} of {len(rows)}')
        encoder.stdin.close()
        error = encoder.stderr.read().decode()
        if encoder.wait():
            raise ValueError(error[-1000:])
    finally:
        decoder.close()
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        encoder.stdin.close()
        encoder.stderr.close()
    if count != len(rows):
        raise ValueError('Source frame count changed between measurement and rendering.')

    progress(.81, 'Checking the actual encoded scene')
    original = yuv_frames(source, width, height)
    decoded = yuv_frames(destination, width, height)
    points = audit_points(measurements['analysis'], width, height)
    window = np.rint(measurements['analysis']['points'][0]).astype(int)
    audit = []
    first = previous_center = None
    image_center = np.array([width / 2, height / 2])
    try:
        for index, pair in enumerate(zip_longest(original, decoded)):
            before, after = pair
            if before is None or after is None or index >= len(rows):
                raise ValueError('Encoded frame count differs from the source.')
            if first is None:
                first = after[0].copy()
            pose = np.asarray(rows[index]['pose'])
            maximum, mean, samples = rigid_sample_error(before[0], after[0], pose, window)
            observed, scores = image_observations(first, after[0], points)
            try:
                residual_pose, inliers, disagreement = robust_pose(
                    points, observed, width, height,
                    rotation=measurements['motion_model'] == 'rigid', tolerance=1.5)
                center_displacement = project(image_center[None, :], residual_pose)[0] - image_center
                if previous_center is None:
                    previous_center = center_displacement
                step = float(np.linalg.norm(center_displacement - previous_center))
                previous_center = center_displacement
                drift = float(np.linalg.norm(center_displacement))
                residual_angle = float(np.degrees(np.arctan2(residual_pose[1, 0], residual_pose[0, 0])))
                inlier_count = int(inliers.sum())
            except ValueError:
                drift = step = residual_angle = disagreement = None
                center_displacement = None
                inlier_count = 0
            audit.append(dict(frame=index, rigid_sample_max_error=maximum,
                              rigid_sample_mean_error=mean, rigid_samples=samples,
                              scene_drift_pixels=drift, scene_step_pixels=step,
                              residual_angle_degrees=residual_angle,
                              audit_reference_disagreement_pixels=disagreement,
                              audit_inliers=inlier_count, reference_scores=scores,
                              observed_points=[[float(v) if np.isfinite(v) else None for v in point] for point in observed],
                              residual_translation=center_displacement.tolist() if center_displacement is not None else None))
            if index % 16 == 0:
                progress(.81 + .18 * (index + 1) / len(rows), f'Auditing encoded frame {index + 1}')
    finally:
        original.close()
        decoded.close()
    if len(audit) != len(rows):
        raise ValueError('Encoded and measured frame counts differ.')
    measurable = [row for row in audit if row['scene_drift_pixels'] is not None]
    unmeasurable = len(audit) - len(measurable)
    sample_limit = 4 if measurements['motion_model'] == 'translation' else 6
    geometry_passed = all(row['rigid_sample_max_error'] <= sample_limit for row in audit)
    report = dict(
        engine=ENGINE, **measurements, frames=len(rows), output_frames=len(audit), extension='mp4',
        anchor=measurements['analysis']['points'][0], window=window.tolist(),
        audit_points=points.tolist(), audit=audit,
        rigid_geometry_passed=geometry_passed,
        max_rigid_sample_error=max(row['rigid_sample_max_error'] for row in audit),
        max_residual_pixels=max((row['scene_drift_pixels'] for row in measurable), default=None),
        max_step_pixels=max((row['scene_step_pixels'] for row in measurable), default=None),
        unmeasurable_frames=unmeasurable,
        zero_motion_passed=unmeasurable == 0 and all(row['scene_drift_pixels'] == 0 and
                            row['scene_step_pixels'] == 0 and row['residual_angle_degrees'] == 0 for row in measurable),
        pixels_unchanged=False, codec='H.264 QP 1, 8-bit 4:2:0; near-lossless, not lossless.',
        timing='Every decoded source frame is retained in order at the nominal source rate.',
        geometry='One camera rotation/translation per frame. No scale, shear, local warps, crop, or subject recentering.',
        measurement='Motion is measured from source frames. Independent ECC checks encoded background patches; '
                    'image-profile variations may disagree with the camera model. No zero-motion claim follows from encoding or geometry checks.',
    )
    destination.with_suffix('.json').write_text(json.dumps(report, indent=2))
    with destination.with_suffix('.csv').open('w') as handle:
        fields = ['frame', 'dx', 'dy', 'angle_degrees', 'inliers', 'reference_rms_pixels',
                  'scene_drift_pixels', 'scene_step_pixels', 'audit_inliers', 'rigid_sample_max_error']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, check in zip(rows, audit):
            writer.writerow(dict(frame=row['frame'], dx=row['pose'][0][2], dy=row['pose'][1][2],
                                 angle_degrees=row['angle_degrees'], inliers=sum(row['inliers']),
                                 reference_rms_pixels=row['reference_rms_pixels'],
                                 **{key: check[key] for key in fields[6:]}))
    if not geometry_passed:
        raise ValueError('The encoded video failed the rigid-geometry sample check. See the frame audit.')
    return report
