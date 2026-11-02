"""Subpixel translation of a reference light, with an audit of decoded output.

The tracker measures a light distribution, not the coordinate of its maximum.
No temporal replacement, deconvolution, scale, rotation, or local warp is used.
The separate profile diagnostics must not be confused with the tracked center.
"""
from itertools import zip_longest
from pathlib import Path
import csv
import json
import subprocess

import cv2
import numpy as np

from .point_lock import peak, reference as point_reference, translate, yuv_frames


ENGINE = 'light-center-1'


def center(plane, window, exponent=8):
    x, y = map(int, window)
    region = plane[y - 32:y + 33, x - 32:x + 33].astype(np.float64)
    if region.shape != (65, 65):
        raise ValueError('The reference light is too close to an image edge.')
    signal = np.maximum(region - np.median(region), 0)
    maximum = float(signal.max())
    if maximum < 12:
        raise ValueError('The reference light has insufficient contrast.')
    # Normalize before exponentiation; this does not change the centroid.
    weights = (signal / maximum) ** exponent
    total = weights.sum()
    return np.array([
        x - 32 + np.dot(weights.sum(axis=0), np.arange(65)) / total,
        y - 32 + np.dot(weights.sum(axis=1), np.arange(65)) / total,
    ])


def reference(source, box):
    metadata, width, height, window, search = point_reference(source, box)
    decoder = yuv_frames(source, width, height)
    try:
        anchor = center(next(decoder)[0], window)
    finally:
        decoder.close()
    return metadata, width, height, anchor, window, search


def refine(plane, window, anchor, position, maximum_iterations=16):
    """Fit only a translation, accounting for 8-bit interpolation rounding.

    Measure a small rendering of the actual source, retaining a margin around
    both reference and source light. The full frame is rendered exactly once.
    """
    source_peak = np.asarray(position, int)
    origin = np.minimum(window, source_peak) - 36
    end = np.maximum(window, source_peak) + 37
    patch = plane[origin[1]:end[1], origin[0]:end[0]]
    target_window = window - origin
    target = anchor - origin
    initial = center(plane, source_peak) - anchor

    def evaluate(displacement):
        return center(translate(patch, displacement), target_window) - target

    current = initial.copy()
    best_error = float('inf')
    best_position = current.copy()
    for _ in range(maximum_iterations):
        error = evaluate(current)
        norm = float(np.linalg.norm(error))
        if norm < best_error:
            best_error, best_position = norm, current.copy()
        if norm < .002:
            break
        epsilon = .08
        jacobian = np.column_stack([
            (evaluate(current + axis * epsilon) - evaluate(current - axis * epsilon)) / (2 * epsilon)
            for axis in np.eye(2)
        ])
        correction = np.clip(np.linalg.lstsq(jacobian, -error, rcond=None)[0], -1, 1)
        candidates = [current + correction * gain for gain in (1, .5, .25)]
        candidates += [current + np.array(axis) * .02 for axis in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        next_error, next_position = min(
            ((float(np.linalg.norm(evaluate(candidate))), candidate) for candidate in candidates),
            key=lambda item: item[0],
        )
        if next_error < best_error:
            best_error, best_position = next_error, next_position.copy()
        if next_error >= norm:
            break
        current = next_position
    if not np.isfinite(best_position).all() or np.max(np.abs(best_position)) >= 48:
        raise ValueError('The light moved outside the tripod registration range.')
    return best_position, best_error


def profile_observations(plane, window):
    """Foreground moments across brightness weightings.

    Scene tracking uses the mass centroid. Selected-light mode uses these as
    separate diagnostics rather than fitting them all to the same coordinate.
    """
    x, y = map(int, window)
    region = plane[y - 32:y + 33, x - 32:x + 33].astype(np.float64)
    background = float(np.median(region))
    noise = max(1., 1.4826 * float(np.median(abs(region - background))))
    signal = np.maximum(region - background - 3 * noise, 0)
    mask = (signal > max(8., signal.max() * .08)).astype(np.uint8)
    _, labels = cv2.connectedComponents(mask)
    label = labels[np.unravel_index(np.argmax(signal), signal.shape)]
    if label == 0:
        raise ValueError('The light profile cannot be measured in the exported frame.')
    mask = cv2.dilate((labels == label).astype(np.uint8), np.ones((3, 3), np.uint8))
    signal *= mask
    centers = {}
    for exponent in (1, 2, 4, 8):
        weights = (signal / signal.max()) ** exponent
        centers[f'profile_center_{exponent}'] = np.array([
            x - 32 + np.dot(weights.sum(0), np.arange(65)) / weights.sum(),
            y - 32 + np.dot(weights.sum(1), np.arange(65)) / weights.sum(),
        ])
    return centers


def translation_sample_error(source, output, displacement, window, visible_box=None):
    """Independently gather source samples at known output coordinates.

    This does not call the renderer. Check a grid over the entire image plus
    every sample in the reference patch. Exclude the uncovered image boundary.
    H.264 QP 1 is near-lossless, so report the actual sample error, not equality.
    """
    height, width = output.shape
    gy, gx = np.mgrid[4:height - 4:16, 4:width - 4:16]
    x, y = map(int, window)
    py, px = np.mgrid[y - 32:y + 33, x - 32:x + 33]
    ox = np.concatenate((gx.ravel(), px.ravel()))
    oy = np.concatenate((gy.ravel(), py.ravel()))
    if visible_box is not None:
        vx, vy, vw, vh = visible_box
        visible = (ox >= vx) & (ox < vx + vw) & (oy >= vy) & (oy < vy + vh)
        ox, oy = ox[visible], oy[visible]
    sx = ox + displacement[0]
    sy = oy + displacement[1]
    ix, iy = np.floor(sx).astype(int), np.floor(sy).astype(int)
    inside = (ix >= 1) & (iy >= 1) & (ix + 2 < width) & (iy + 2 < height)
    ox, oy, ix, iy, sx, sy = (a[inside] for a in (ox, oy, ix, iy, sx, sy))

    def kernel(distance):
        t = np.abs(distance)
        return np.where(t <= 1, 1.25 * t**3 - 2.25 * t**2 + 1,
                        np.where(t < 2, -.75 * t**3 + 3.75 * t**2 - 6 * t + 3, 0))

    expected = np.zeros(len(ox), np.float64)
    for row in (-1, 0, 1, 2):
        for column in (-1, 0, 1, 2):
            expected += (source[iy + row, ix + column] * kernel(sx - ix - column)
                         * kernel(sy - iy - row))
    expected = np.clip(np.rint(expected), 0, 255)
    errors = abs(output[oy, ox].astype(float) - expected)
    return float(errors.max()), float(errors.mean()), len(errors)


def stabilize(source, destination, box, progress=lambda *_: None, *, crop_to_selection=False):
    metadata, width, height, anchor, window, search = reference(source, box)
    crop_box = tuple(int(round(value)) for value in box) if crop_to_selection else None
    crop_x, crop_y, content_width, content_height = crop_box or (0, 0, width, height)
    output_width = content_width + content_width % 2
    output_height = content_height + content_height % 2
    crop_origin = np.array([crop_x, crop_y])
    destination = Path(destination)
    if destination.suffix != '.mp4':
        raise ValueError('Light-center stabilization requires an MP4 destination.')
    space = metadata.get('color_space', 'bt709')
    command = [
        'ffmpeg', '-v', 'error', '-nostdin', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'yuv420p', '-s', f'{output_width}x{output_height}', '-r', metadata['rate'],
        '-color_range', 'pc', '-colorspace', space,
        '-i', '-', '-i', str(source), '-map', '0:v:0', '-map', '1:a:0?',
        '-map_metadata', '-1', '-c:a', 'aac', '-c:v', 'libx264', '-qp', '1',
        '-preset', 'fast', '-profile:v', 'high', '-threads', '4',
        '-pix_fmt', 'yuv420p', '-color_range', 'pc', '-colorspace', space,
        '-movflags', '+faststart', str(destination),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    decoder = yuv_frames(source, width, height)
    rows = []
    try:
        for index, planes in enumerate(decoder):
            position = peak(planes[0], search)
            if index == 0:
                displacement, fit_error = np.zeros(2), 0.
            else:
                displacement, fit_error = refine(planes[0], window, anchor, position)
            # Combine the camera correction and crop origin into one sampling
            # operation. Odd crop origins retain the correct chroma phase.
            effective = displacement + crop_origin
            luma = translate(planes[0], effective)[:content_height, :content_width]
            if content_width != output_width or content_height != output_height:
                luma = np.pad(luma, ((0, output_height - content_height),
                                    (0, output_width - content_width)), constant_values=0)
            corrected = (luma, *(translate(plane, effective / 2, 128)
                                [:output_height // 2, :output_width // 2] for plane in planes[1:]))
            for plane in corrected:
                encoder.stdin.write(plane.tobytes())
            measured = center(planes[0], position)
            rows.append(dict(frame=index, source_x=float(measured[0]), source_y=float(measured[1]),
                             dx=float(displacement[0]), dy=float(displacement[1]), fit_error_pixels=fit_error))
            progress(min(.8, (index + 1) / metadata['fps'] / metadata['duration'] * .8),
                     f'Aligning light center in frame {index + 1}')
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

    progress(.81, 'Measuring the actual encoded frames')
    source_decoder = yuv_frames(source, width, height)
    output_decoder = yuv_frames(destination, output_width, output_height)
    first = previous = first_profile = previous_profile = first_patch = previous_patch = None
    audit = []
    x, y = map(int, window)
    try:
        for index, (original, output) in enumerate(zip_longest(source_decoder, output_decoder)):
            if original is None or output is None or index >= len(rows):
                raise ValueError('The encoded frame count differs from the source.')
            if crop_box is not None:
                # Place only the visible decoded crop in source coordinates for
                # measurement. Discarded pixels never enter the geometry check.
                visible = np.zeros((height, width), np.uint8)
                visible[crop_y:crop_y + content_height, crop_x:crop_x + content_width] = output[0][:content_height, :content_width]
                output = (visible, *output[1:])
            observed = center(output[0], window)
            profile = profile_observations(output[0], window)
            current_patch = output[0][y - 32:y + 33, x - 32:x + 33].astype(np.float32) / 255
            if first is None:
                first = previous = observed.copy()
                first_profile = previous_profile = profile
                first_patch = previous_patch = current_patch
            maximum = peak(output[0], (x - 32, y - 32, 65, 65))
            row = dict(frame=index, center_x=float(observed[0] - crop_x), center_y=float(observed[1] - crop_y),
                       center_drift_pixels=float(np.linalg.norm(observed - first)),
                       center_step_pixels=float(np.linalg.norm(observed - previous)),
                       center_target_error_pixels=float(np.linalg.norm(observed - anchor)),
                       peak_x=int(maximum[0] - crop_x), peak_y=int(maximum[1] - crop_y))
            sample_maximum, sample_mean, samples = translation_sample_error(
                original[0], output[0], (rows[index]['dx'], rows[index]['dy']), window, crop_box)
            row.update(translation_sample_max_error=sample_maximum,
                       translation_sample_mean_error=sample_mean, translation_samples=samples)
            for name, point in profile.items():
                row[name + '_drift_pixels'] = float(np.linalg.norm(point - first_profile[name]))
                row[name + '_step_pixels'] = float(np.linalg.norm(point - previous_profile[name]))
            for name, template in (('profile_registration_drift', first_patch), ('profile_registration_step', previous_patch)):
                try:
                    score, warp = cv2.findTransformECC(template, current_patch, np.eye(2, 3, dtype=np.float32),
                                                     cv2.MOTION_TRANSLATION, (3, 100, 1e-6), None, 3)
                    row[name + '_pixels'] = float(np.linalg.norm(warp[:, 2]))
                    row[name + '_score'] = float(score)
                except cv2.error:
                    row[name + '_pixels'] = row[name + '_score'] = None
            audit.append(row)
            previous, previous_profile, previous_patch = observed, profile, current_patch
            if index % 16 == 0:
                progress(.81 + .18 * (index + 1) / len(rows), f'Auditing encoded frame {index + 1}')
    finally:
        source_decoder.close()
        output_decoder.close()
    if not rows or len(audit) != len(rows):
        raise ValueError('Source and encoded frame counts differ.')
    maximum_drift = max(row['center_drift_pixels'] for row in audit)
    maximum_step = max(row['center_step_pixels'] for row in audit)
    profile_metrics = {name: max(row[name] for row in audit)
                       for name in audit[0] if name.startswith('profile_center_')}
    unmeasurable = sum(row['profile_registration_drift_pixels'] is None or
                       row['profile_registration_step_pixels'] is None for row in audit)
    report = dict(
        engine=ENGINE, source=metadata, box=list(box), anchor=(anchor - crop_origin).tolist(),
        window=(window - crop_origin).tolist(), source_anchor=anchor.tolist(), source_window=window.tolist(),
        crop_to_selection=bool(crop_to_selection), crop_box=list(crop_box) if crop_box else None,
        output_width=output_width, output_height=output_height,
        frames=len(rows), output_frames=len(audit), extension='mp4', rows=rows, audit=audit,
        max_residual_pixels=maximum_drift, max_step_pixels=maximum_step,
        max_target_error_pixels=max(row['center_target_error_pixels'] for row in audit),
        center_lock_passed=maximum_drift <= .05 and maximum_step <= .05,
        rigid_translation_samples_passed=all(row['translation_sample_max_error'] <= 4 for row in audit),
        max_translation_sample_error=max(row['translation_sample_max_error'] for row in audit),
        profile_metrics=profile_metrics, unmeasurable_frames=unmeasurable,
        zero_motion_passed=maximum_drift == 0 and maximum_step == 0
                           and all(value == 0 for value in profile_metrics.values()) and unmeasurable == 0,
        geometry='One uniform subpixel translation per frame. No rotation, scale, or local warping. '
                 + ('Fixed crop to the selected first-frame box; odd dimensions receive one black padding pixel.'
                    if crop_box else 'Full frame; no crop.'),
        pixels_unchanged=False, codec='H.264 QP 1, 8-bit 4:2:0; near-lossless, not lossless.',
        measurement='Tracked reference: median-background-subtracted native-luma intensity^8 centroid '
                    'in a 65x65 window. Separate foreground profile moments and ECC are diagnostics; '
                    'center_lock_passed does not certify zero movement of the whole visible star.',
    )
    destination.with_suffix('.json').write_text(json.dumps(report, indent=2))
    with destination.with_suffix('.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) + list(audit[0])[1:])
        writer.writeheader()
        writer.writerows(dict(**row, **{key: value for key, value in observation.items() if key != 'frame'})
                         for row, observation in zip(rows, audit))
    if not report['rigid_translation_samples_passed']:
        raise ValueError('The encoded selected-light export failed its rigid-translation sample check.')
    return report
