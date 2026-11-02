"""Native-luma point lock with a lossless, independently checked export.

The anchor is explicitly the brightest pixel of the selected isolated light.
This does not claim that its entire surrounding light distribution is fixed.
"""
from itertools import zip_longest
from pathlib import Path
import csv
import json
import subprocess

import cv2
import numpy as np

from .tripod import info


def yuv_frames(path, width, height):
    """Decode sequentially, in display orientation, to full-range 4:2:0."""
    command = [
        'ffmpeg', '-v', 'error', '-nostdin', '-i', str(path),
        '-map', '0:v:0', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'yuvj420p', '-',
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    pixels = width * height
    size = pixels * 3 // 2
    try:
        while True:
            data = process.stdout.read(size)
            if not data:
                break
            if len(data) != size:
                raise ValueError('Incomplete decoded video frame.')
            values = np.frombuffer(data, np.uint8)
            yield (
                values[:pixels].reshape(height, width),
                values[pixels:pixels + pixels // 4].reshape(height // 2, width // 2),
                values[pixels + pixels // 4:].reshape(height // 2, width // 2),
            )
        if process.wait():
            raise ValueError('Video decoding failed.')
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            process.wait()


def translate(plane, displacement, fill=0):
    """One rigid translation; integer luma shifts copy original samples exactly.

    Chroma has half the resolution, so an odd luma shift needs an exact half
    chroma-sample translation. Fractional coefficients are not phase-quantized.
    """
    delta = np.asarray(displacement, float)
    integer = np.floor(delta).astype(int)
    fraction = delta - integer
    if np.any(fraction):
        offsets = np.arange(-1, 3)

        def weights(phase):
            distance = np.abs(phase - offsets)
            alpha = -.75
            return np.where(
                distance <= 1,
                (alpha + 2) * distance ** 3 - (alpha + 3) * distance ** 2 + 1,
                np.where(distance < 2,
                         alpha * distance ** 3 - 5 * alpha * distance ** 2
                         + 8 * alpha * distance - 4 * alpha, 0),
            ).astype(np.float32)

        samples = cv2.sepFilter2D(
            plane.astype(np.float32) - fill, cv2.CV_32F,
            weights(fraction[0]), weights(fraction[1]),
            anchor=(1, 1), borderType=cv2.BORDER_CONSTANT,
        ) + fill
        samples = np.clip(np.rint(samples), 0, 255).astype(np.uint8)
    else:
        samples = plane
    height, width = plane.shape
    dx, dy = integer
    x0, x1 = max(0, -dx), min(width, width - dx)
    y0, y1 = max(0, -dy), min(height, height - dy)
    result = np.full_like(plane, fill)
    if x1 > x0 and y1 > y0:
        result[y0:y1, x0:x1] = samples[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return result


def peak(plane, box):
    x, y, width, height = box
    region = plane[y:y + height, x:x + width]
    background = float(np.median(region))
    noise = max(1., 1.4826 * float(np.median(np.abs(region.astype(float) - background))))
    maximum = int(region.max())
    if maximum - background < max(12, 6 * noise):
        raise ValueError('The selected light cannot be distinguished from the background.')
    # Deterministic tie handling: topmost, then leftmost brightest pixel.
    row, column = np.unravel_index(np.argmax(region), region.shape)
    return np.array([x + column, y + row], dtype=int)


def center_of_light(plane, anchor, radius=32):
    """Separate appearance diagnostic, never used to tune or certify peak lock."""
    x, y = anchor
    region = plane[y - radius:y + radius + 1, x - radius:x + radius + 1].astype(float)
    weights = np.maximum(region - np.median(region), 0) ** 8
    total = float(weights.sum())
    if not total:
        raise ValueError('The selected light cannot be measured.')
    return np.array([
        x - radius + np.dot(weights.sum(axis=0), np.arange(region.shape[1])) / total,
        y - radius + np.dot(weights.sum(axis=1), np.arange(region.shape[0])) / total,
    ])


def source_overlap_equal(source, output, displacement):
    """Check output pixels against source by direct indexing, without the renderer."""
    dx, dy = map(int, displacement)
    height, width = source.shape
    sx0, sx1 = max(0, dx), min(width, width + dx)
    sy0, sy1 = max(0, dy), min(height, height + dy)
    return np.array_equal(
        source[sy0:sy1, sx0:sx1],
        output[sy0 - dy:sy1 - dy, sx0 - dx:sx1 - dx],
    )


def reference(source, box):
    metadata = info(source)
    width = metadata['width'] + metadata['width'] % 2
    height = metadata['height'] + metadata['height'] % 2
    x, y, bw, bh = [int(round(value)) for value in box]
    if min(bw, bh) < 12 or x < 0 or y < 0 or x + bw > width or y + bh > height:
        raise ValueError('Draw a box at least 12 pixels wide and high inside the image.')
    decoder = yuv_frames(source, width, height)
    try:
        first = next(decoder)
        anchor = peak(first[0], (x, y, bw, bh))
    finally:
        decoder.close()
    ax, ay = map(int, anchor)
    if min(ax, ay, width - ax - 1, height - ay - 1) < 49:
        raise ValueError('Choose a light at least 49 pixels from the image edge.')
    search_box = (ax - 48, ay - 48, 97, 97)
    if np.any(peak(first[0], search_box) != anchor):
        raise ValueError('A brighter light is too close to this selection. Choose an isolated light.')
    return metadata, width, height, anchor, search_box


def stabilize(source, destination, box, progress=lambda *_: None):
    metadata, width, height, anchor, search_box = reference(source, box)
    destination = Path(destination)
    if destination.suffix != '.webm':
        raise ValueError('Point lock requires the lossless WebM export.')
    command = [
        'ffmpeg', '-v', 'error', '-nostdin', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'yuv420p', '-s', f'{width}x{height}', '-r', metadata['rate'],
        '-color_range', 'pc', '-colorspace', metadata.get('color_space', 'bt709'),
        '-i', '-', '-i', str(source), '-map', '0:v:0', '-map', '1:a:0?',
        '-map_metadata', '-1', '-c:a', 'libopus', '-c:v', 'libvpx-vp9',
        '-lossless', '1', '-deadline', 'good', '-cpu-used', '5', '-row-mt', '1',
        '-tile-columns', '2', '-tile-rows', '0', '-frame-parallel', '1',
        '-threads', '4', '-color_range', 'pc',
        '-colorspace', metadata.get('color_space', 'bt709'), str(destination),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    decoder = yuv_frames(source, width, height)
    rows = []
    try:
        for index, planes in enumerate(decoder):
            position = peak(planes[0], search_box)
            displacement = position - anchor
            if index == 0 and np.any(displacement):
                raise ValueError('The box contains competing lights. Select just one light.')
            corrected = (
                translate(planes[0], displacement),
                translate(planes[1], displacement / 2, fill=128),
                translate(planes[2], displacement / 2, fill=128),
            )
            for plane in corrected:
                encoder.stdin.write(plane.tobytes())
            rows.append(dict(frame=index, source_x=int(position[0]), source_y=int(position[1]),
                             dx=int(displacement[0]), dy=int(displacement[1])))
            progress(min(.8, (index + 1) / metadata['fps'] / metadata['duration'] * .8),
                     f'Locking frame {index + 1}')
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

    progress(.85, 'Checking every exported frame against its original')
    source_decoder = yuv_frames(source, width, height)
    output_decoder = yuv_frames(destination, width, height)
    audit = []
    previous_center = initial_center = None
    try:
        for index, (original, output) in enumerate(zip_longest(source_decoder, output_decoder)):
            if original is None or output is None or index >= len(rows):
                raise ValueError('Export frame count does not match the source.')
            observed = peak(output[0], search_box)
            center = center_of_light(output[0], anchor)
            if initial_center is None:
                initial_center = previous_center = center
            row = rows[index]
            unchanged = source_overlap_equal(original[0], output[0], (row['dx'], row['dy']))
            audit.append(dict(
                frame=index, peak_x=int(observed[0]), peak_y=int(observed[1]),
                peak_offset_pixels=float(np.linalg.norm(observed - anchor)),
                source_luma_preserved=bool(unchanged),
                center_x=float(center[0]), center_y=float(center[1]),
                center_step_pixels=float(np.linalg.norm(center - previous_center)),
                center_drift_pixels=float(np.linalg.norm(center - initial_center)),
            ))
            previous_center = center
            if index % 16 == 0:
                progress(.85 + .14 * (index + 1) / len(rows), f'Checking exported frame {index + 1}')
    finally:
        source_decoder.close()
        output_decoder.close()
    if len(audit) != len(rows):
        raise ValueError('Export frame count does not match the source.')
    maximum_peak = max(row['peak_offset_pixels'] for row in audit)
    preserved = all(row['source_luma_preserved'] for row in audit)
    center_step = max(row['center_step_pixels'] for row in audit)
    center_drift = max(row['center_drift_pixels'] for row in audit)
    report = dict(
        engine='native-peak-lock-3', reference_kind='brightest_pixel', source=metadata,
        box=list(box), anchor=anchor.tolist(), frames=len(rows), output_frames=len(audit),
        extension='webm', rows=rows, audit=audit, max_peak_offset_pixels=maximum_peak,
        peak_lock_passed=maximum_peak == 0 and preserved,
        source_luma_preserved=preserved, max_step_pixels=center_step,
        max_residual_pixels=center_drift, unmeasurable_frames=0,
        zero_motion_passed=maximum_peak == 0 and center_step == 0 and center_drift == 0,
        measurement='Anchor: brightest native luma pixel. Separate appearance diagnostic: '
                    'background-subtracted intensity^8 centroid within 65x65 pixels. '
                    'Peak lock does not certify an unchanged surrounding light distribution.',
    )
    destination.with_suffix('.json').write_text(json.dumps(report, indent=2))
    with destination.with_suffix('.csv').open('w') as handle:
        fields = list(rows[0]) + list(audit[0])[1:]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dict(**row, **{k: v for k, v in check.items() if k != 'frame'})
                         for row, check in zip(rows, audit))
    if not report['peak_lock_passed']:
        raise ValueError('The exported reference pixel did not remain fixed. See the frame audit.')
    return report
