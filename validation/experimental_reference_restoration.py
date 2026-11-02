"""OFFLINE EXPERIMENT: guide-light restoration, not faithful rigid stabilization.

Applies a frame-specific global convolution derived from the reference profile.
It changes pixel values and can create ringing, change apparent object detail,
and amplify noise. This is deliberately NOT imported by the serving app.
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from backend.point_lock import center_of_light, peak, translate, yuv_frames
from backend.tripod import info


def profile(patch):
    patch = patch.astype(float)
    background = float(np.median(patch))
    noise = max(1., 1.4826 * float(np.median(np.abs(patch - background))))
    values = np.maximum(patch - background - 2 * noise, 0)
    yy, xx = np.mgrid[:65, :65]
    values *= np.clip((32 - np.hypot(xx - 32, yy - 32)) / 5, 0, 1)
    return values / max(float(values.sum()), 1.)


def spectrum(patch, shape):
    values = np.zeros(shape, np.float32)
    cy, cx = np.array(shape) // 2
    values[cy - 32:cy + 33, cx - 32:cx + 33] = profile(patch)
    return np.fft.rfft2(np.fft.ifftshift(values))


def restore_plane(plane, transfer, padding, fill):
    py, px = padding
    samples = np.pad(plane.astype(np.float32) - fill, ((py, py), (px, px)), mode='reflect')
    corrected = np.fft.irfft2(np.fft.rfft2(samples) * transfer, s=samples.shape)
    corrected = corrected[py:py + plane.shape[0], px:px + plane.shape[1]] + fill
    return np.clip(np.rint(corrected), 0, 255).astype(np.uint8)


def inspect_patch(first, current):
    result = {}
    for name, exponent in [('mass_center', 1), ('bright_center', 8)]:
        centers = []
        for patch in [first, current]:
            weights = np.maximum(patch.astype(float) - np.median(patch), 0) ** exponent
            total = weights.sum()
            centers.append(np.array([np.dot(weights.sum(0), np.arange(65)), np.dot(weights.sum(1), np.arange(65))]) / total)
        result[name + '_offset'] = float(np.linalg.norm(centers[1] - centers[0]))
    try:
        score, warp = cv2.findTransformECC(
            first.astype(np.float32) / 255, current.astype(np.float32) / 255,
            np.eye(2, 3, dtype=np.float32), cv2.MOTION_TRANSLATION, (3, 100, 1e-6), None, 3,
        )
        result.update(profile_offset=float(np.linalg.norm(warp[:, 2])), profile_correlation=float(score))
    except cv2.error:
        result.update(profile_offset=None, profile_correlation=None)
    return result


def render(source, prefix, regularization=.0001, gain_limit=2.):
    metadata = info(source)
    width, height = metadata['width'], metadata['height']
    if width % 2 or height % 2:
        raise ValueError('This offline experiment currently requires even dimensions.')
    anchor = np.array([1296, 266]); ax, ay = anchor
    search = (ax - 48, ay - 48, 97, 97)
    fft_width = cv2.getOptimalDFTSize(width + 128)
    fft_height = cv2.getOptimalDFTSize(height + 128)
    while (fft_width - width) % 4:
        fft_width = cv2.getOptimalDFTSize(fft_width + 1)
    while (fft_height - height) % 4:
        fft_height = cv2.getOptimalDFTSize(fft_height + 1)
    padding = ((fft_height - height) // 2, (fft_width - width) // 2)
    chroma_shape = (fft_height // 2, fft_width // 2)
    vertical_frequencies = np.rint(np.fft.fftfreq(chroma_shape[0]) * chroma_shape[0]).astype(int) % fft_height
    prefix = Path(prefix)
    output = prefix.with_suffix('.webm'); review = prefix.with_name(prefix.name + '-reference-review.mp4')
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-y', '-f', 'rawvideo', '-pix_fmt', 'yuv420p',
               '-s', f'{width}x{height}', '-r', metadata['rate'], '-color_range', 'pc', '-colorspace', 'bt709',
               '-i', '-', '-an', '-c:v', 'libvpx-vp9', '-lossless', '1', '-deadline', 'good', '-cpu-used', '5',
               '-tile-columns', '2', '-tile-rows', '0', '-frame-parallel', '1', '-row-mt', '1', '-threads', '4',
               '-color_range', 'pc', '-colorspace', 'bt709', str(output)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    preview = subprocess.Popen(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray',
                                '-s', '780x430', '-r', metadata['rate'], '-i', '-', '-an', '-c:v', 'libx264',
                                '-preset', 'fast', '-threads', '2', '-crf', '10', '-pix_fmt', 'yuv420p',
                                '-movflags', '+faststart', str(review)], stdin=subprocess.PIPE)
    decoder = yuv_frames(source, width, height)
    first_spectrum = first_patch = None
    records = []; examples = []; selected_frames = {0, 35, 39, 187, 255, 283, 287}
    try:
        for index, original in enumerate(decoder):
            position = peak(original[0], search)
            delta = position - anchor
            registered = (translate(original[0], delta), translate(original[1], delta / 2, 128), translate(original[2], delta / 2, 128))
            patch = registered[0][ay - 32:ay + 33, ax - 32:ax + 33]
            current_spectrum = spectrum(patch, (fft_height, fft_width))
            if first_spectrum is None:
                first_spectrum = current_spectrum.copy(); first_patch = patch.copy()
            if index == 0:
                restored = registered
            else:
                transfer = first_spectrum * current_spectrum.conj() / (np.abs(current_spectrum)**2 + regularization)
                transfer /= transfer[0, 0]
                transfer *= np.minimum(1, gain_limit / np.maximum(np.abs(transfer), 1e-12))
                chroma_transfer = transfer[vertical_frequencies, :chroma_shape[1] // 2 + 1]
                restored = (
                    restore_plane(registered[0], transfer, padding, 0),
                    restore_plane(registered[1], chroma_transfer, tuple(p // 2 for p in padding), 128),
                    restore_plane(registered[2], chroma_transfer, tuple(p // 2 for p in padding), 128),
                )
            for plane in restored:
                encoder.stdin.write(plane.tobytes())
            result_patch = restored[0][ay - 32:ay + 33, ax - 32:ax + 33]
            records.append(dict(frame=index, dx=int(delta[0]), dy=int(delta[1]), **inspect_patch(first_patch, result_patch)))
            panels = []
            for label, values in [('RIGID-ONLY REFERENCE', patch), ('RESTORATION EXPERIMENT', result_patch)]:
                panel = np.zeros((430, 390), np.uint8)
                panel[40:] = cv2.resize(values, (390, 390), interpolation=cv2.INTER_NEAREST)
                cv2.putText(panel, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, 230, 1)
                cv2.putText(panel, f'Frame {index}', (8, 34), cv2.FONT_HERSHEY_SIMPLEX, .4, 180, 1)
                panels.append(panel)
            preview.stdin.write(np.concatenate(panels, axis=1).tobytes())
            if index in selected_frames:
                # Fixed native-position reference crops. No result recentering.
                examples.append(np.concatenate(panels, axis=1))
            if index % 25 == 0:
                print(f'Restoration experiment: {index + 1} frames', flush=True)
        encoder.stdin.close(); preview.stdin.close()
        if encoder.wait() or preview.wait():
            raise ValueError('Experimental encoding failed.')
    finally:
        decoder.close()
        for process in (encoder, preview):
            if process.poll() is None:
                process.kill(); process.wait()
    report = dict(experimental=True, source=str(source), output=str(output), reference_review=str(review),
                  frames=len(records), anchor=anchor.tolist(), regularization=regularization, gain_limit=gain_limit,
                  processing='Integer translation plus a spatially uniform frame-specific reference-profile filter.',
                  source_pixels_unchanged=False, pure_rigid_stabilization=False, visual_qa_passed=False,
                  limitation='This changes image detail, may amplify noise or create ringing, and has NOT been approved for the serving app or scientific interpretation.',
                  rows=records)
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2))
    cv2.imwrite(str(prefix.with_name(prefix.name + '-review.png')), np.concatenate(examples, axis=0))
    print(json.dumps({key: value for key, value in report.items() if key != 'rows'}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    render(args.source, args.prefix)
