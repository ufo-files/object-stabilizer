"""Bidirectional tracking of a transient moving light, seeded on any frame."""
from pathlib import Path
from itertools import islice, zip_longest
import csv
import hashlib
import json
import uuid
import subprocess

import cv2
import numpy as np

from .tripod import info
from .point_lock import yuv_frames, translate
from .light_center import translation_sample_error

ENGINE = 'moving-light-segment-2'


def describe(source):
    metadata = info(source)
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(source)]))
    metadata['frame_count'] = int(probe['streams'][0]['nb_read_frames'])
    return metadata


def frame_at(source, index):
    if index < 0:
        raise ValueError('Frame numbers must be nonnegative.')
    # Decode in sequence, in display orientation. Seeking by time can select a
    # neighboring frame, particularly with long GOPs or variable frame timing.
    result = subprocess.run([
        'ffmpeg', '-v', 'error', '-nostdin', '-i', str(source), '-map', '0:v:0',
        '-vf', f'select=eq(n\\,{index})', '-frames:v', '1', '-f', 'image2pipe',
        '-c:v', 'png', '-'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode or not result.stdout:
        raise ValueError('That frame is outside the video or cannot be decoded.')
    return result.stdout


def patch_at(plane, point, radius=12):
    x, y = np.rint(point).astype(int)
    result = np.zeros((2*radius+1,2*radius+1),dtype=plane.dtype)
    x0,x1=max(0,x-radius),min(plane.shape[1],x+radius+1)
    y0,y1=max(0,y-radius),min(plane.shape[0],y+radius+1)
    if x1>x0 and y1>y0:
        result[y0-(y-radius):y1-(y-radius),x0-(x-radius):x1-(x-radius)]=plane[y0:y1,x0:x1]
    return result


def moment(patch, exponent=8):
    signal = np.maximum(patch.astype(float) - np.median(patch), 0)
    maximum = float(signal.max())
    if maximum < 3:
        raise ValueError('The light is not measurable in this frame.')
    weights = (signal / maximum) ** exponent
    yy, xx = np.indices(weights.shape)
    return np.array([(weights*xx).sum(), (weights*yy).sum()]) / weights.sum()


def feature(plane, position):
    patch = patch_at(plane, position, 8)
    local = moment(patch)
    measured = np.rint(position) - 8 + local
    profile = patch.astype(float) - np.median(patch)
    norm = np.linalg.norm(profile)
    return dict(point=measured, contrast=float(profile.max()), profile=profile / max(norm, 1e-9))


def candidates(plane):
    background = cv2.GaussianBlur(plane.astype(np.float32), (0, 0), 5)
    signal = plane.astype(np.float32) - background
    sample = signal[::4, ::4]
    noise = max(1., 1.4826 * float(np.median(abs(sample - np.median(sample)))))
    mask = (signal > max(3., 5*noise)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    found = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area > 1024 or max(w, h) > 64:
            continue
        region = signal[y:y+h, x:x+w]
        yy, xx = np.unravel_index(np.argmax(np.where(labels[y:y+h, x:x+w] == label, region, -np.inf)), region.shape)
        try:
            found.append(feature(plane, (x+xx, y+yy)))
        except ValueError:
            continue
    # Separate threshold islands can converge to the very same light centroid
    # (compression specks beside a bright core are common in dark footage).
    # Recenter each patch before merging so tracking uses one stable profile.
    unique, cells = [], {}
    for item in sorted(found, key=lambda item: item['contrast'], reverse=True):
        item = feature(plane, item['point'])
        cx, cy = np.floor(item['point']).astype(int)
        neighbors = (other for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                     for other in cells.get((cx+dx, cy+dy), ()))
        if any(np.linalg.norm(item['point']-other['point']) < 1. for other in neighbors):
            continue
        unique.append(item)
        cells.setdefault((cx, cy), []).append(item)
    return unique


def selected(plane, box):
    x, y, w, h = (int(round(v)) for v in box)
    if min(w, h) < 12 or x < 0 or y < 0 or x+w > plane.shape[1] or y+h > plane.shape[0]:
        raise ValueError('Draw a box of at least 12 × 12 pixels inside the frame.')
    inside = [item for item in candidates(plane) if x <= item['point'][0] < x+w and y <= item['point'][1] < y+h]
    if not inside:
        raise ValueError('No distinct light was found in this box. Choose a frame where it is clearly visible.')
    inside.sort(key=lambda item: item['contrast'], reverse=True)
    # Weak sensor/compression flecks are not competing target lights. Keep the
    # ambiguity guard for genuinely separate lights of comparable brightness.
    if len(inside) > 1 and inside[1]['contrast'] > inside[0]['contrast'] / 2:
        raise ValueError('Two distinct lights have similar brightness in this box. Center the box on the intended light.')
    return inside[0]


def follow(detections, seed, origin, box):
    positions = {seed: origin['point']}
    reasons = {}
    initial_gate = max(16., min(128., max(box[2:])*2))
    for direction, name in ((-1, 'before'), (1, 'after')):
        point = origin['point'].copy()
        velocity = np.zeros(2)
        steps = 0
        for index in range(seed+direction, len(detections) if direction > 0 else -1, direction):
            predicted = point + velocity
            gate = initial_gate if not steps else min(initial_gate, max(6., 4+2*np.linalg.norm(velocity)))
            scored = []
            for item in detections[index]:
                distance = float(np.linalg.norm(item[:2]-predicted))
                ratio = item[2] / origin['contrast']
                similarity = item[3]
                if distance <= gate and .005 <= ratio <= 200 and similarity >= .45:
                    scored.append((distance + 2*(1-similarity), item))
            scored.sort(key=lambda pair: pair[0])
            if not scored:
                reasons[name] = 'light-not-detected'
                break
            if len(scored) > 1 and scored[1][0]-scored[0][0] < 1:
                reasons[name] = 'ambiguous-match'
                break
            new = scored[0][1][:2].copy()
            velocity = new-point
            point = new
            positions[index] = new
            steps += 1
        else:
            reasons[name] = 'video-boundary'
    return positions, reasons


def analyse(source, box, seed, progress=lambda *_: None):
    source = Path(source)
    key = hashlib.sha256(json.dumps([ENGINE, list(box), seed]).encode()).hexdigest()[:24]
    cache = source.parent / f'light-track-{key}.json'
    if cache.exists():
        return json.loads(cache.read_text())
    metadata = describe(source)
    if min(metadata['width'],metadata['height']) < 66:
        raise ValueError('Moving-light tracking requires video dimensions of at least 66 pixels.')
    if not 0 <= seed < metadata['frame_count']:
        raise ValueError('Choose a reference frame inside the video.')
    seed_image = cv2.imdecode(np.frombuffer(frame_at(source, seed), np.uint8), cv2.IMREAD_GRAYSCALE)
    origin = selected(seed_image, box)
    width = metadata['width'] + metadata['width'] % 2
    height = metadata['height'] + metadata['height'] % 2
    detections = []
    decoder = yuv_frames(source, width, height)
    try:
        for index, planes in enumerate(decoder):
            # Measure the seed in the same decoded native-luma space as tracking.
            if index == seed:
                origin = selected(planes[0], box)
            found = []
            for item in candidates(planes[0]):
                similarity = float(np.sum(item['profile']*origin['profile']))
                if similarity >= .45:
                    found.append((*item['point'], item['contrast'], similarity))
            # Keep compact measurements, not 17x17 profiles for every noise
            # speck in every frame. Only candidates near the predicted track
            # compete in follow(); unrelated parts of the sky cannot veto it.
            detections.append(np.asarray(found,dtype=np.float64).reshape(-1,4))
            progress(.3*(index+1)/metadata['frame_count'], f'Finding moving-light candidates in frame {index+1}')
    finally:
        decoder.close()
    positions, reasons = follow(detections, seed, origin, box)
    first, last = min(positions), max(positions)
    result = dict(engine=ENGINE, metadata=metadata, width=width, height=height,
                  reference_frame=seed, anchor=origin['point'].tolist(),
                  first_visible_frame=first, last_visible_frame=last,
                  start_frame=max(0, first-1), end_frame=min(len(detections)-1, last+1),
                  frame_count=len(detections), boundary_reasons=reasons,
                  positions={str(index): point.tolist() for index, point in positions.items()})
    temporary = cache.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(result));temporary.replace(cache)
    return result


def fitted_shift(plane, position, anchor):
    """Refine a uniform shift on a small patch; movement is not limited to 48px."""
    peak = np.rint(position).astype(int)
    target = np.rint(anchor).astype(int)
    patch = patch_at(plane, peak, 16)
    desired = anchor-target+16
    shift = np.asarray(position)-anchor
    best = shift.copy();best_error = float('inf')
    def error(delta):
        rendered = translate(patch, delta-(peak-target))
        return moment(rendered[8:25, 8:25])+8-desired
    for _ in range(16):
        residual = error(shift);norm = float(np.linalg.norm(residual))
        if norm < best_error:best, best_error = shift.copy(), norm
        if norm < .003:break
        jacobian = np.column_stack([(error(shift+axis*.08)-error(shift-axis*.08))/.16 for axis in np.eye(2)])
        correction = np.clip(np.linalg.lstsq(jacobian, -residual, rcond=None)[0], -1, 1)
        options = [shift+correction*gain for gain in (1, .5, .25)]
        candidate = min(options, key=lambda value: np.linalg.norm(error(value)))
        if np.linalg.norm(error(candidate)) >= norm:break
        shift = candidate
    return best


def stabilize(source, destination, box, reference_frame, progress=lambda *_: None, *, crop_to_selection=False):
    track = analyse(source, box, reference_frame, progress)
    metadata = track['metadata'];width, height = track['width'], track['height']
    anchor = np.asarray(track['anchor']);window = np.rint(anchor).astype(int)
    first, last = track['first_visible_frame'], track['last_visible_frame']
    begin, end = track['start_frame'], track['end_frame']
    positions = {int(k): np.asarray(v) for k,v in track['positions'].items()}
    crop_box = tuple(int(round(v)) for v in box) if crop_to_selection else None
    x, y, cw, ch = crop_box or (0, 0, width, height)
    ow, oh = cw+cw%2, ch+ch%2
    destination = Path(destination)
    span = end-begin+1
    audio = f'atrim=start={begin/metadata["fps"]}:end={(end+1)/metadata["fps"]},asetpts=PTS-STARTPTS'
    encoder = subprocess.Popen([
        'ffmpeg','-v','error','-nostdin','-y','-f','rawvideo','-pix_fmt','yuv420p',
        '-s',f'{ow}x{oh}','-r',metadata['rate'],'-color_range','pc','-i','-',
        '-i',str(source),'-map','0:v:0','-map','1:a:0?','-af',audio,
        '-map_metadata','-1','-c:v','libx264','-qp','1','-preset','fast',
        '-pix_fmt','yuv420p','-color_range','pc','-threads','4','-c:a','aac',
        '-movflags','+faststart',str(destination)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    rows=[];shifts={};pending_before=None
    def write_frame(index, planes, displacement, visible):
        effective=displacement+[x,y]
        luma=translate(planes[0],effective)[:ch,:cw]
        luma=np.pad(luma,((0,oh-ch),(0,ow-cw)),constant_values=0)
        corrected=(luma,*(translate(p,effective/2,128)[:oh//2,:ow//2] for p in planes[1:]))
        for plane in corrected:encoder.stdin.write(plane.tobytes())
        point=positions.get(index)
        rows.append(dict(frame=len(rows),source_frame=index,visible=visible,
                         source_x=None if point is None else float(point[0]),
                         source_y=None if point is None else float(point[1]),
                         dx=float(displacement[0]),dy=float(displacement[1])))
        progress(.3+.45*len(rows)/span,f'Locking moving light in source frame {index+1}')
    decoder=yuv_frames(source,width,height)
    try:
        for index,planes in enumerate(decoder):
            if index < begin:continue
            if index > end:break
            if index < first:
                pending_before=(index,planes)
                continue
            visible = index in positions
            if visible:
                displacement=fitted_shift(planes[0],positions[index],anchor)
                shifts[index]=displacement
                if pending_before is not None:
                    write_frame(*pending_before,displacement,False)
                    pending_before=None
            else:
                displacement=shifts[last]
            write_frame(index,planes,displacement,visible)
        encoder.stdin.close();error=encoder.stderr.read().decode()
        if encoder.wait():raise ValueError(error[-1000:])
    finally:
        decoder.close()
        if encoder.poll() is None:encoder.kill();encoder.wait()
        encoder.stdin.close();encoder.stderr.close()
    original=yuv_frames(source,width,height);decoded=yuv_frames(destination,ow,oh)
    audit=[];previous=None;previous_profile=None;initial_profile=None
    safe_window=np.clip(window,[32,32],[width-33,height-33])
    try:
        for index,(planes,output) in enumerate(zip_longest(islice(original,begin,end+1),decoded)):
            if planes is None or output is None or index >= len(rows):raise ValueError('Export frame count differs from the selected source range.')
            row=rows[index]
            visible=np.zeros((height,width),np.uint8)
            visible[y:y+ch,x:x+cw]=output[0][:ch,:cw]
            maximum,mean,samples=translation_sample_error(planes[0],visible,(row['dx'],row['dy']),safe_window,crop_box)
            observation=dict(frame=index,source_frame=row['source_frame'],visible=row['visible'],
                             translation_sample_max_error=maximum,translation_sample_mean_error=mean,translation_samples=samples,
                             center_x=None,center_y=None,center_drift_pixels=None,center_step_pixels=None,profile_step_pixels=None)
            if row['visible']:
                local=patch_at(visible,window,8)
                observed=moment(local)+window-8
                profile=moment(local,2)+window-8
                if initial_profile is None:initial_profile=profile.copy()
                observation.update(center_x=float(observed[0]-x),center_y=float(observed[1]-y),
                                   center_drift_pixels=float(np.linalg.norm(observed-anchor)),
                                   center_step_pixels=0. if previous is None else float(np.linalg.norm(observed-previous)),
                                   profile_step_pixels=0. if previous_profile is None else float(np.linalg.norm(profile-previous_profile)))
                previous,previous_profile=observed,profile
            audit.append(observation)
            progress(.76+.23*(index+1)/span,f'Checking encoded frame {index+1} of {span}')
    finally:original.close();decoded.close()
    if len(audit)!=span:raise ValueError('Export is missing frames from the selected range.')
    measured=[row for row in audit if row['visible']]
    drift=max(row['center_drift_pixels'] for row in measured);step=max(row['center_step_pixels'] for row in measured)
    profile_step=max(row['profile_step_pixels'] for row in measured)
    report=dict(engine=ENGINE,source=metadata,box=list(box),anchor=(anchor-[x,y]).tolist(),window=(window-[x,y]).tolist(),
                source_anchor=anchor.tolist(),crop_to_selection=bool(crop_to_selection),crop_box=list(crop_box) if crop_box else None,
                output_width=ow,output_height=oh,frames=span,output_frames=len(audit),extension='mp4',rows=rows,audit=audit,
                reference_frame=reference_frame,first_visible_frame=first,last_visible_frame=last,start_frame=begin,end_frame=end,
                source_frame_count=track['frame_count'],boundary_reasons=track['boundary_reasons'],
                max_residual_pixels=drift,max_step_pixels=step,unmeasurable_frames=0,
                center_lock_passed=drift<=.05 and step<=.05,
                profile_metrics={'profile_center_2_step_pixels':profile_step},
                zero_motion_passed=drift==0 and step==0 and profile_step==0,
                rigid_translation_samples_passed=all(row['translation_sample_max_error']<=4 for row in audit),
                geometry='Uniform translation only. The selected moving light is held at its reference-frame position; background motion is expected.',
                boundary_policy='One source frame before first detection and after last detection, when available. Padding frames hold adjacent correction; no light measurement is invented.',
                timing='Frames are encoded at the source nominal frame rate; audio is trimmed to the same frame range.',
                pixels_unchanged=False)
    destination.with_suffix('.json').write_text(json.dumps(report,indent=2))
    with destination.with_suffix('.csv').open('w') as stream:
        fields=list(rows[0])+[key for key in audit[0] if key not in rows[0]]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows({**r,**a} for r,a in zip(rows,audit))
    if not report['rigid_translation_samples_passed']:raise ValueError('The encoded moving-light export failed its rigid-translation check.')
    return report
