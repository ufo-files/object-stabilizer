"""Automatic scene references and one rigid camera pose per source frame.

Sparse light fields use background-subtracted flux centroids. Detailed scenes
use distributed image features with forward/backward optical-flow checks.
Reference selection is automatic; no chosen subject steers camera estimation.
"""
from itertools import combinations
import cv2
import numpy as np

from .light_center import profile_observations
from .point_lock import yuv_frames
from .tripod import info


def distributed_features(gray):
    points = cv2.goodFeaturesToTrack(gray, maxCorners=1600, qualityLevel=.01,
                                     minDistance=12, blockSize=7)
    if points is None:
        return np.empty((0, 2), np.float32)
    height, width = gray.shape
    counts = {}
    selected = []
    for x, y in points.reshape(-1, 2):
        if min(x, y, width - x - 1, height - y - 1) < 35:
            continue
        cell = (int(x * 4 / width), int(y * 3 / height))
        if counts.get(cell, 0) >= 12:
            continue
        counts[cell] = counts.get(cell, 0) + 1
        selected.append((x, y))
    return np.asarray(selected, np.float32).reshape(-1, 2)


def compact_lights(gray):
    background = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 8)
    signal = np.maximum(gray.astype(np.float32) - background, 0)
    residual = gray.astype(np.float32) - background
    noise = max(1., 1.4826 * float(np.median(abs(residual - np.median(residual)))))
    mask = (signal > max(16., 6 * noise)).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    energies = np.bincount(labels.ravel(), weights=signal.ravel())
    candidates = []
    height, width = gray.shape
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        cx, cy = centers[label]
        if not (4 <= area <= 500 and max(w, h) <= 35):
            continue
        if min(cx, cy, width - cx - 1, height - cy - 1) < 49:
            continue
        candidates.append((float(energies[label]), (cx, cy)))
    if not candidates:
        return np.empty((0, 2), np.float64)
    candidates.sort(reverse=True)
    cutoff = max(120., candidates[0][0] * .03)
    selected = []
    for energy, point in candidates:
        if energy < cutoff or len(selected) >= 40:
            continue
        if selected and np.min(np.linalg.norm(np.asarray(selected) - point, axis=1)) < 48:
            continue
        window = np.rint(point).astype(int)
        try:
            point = profile_observations(gray, window)['profile_center_1']
        except ValueError:
            continue
        selected.append(point)
    return np.asarray(selected, np.float64).reshape(-1, 2)


def inspect_frame(gray):
    features = distributed_features(gray)
    lights = compact_lights(gray)
    if len(features) < 25 and len(lights) >= 3:
        method, points, audit_points = 'sparse_lights', lights, lights
    elif len(features) >= 6:
        method = 'scene_features'
        points = features[::2] if len(features) >= 12 else features
        audit_points = features[1::2] if len(features) >= 12 else features
    else:
        method, points, audit_points = 'insufficient_detail', features, features
    return dict(method=method, points=points.tolist(), audit_points=audit_points.tolist(), reference_count=len(points),
                ready=method != 'insufficient_detail')


def inspect(source):
    metadata = info(source)
    width = metadata['width'] + metadata['width'] % 2
    height = metadata['height'] + metadata['height'] % 2
    decoder = yuv_frames(source, width, height)
    try:
        return inspect_frame(next(decoder)[0])
    finally:
        decoder.close()


def rigid_fit(reference, current, weights=None, rotation=True):
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    if weights is None:
        weights = np.ones(len(reference))
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    first_mean = weights @ reference
    current_mean = weights @ current
    matrix = np.eye(3)
    if rotation:
        covariance = ((reference - first_mean) * weights[:, None]).T @ (current - current_mean)
        left, _, right = np.linalg.svd(covariance)
        orientation = right.T @ left.T
        if np.linalg.det(orientation) < 0:
            right[-1] *= -1
            orientation = right.T @ left.T
        matrix[:2, :2] = orientation
    matrix[:2, 2] = current_mean - matrix[:2, :2] @ first_mean
    return matrix


def project(points, matrix):
    return np.asarray(points) @ matrix[:2, :2].T + matrix[:2, 2]


def robust_pose(reference, current, width, height, rotation=False, tolerance=1.5):
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    valid = np.isfinite(current).all(axis=1)
    indices = np.flatnonzero(valid)
    if len(indices) < 3:
        raise ValueError('Too few background references remain observable.')
    first, observed = reference[valid], current[valid]
    hypotheses = []
    if rotation:
        pairs = list(combinations(range(len(first)), 2))
        if len(pairs) > 128:
            order = np.random.default_rng(0).choice(len(pairs), 128, replace=False)
            pairs = [pairs[index] for index in order]
        for a, b in pairs:
            if np.linalg.norm(first[a] - first[b]) < min(width, height) * .1:
                continue
            hypotheses.append(rigid_fit(first[[a, b]], observed[[a, b]], rotation=True))
    else:
        differences = observed - first
        candidates = list(differences)
        candidates += [np.median(differences, axis=0), np.mean(differences, axis=0)]
        if len(first) <= 8:
            candidates += [(differences[a] + differences[b]) / 2
                           for a, b in combinations(range(len(first)), 2)]
        for delta in candidates:
            pose = np.eye(3)
            pose[:2, 2] = delta
            hypotheses.append(pose)
    if not hypotheses:
        raise ValueError('Scene references do not span enough of the image.')
    cells = np.column_stack((np.minimum(3, (first[:, 0] * 4 / width).astype(int)),
                             np.minimum(2, (first[:, 1] * 3 / height).astype(int))))
    scored = []
    for pose in hypotheses:
        residuals = np.linalg.norm(project(first, pose) - observed, axis=1)
        inliers = residuals <= tolerance
        coverage = len(np.unique(cells[inliers], axis=0))
        scored.append(((coverage, int(inliers.sum()), -float(np.median(residuals[inliers]))
                        if inliers.any() else -float('inf')), pose, inliers))
    _, pose, inliers = max(scored, key=lambda item: item[0])
    if inliers.sum() < 3:
        raise ValueError('No consistent background motion was found.')
    for _ in range(4):
        residuals = np.linalg.norm(project(first[inliers], pose) - observed[inliers], axis=1)
        weights = np.minimum(1., .5 / np.maximum(residuals, 1e-9))
        refined_pose = rigid_fit(first[inliers], observed[inliers], weights, rotation=rotation)
        residuals = np.linalg.norm(project(first, refined_pose) - observed, axis=1)
        next_inliers = residuals <= tolerance
        if next_inliers.sum() < 3:
            break
        pose = refined_pose
        if np.array_equal(next_inliers, inliers):
            break
        inliers = next_inliers
    mask = np.zeros(len(reference), bool)
    mask[indices[inliers]] = True
    residuals = np.linalg.norm(project(reference[mask], pose) - current[mask], axis=1)
    return pose, mask, float(np.sqrt(np.mean(residuals**2)))


def cross_validation(reference, observations, width, height, tolerance):
    """Choose one camera model for the entire video, using held-out points.

    Rotation is accepted only when it predicts omitted references substantially
    better. This prevents frame-by-frame model switching and noisy rotations.
    """
    errors = {'translation': [], 'rigid': []}
    frame_indices = np.unique(np.linspace(0, len(observations) - 1, min(80, len(observations))).astype(int))
    for frame_index in frame_indices:
        current = observations[frame_index]
        valid = np.flatnonzero(np.isfinite(current).all(axis=1))
        if len(valid) < 4:
            continue
        if len(valid) <= 8:
            splits = [(valid[valid != held], np.array([held])) for held in valid]
        else:
            splits = [(valid[::2], valid[1::2]), (valid[1::2], valid[::2])]
        for train, held in splits:
            for kind, rotation in [('translation', False), ('rigid', True)]:
                try:
                    pose, _, _ = robust_pose(reference[train], current[train], width, height,
                                             rotation=rotation, tolerance=tolerance)
                except ValueError:
                    # Score failed consensus fits too. Dropping their hard
                    # validation folds would make model comparison optimistic.
                    pose = rigid_fit(reference[train], current[train], rotation=rotation)
                residuals = np.linalg.norm(project(reference[held], pose) - current[held], axis=1)
                errors[kind].extend(residuals.tolist())
    costs = {key: float(np.median(values)) if values else None for key, values in errors.items()}
    use_rotation = (costs['rigid'] is not None and costs['translation'] is not None
                    and costs['rigid'] < costs['translation'] * .8)
    return 'rigid' if use_rotation else 'translation', costs


class SceneTracker:
    def __init__(self, first, analysis):
        self.first = first
        self.method = analysis['method']
        self.reference = np.asarray(analysis['points'], np.float64)
        if not analysis['ready']:
            raise ValueError('The scene has too little fixed detail for automatic tripod stabilization.')
        self.previous = self.reference.astype(np.float32).reshape(-1, 1, 2)
        self.templates = []
        if self.method == 'sparse_lights':
            for x, y in np.rint(self.reference).astype(int):
                self.templates.append(first[y - 16:y + 17, x - 16:x + 17].copy())

    def locate(self, current):
        if np.array_equal(current, self.first):
            return self.reference.copy()
        if self.method == 'sparse_lights':
            found = np.full_like(self.reference, np.nan)
            height, width = current.shape
            for index, (at, template) in enumerate(zip(self.reference, self.templates)):
                x, y = np.rint(at).astype(int)
                x0, y0 = max(0, x - 64), max(0, y - 64)
                region = current[y0:min(height, y + 65), x0:min(width, x + 65)]
                matches = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
                _, score, _, offset = cv2.minMaxLoc(matches)
                if score < .3:
                    continue
                coarse = np.array([x0 + offset[0] + 16, y0 + offset[1] + 16])
                if min(*coarse, width - coarse[0] - 1, height - coarse[1] - 1) < 33:
                    continue
                try:
                    found[index] = profile_observations(current, coarse)['profile_center_1']
                except ValueError:
                    continue
            return found
        points = self.reference.astype(np.float32).reshape(-1, 1, 2)
        predicted = self.previous.copy()
        if not np.isfinite(predicted).all():
            predicted = points.copy()
        found, status, error = cv2.calcOpticalFlowPyrLK(
            self.first, current, points, predicted,
            winSize=(31, 31), maxLevel=4, criteria=(3, 60, .001),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW, minEigThreshold=.0001)
        backward, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
            current, self.first, found, points.copy(),
            winSize=(31, 31), maxLevel=4, criteria=(3, 60, .001),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW, minEigThreshold=.0001)
        valid = (status.ravel().astype(bool) & reverse_status.ravel().astype(bool)
                 & (np.linalg.norm(backward.reshape(-1, 2) - self.reference, axis=1) < .75))
        if valid.any():
            threshold = max(15., float(np.median(error[valid])) * 3)
            valid &= error.ravel() <= threshold
        self.previous = found.copy()
        result = found.reshape(-1, 2).astype(np.float64)
        result[~valid] = np.nan
        return result


def measure(source, progress=lambda *_: None):
    metadata = info(source)
    width, height = metadata['width'] + metadata['width'] % 2, metadata['height'] + metadata['height'] % 2
    decoder = yuv_frames(source, width, height)
    observations = []
    try:
        first = next(decoder)[0]
        analysis = inspect_frame(first)
        tracker = SceneTracker(first, analysis)
        observations.append(tracker.reference.copy())
        for index, planes in enumerate(decoder, 1):
            observations.append(tracker.locate(planes[0]))
            progress(min(.4, .4 * (index + 1) / metadata['fps'] / metadata['duration']),
                     f'Measuring background motion in frame {index + 1}')
    finally:
        decoder.close()
    tolerance = 1.5 if analysis['method'] == 'sparse_lights' else .9
    model, costs = cross_validation(tracker.reference, observations, width, height, tolerance)
    rows = []
    for index, current in enumerate(observations):
        pose, inliers, error = robust_pose(tracker.reference, current, width, height,
                                           rotation=model == 'rigid', tolerance=tolerance)
        if index == 0:
            pose = np.eye(3)
        angle = float(np.degrees(np.arctan2(pose[1, 0], pose[0, 0])))
        if abs(angle) > 3 or np.linalg.norm(pose[:2, 2]) > min(width, height) * .15:
            raise ValueError('Camera movement exceeds the current tripod mode range.')
        rows.append(dict(frame=index, pose=pose.tolist(), angle_degrees=angle,
                         inliers=inliers.tolist(), reference_rms_pixels=error,
                         observed_points=[[float(v) if np.isfinite(v) else None for v in point] for point in current]))
    return dict(source=metadata, width=width, height=height, analysis=analysis,
                motion_model=model, model_validation_costs=costs, rows=rows)
