"""Check native source/export landmarks and make fixed-position review crops.

This is an independent inspection script, not the tracker's acceptance metric.
No OpenCV frame seeking is used: every source and output frame is decoded in order.
"""
import argparse
import csv
import json
from itertools import zip_longest
from pathlib import Path

import cv2
import numpy as np

from backend.point_lock import yuv_frames


def inspect(source, export, manifest, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads(Path(manifest).read_text())
    width = report['source']['width']; height = report['source']['height']
    width += width % 2; height += height % 2
    ax, ay = report['anchor']
    pairs = [(34, 35), (38, 39), (186, 187), (254, 255), (282, 283), (286, 287)]
    wanted = {frame for pair in pairs for frame in pair}
    crops = {}; rows = []
    source_frames = yuv_frames(source, width, height)
    export_frames = yuv_frames(export, width, height)
    try:
        for index, (original, corrected) in enumerate(zip_longest(source_frames, export_frames)):
            if original is None or corrected is None:
                raise ValueError('Frame counts differ.')
            before, after = original[0], corrected[0]
            search = before[ay - 48:ay + 49, ax - 48:ax + 49]
            py, px = np.unravel_index(np.argmax(search), search.shape)
            px += ax - 48; py += ay - 48
            output_search = after[ay - 48:ay + 49, ax - 48:ax + 49]
            oy, ox = np.unravel_index(np.argmax(output_search), output_search.shape)
            ox += ax - 48; oy += ay - 48
            # These two native crops should be bit-for-bit equal. Each follows
            # only the already measured integer peak, without resampling.
            source_patch = before[py - 32:py + 33, px - 32:px + 33]
            exported_patch = after[ay - 32:ay + 33, ax - 32:ax + 33]
            rows.append(dict(frame=index, source_x=int(px), source_y=int(py),
                             output_x=int(ox), output_y=int(oy),
                             peak_aligned_source_patch_unchanged=bool(np.array_equal(source_patch, exported_patch)),
                             peak_error=float(np.hypot(ox - ax, oy - ay))))
            if index in wanted:
                crops[index] = (before[ay - 32:ay + 33, ax - 32:ax + 33].copy(), exported_patch.copy())
    finally:
        source_frames.close(); export_frames.close()
    with (output / 'all-frames.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    panels = []
    for first, second in pairs:
        tiles = []
        for column, label in [(0, 'SOURCE'), (1, 'EXPORT')]:
            for index in (first, second):
                tile = cv2.cvtColor(cv2.resize(crops[index][column], (260, 260), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
                cv2.putText(tile, f'{label} frame {index}', (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 0), 1)
                for start, end in [((114, 130), (124, 130)), ((136, 130), (146, 130)), ((130, 114), (130, 124)), ((130, 136), (130, 146))]:
                    cv2.line(tile, start, end, (0, 180, 0), 1)
                tiles.append(tile)
        panels.append(np.concatenate(tiles, axis=1))
    cv2.imwrite(str(output / 'source-and-export.png'), np.concatenate(panels, axis=0))
    summary = dict(
        decoded_frames=len(rows), anchor=[ax, ay],
        maximum_native_peak_error=max(row['peak_error'] for row in rows),
        original_peak_aligned_patches_preserved=all(row['peak_aligned_source_patch_unchanged'] for row in rows),
        method='Sequential native-luma decoding; direct source/output peak coordinates and exact 65x65 patch equality. No frame seeking, geometric fitting, or recentering of review exports.',
        limitation='A fixed maximum-intensity pixel does not certify a fixed entire light distribution. Differences between exported peak-aligned patches are also present in the original peak-aligned patches.',
    )
    (output / 'report.json').write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('source', 'export', 'manifest', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.source, args.export, args.manifest, args.output), indent=2))
