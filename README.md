# Object Stabilizer

Local Python/OpenCV tracking with a browser interface for zooming, selecting a point or box, previewing, and exporting centered video. All video processing runs on your computer. The interface and export use the same zoom.

## Local setup

Requires Python 3.10 or newer and FFmpeg (including ffprobe) on your PATH. On macOS, `brew install ffmpeg` installs FFmpeg.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m backend.server
```

Open **http://127.0.0.1:8080**. On Windows, activate with `.venv\Scripts\activate` instead.

Choose a video, zoom at the target (wheel, double-click, or slider), click its center, and choose the target type:

- **Bright point / night footage** finds compact bright components after local background subtraction, measures a subpixel intensity centroid, and checks motion and brightness consistency. It snaps the initial click to the nearby bright object's center.
- **Textured object / optical flow** tracks local features with forward/backward checks and compares appearance with the original selected patch to reduce drift. Use a box large enough to contain several object features.

Tracking processes every decoded source frame at full resolution. The preview uses the displayed frame's presentation timestamp. Unreliable detections hold position and are flagged; only short, bounded gaps between good observations are interpolated. Confident centers are used directly, without a lagging camera smoother.

**Export video** writes an H.264 MP4 with FFmpeg, at the source nominal frame rate. Original timestamp gaps duplicate the corresponding source frame, preserving the source timeline. Source audio is included when present. Export can run faster or slower than playback and does not depend on browser recording. Translation places the measured target center at the output center; zoom is applied around that point, and exposed edges are black.

Uploaded files, tracks, and exports remain under the ignored `.runs/` directory until removed locally. The server binds only to `127.0.0.1`; stopping the terminal process stops it. After a restart, reload and upload the video again.

## Verification

```sh
python -m pip install -r requirements-dev.txt
python -m unittest discover -s backend/tests -v
npm test
python validation/validate_clip.py /path/to/02b43ebba81c502f57cdce5721308258.MP4
```

The Python tests cover subpixel moving targets, noise/exposure changes, competing bright points, disappearance, optical flow, centering transforms, and an end-to-end upload → track → MP4 download. JavaScript tests cover source zoom, selection coordinates, and matching preview positions to source timestamps.

[Sample report](validation/sample-report.json): the user-confirmed bright point in the first supplied clip was tracked across all 105 decoded frames without loss. At 8× zoom, the encoded output measured a mean centering error of 0.24 output pixels and a maximum of 1.77 pixels. Twelve visually reviewed annotations agreed within 0.54 source pixels (annotation uncertainty about ±1 pixel). The real-browser check also completed upload, tracking, playback, and MP4 download without console errors; preview center error averaged 0.47 output pixels and peaked at 1.10 pixels ([browser report](validation/browser-report.json)). The 107-frame, 15 fps export preserves the 7.133-second timeline, including the initial timestamp gap.

This is a short, nearly stationary target. It does not establish reliable tracking for every moving object or scene. Synthetic moving-target checks support the algorithm, but occlusion, crowded backgrounds, changes in object shape, and large movements can still lose the target. Inspect flagged frames, use a tighter selection, or begin a new segment when needed. Rotation and scale stabilization are not implemented.

Validation media is generated in ignored `validation/artifacts/`; raw user videos are not committed.

## GitHub Pages

`dist/` remains static and uses relative asset paths, so it is compatible with a GitHub Pages project URL. GitHub Pages cannot execute Python. The hosted interface directs users to run the local app; upload, tracking, and export work in the local interface served by `backend.server`.

The manual **Deploy to GitHub Pages** workflow publishes `dist/`. It does not deploy or expose the Python service. There is no `.openai` folder or ChatGPT Sites dependency.
