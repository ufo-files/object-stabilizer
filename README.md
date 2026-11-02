# Tripod scene stabilizer

Automatic **scene stabilization** is the default. Load the video and run it; drawing a box is unnecessary. The app estimates camera motion from background references distributed across the image. A separate selected-light mode remains available for experiments. General handheld subject tracking is deferred.

```sh
npm run dev
```

The preview and download use the same encoded MP4. Output retains the full frame, its original framing at frame zero, and rigid geometry. Uncovered edges are black. Encoding is H.264 QP 1, near-lossless 8-bit 4:2:0, with AAC audio when present. Fractional resampling and encoding change pixel values; this is not a lossless archival export. Existing source uploads and previous exports are preserved.

In selected-light mode, scrub to the clearest or brightest frame and draw a box around one moving light. Tracking runs backward and forward from that reference frame, stopping when the light is no longer detected or matching becomes ambiguous. The export includes one extra source frame at each end when available. The selected light is held at its reference position using uniform translation; the background moves accordingly. UI frame numbers start at 1; API/report frame indices start at 0.

Selected-light mode includes **Crop to selection**, off by default. When enabled, the export uses the box drawn on the selected reference frame after stabilization, preserving the light's position within that box. Tracking and translations are identical with the option on or off. An odd crop width/height receives one black padding pixel on the right/bottom for encoding. Preview, download, and the encoded-frame audit all use the cropped result. Automatic scene mode retains the full frame.

After rendering, the result has one **Download ZIP** button. The app prepares the full bundle in the background and downloads it when ready. The same button downloads the cached ZIP again on subsequent clicks. There are no export settings to choose.

The ZIP contains:

- `original/`: the unchanged uploaded video.
- `stabilized/`: the exact MP4 shown in the preview, its JSON audit, and motion CSV.
- `ai/mild/`, `ai/standard/`, `ai/strong/`: separate Real-ESRGAN videos at denoising strengths 0.25, 0.50, and 0.75, each with its model/settings report.
- `frames/stabilized/` and `frames/ai-{preset}/`: exactly one native PNG per decoded video frame, with CSV timestamps and source-frame indices. Each native set matches its associated video pixels.
- `frames/stabilized-viewing-1920/`: additional enlarged viewing copies when the stabilized selection is small; native frames remain separately available.
- `manifest.json` and `README.txt`: provenance, original/stabilized SHA-256 hashes, preset definitions, and enlargement details.

AI copies use the official general-purpose `realesr-general-x4v3` model with weak/strong-denoise checkpoint interpolation. Inference produces a 2× result; small crops receive additional Lanczos enlargement to a minimum 1920-pixel long edge for viewing. That additional enlargement adds no captured detail. The original and stabilized MP4 remain unchanged. AI processing is framewise, preserves every stabilized frame and timestamp, copies the stabilized audio without re-encoding, and can alter appearance or apparent motion. The original stabilization audit does not certify AI detail or zero residual motion. Each selected-light AI report also records independently measured enhanced-center movement and unmeasurable frames; enhancement does not inherit a zero-motion pass from its parent. PNGs are always extracted from their final encoded video.

Install the optional AI runtime and verified official weights locally:

```sh
.venv/bin/python -m pip install -r requirements-enhancement.txt
.venv/bin/python scripts/setup_enhancement.py
```

Apple Silicon uses MPS when available; other environments use CPU. Both official checkpoint SHA-256 values are pinned in `backend/enhancement.py`. Upstream architecture attribution and the BSD license are retained in `backend/vendor/Real-ESRGAN-LICENSE`.

Complete bundles can be large. `STABILIZER_EXPORT_ROOT` or the local, ignored `.export-storage.json` selects the bundle cache. This installation uses `/Volumes/UFO Files Archive 2/.stabilizer-exports` so archives do not fill the internal drive. Disconnecting that configured drive produces a retryable error rather than falling back to internal storage. Completed AI derivatives are reused when retrying packaging. Original uploads and stabilized exports retain their existing locations.

## Processing

`backend/server.py` dispatches automatic mode to `backend/scene_pipeline.py` and selected-light mode to `backend/light_segment.py`. Older API requests without temporal tracking parameters retain the legacy `backend/light_center.py` path.

1. `backend/scene_motion.py` examines the first frame and chooses reference detection automatically. Sparse light fields use isolated, background-subtracted light profiles. Detailed scenes use image features distributed across a grid, with separate features reserved for verification.
2. Every source frame is decoded sequentially in display orientation. Sparse references use flux centroids; textured references use initial-frame optical flow with forward/backward checks. Robust fitting rejects inconsistent foreground motion. Camera estimates refer to frame zero, avoiding accumulated tracking drift.
3. Held-out references compare translation against rigid rotation/translation. One model is selected for the entire video. Rotation must improve prediction substantially; scale, shear, local warping, and frame-by-frame switching of models are excluded.
4. Each original frame receives its camera correction once. Translation uses precise fractional cubic sampling. Rotation uses a rigid matrix and Lanczos sampling. Chroma follows the same geometry at half resolution.
5. The actual encoded output is decoded again. Every frame is checked against source-coordinate samples across the image. Independent image-patch registration measures residual background motion and records disagreements or frames it cannot verify. A geometry-check failure rejects the export.

The current tripod range is bounded to small camera movements (3 degrees and 15% of the shorter image dimension). Frames retain source order at the nominal frame rate; variable frame timing is not preserved. Metadata pads odd dimensions on the right/bottom by one pixel.

## Verification

```sh
.venv/bin/python -m unittest discover -s backend/tests -q
npm test
node --check dist/app.js
node validation/qa_scene_browser.cjs
node validation/qa_bundle_exports.cjs
```

The scene tests use independently generated camera translations/rotations and an unrelated moving foreground. They verify automatic reference selection, static-scene identity, foreground rejection, model selection, source geometry, and the encoded video. Older algorithm tests remain in discovery and do not establish correctness of the default scene pipeline.

The browser QA uploads the real IR fixture, runs automatic mode without a selection, downloads the actual export, checks normal playback, then reads every displayed frame and every detected light at quarter speed. Any missing frame is inspected after a timestamp-verified video-frame callback. It preserves the observations and then enforces the independent zero-motion gate. `--diagnostics` omits only that final acceptance assertion; it does not make a failed result accepted.

## Real IR fixture: acceptance remains unresolved

Source: `/Volumes/UFO Files Archive 2/ir/4E1A78FF-32B3-48D4-9399-7E77B2A45BB1.mp4`.

The automatic detector finds four strong scene references. Held-out prediction favors translation over rotation. The source decodes to **349 frames**, and the exported file retains all 349. Source-coordinate sample checks pass, with a maximum 2-level luma difference in the initial automatic render.

**The real clip has not passed zero-motion QA.** The initial automatic render's independent patch checks still report residual movement and 16 frames without a consistent verified background fit. Those failures remain visible in the UI/report and must not be described as a stabilization fix. The final browser observations are saved in `validation/artifacts/auto-scene-browser-qa.json`; frame-by-frame native measurements are in `auto-scene-browser-report.json`.

The earlier manual brightest-pixel implementation was rejected by the user. Holding one maximum at a fixed coordinate left the surrounding light center moving by 2.77 px. The subsequent selected-light centroid renderer reduces its native bright-center drift to about 0.029 px, but independent profile checks still detect movement. Neither narrow diagnostic certifies whole-scene stability.

A separate whole-reference restoration experiment was rejected because it changed recorded detail and created ringing around other lights and the moving object. It is not in the automatic or manual processing paths. Its review page records that rejection.

Diagnostic artifacts under `validation/artifacts/` and generated review media under `dist/review-*` are local outputs, excluded from version control. They preserve failed candidates as well as measurements; their existence is not evidence of acceptance.
