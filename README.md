# Object Stabilizer

A dependency-free, entirely browser-side video object stabilizer. Video files are opened through local object URLs; video pixels and tracking data are never uploaded.

Run `npm run dev`, then open http://localhost:8080. Run `npm test` for tracker checks.

Choose a video, seek to a clear frame, zoom in with the mouse wheel and click the object (or switch to box selection), and track. The selected frame is the start of the output clip. Tracking samples at 15, 24, or 30 fps and uses fixed-template normalized cross correlation at the original video resolution. Each rendered frame is translated to put the tracked selection center at the output center. Shift-drag pans the source frame, and source zoom reaches 16×. Point selection uses an adjustable patch centered on the exact clicked pixel. Small patches use an exhaustive pixel search to avoid skipping tiny targets. Output zoom reduces exposed black edges.

Export records the output canvas in real time using MediaRecorder, selecting a supported WebM or MP4 encoder. Export is silent and timing depends on browser rendering performance; it is not a deterministic frame-by-frame encoder. Keep the tab visible. Source decoding and export codecs depend on the browser. Tracking JSON includes analysis-space positions and dimensions for conversion to source coordinates.

This first version follows translation, not rotation or scale. Low-confidence matches hold the last known position and are flagged. Occlusion, uniform selections, large motion outside the search radius, and appearance changes can cause loss or drift. Reselect at a later frame to process a new segment. Long or high-resolution videos can take substantial time. No server, external runtime dependencies, or AI service is required.

## GitHub Pages

The app uses relative asset and module paths, so it works at a project URL such as `https://ufo-files.github.io/object-stabilizer/`. The `dist/` directory is the complete static site; no build or backend is needed.

To publish, choose **Settings → Pages → Source → GitHub Actions**, then run **Actions → Deploy to GitHub Pages → Run workflow**. Deployment is manual; pushing source does not publish automatically. Repository/account eligibility for Pages must be enabled in GitHub settings.
