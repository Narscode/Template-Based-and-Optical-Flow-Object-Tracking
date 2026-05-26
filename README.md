# Template-Based and Optical Flow Object Tracking on Real Highway Footage

A research-style Computer Vision project comparing **Template Matching** and **Optical Flow** based
object-tracking pipelines on a real highway-traffic surveillance video (~17.8 MB, 1920&times;1080,
~24 fps). The full implementation, experiments, visualizations, benchmarks, and analysis live
inside a **single Jupyter notebook**: `Tracking_Project.ipynb`.

The notebook downloads the dataset automatically from Google Drive on the first run.

---

## Project Highlights

| Module                          | Methods Implemented                                                          |
|---------------------------------|------------------------------------------------------------------------------|
| Template Matching               | `TM_CCOEFF_NORMED`, `TM_SQDIFF_NORMED`, multi-scale pyramid (10 scales, 0.85&ndash;2.0&times;), confidence heat-maps, dual-metric agreement, loss-and-recovery |
| Sparse Optical Flow             | Lucas-Kanade pyramidal LK, Shi-Tomasi corners, dynamic re-seeding, trajectory drawing |
| Dense Optical Flow              | Farneb&auml;ck dense flow, HSV motion-field visualization, ROI motion summary |
| Filtering & Smoothing           | 1-D Kalman filter for bounding-box centroid stabilization                     |
| Evaluation                      | IoU vs. pseudo-ground-truth, centroid error, FPS, smoothing benefit           |
| Visualization                   | Multi-panel dashboard, trajectory overlays, comparison plots, metrics tables  |

### Dataset

The notebook downloads
[`tracking_video.mp4`](https://drive.google.com/file/d/1Y9qk2wQDtDW4xDScx9WZNM52ghFHweyC/view)
from Google Drive on first run. Three download strategies are tried in order:

1. **`gdown`** &mdash; the standard Python tool for Google Drive (auto-installed if missing).
2. **`wget`** with the explicit confirm-token trick.
3. **`curl`** with the same confirm-token trick.

If all three fail, the notebook prints clear instructions for manual download.

### Pseudo-Ground-Truth

The video has no manual annotations. The notebook generates a **pseudo-ground-truth** trajectory by
running an independent reference tracker once. The reference tracker uses a wider scale pyramid, a
spatial continuity prior, a dual-metric agreement check (`TM_CCOEFF_NORMED` vs.
`TM_SQDIFF_NORMED` must localise within 8 px), and a `0.70` score threshold. Frames where the
reference itself is uncertain are reported as `None` and excluded from the IoU / centroid-error
statistics. This is the same methodology used by semi-supervised tracking benchmarks (YouTube-BB,
OxUvA) when manual annotation is unavailable.

---

## Quick Start

```bash
# 1) Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate                # macOS/Linux
# .venv\Scripts\activate                 # Windows

# 2) Install dependencies (includes gdown for the dataset download)
pip install -r requirements.txt

# 3) Launch Jupyter and open the notebook
jupyter notebook Tracking_Project.ipynb
```

Run cells top-to-bottom. The notebook will:

1. Download `data/tracking_video.mp4` (~17.8 MB) from Google Drive if not present.
2. Pre-load 150 frames at 960&times;540 into memory and pick the red SUV at `(300, 180, 75, 70)` as the target.
3. Generate the pseudo-ground-truth trajectory.
4. Run all three trackers and write the side-by-side dashboard video.
5. Compute metrics, produce publication-style charts, and save every artefact under `./outputs/`.

```
outputs/
├── frames/                  # 5 sampled annotated dashboard frames
├── videos/                  # dashboard_highway.mp4 (2x2 panel video)
├── plots/                   # tracker_analysis, trajectory_comparison, summary_bars, ...
└── reports/                 # summary_metrics.csv + per_frame_metrics.csv
```

---

## Configuration

A single `CONFIG` dictionary near the top of the notebook controls every knob:

```python
CONFIG = {
    "gdrive_file_id":   "1Y9qk2wQDtDW4xDScx9WZNM52ghFHweyC",
    "dataset_path":     "data/tracking_video.mp4",
    "video_source":     "data/tracking_video.mp4",  # or "webcam", or any local path
    "max_frames":       150,
    "downscale":        0.5,                        # 1920x1080 -> 960x540
    "init_roi":         (300, 180, 75, 70),         # red SUV, mid-distance
    "template_scales":  (0.85, 0.93, 1.0, 1.08, 1.18, 1.30, 1.45, 1.62, 1.80, 2.00),
    "loss_threshold":   0.55,
    "kalman_smoothing": True,
    ...
}
```

Pointing at your own video and target is two lines &mdash; see the Appendix cell in the notebook.

---

## Notebook Structure

1. **Theory & Motivation** &mdash; concise primer on template matching and optical flow.
2. **Environment Setup** &mdash; imports, deterministic seed, output directory scaffolding.
3. **Dataset Acquisition** &mdash; gdown / wget / curl fallback chain for Google Drive.
4. **Pre-load & Inspect Frame 0** &mdash; load frames into memory, show the initial ROI.
5. **Reference Trajectory (Pseudo-GT)** &mdash; independent high-quality reference tracker.
6. **Video Source Manager** &mdash; uniform iterator over pre-loaded frames and webcam.
7. **Utility Components** &mdash; FPS meter, Kalman filter, IoU, drawing helpers.
8. **Template-Based Tracker** &mdash; multi-scale, dual-metric, with loss / recovery.
9. **Lucas-Kanade Sparse Tracker** &mdash; pyramidal LK + corner reseed + trajectories.
10. **Farneb&auml;ck Dense Flow Analyzer** &mdash; HSV motion visualization and ROI summary.
11. **Unified Tracking Pipeline** &mdash; runs all trackers and renders a 2x2 dashboard video.
12. **Quantitative Evaluation** &mdash; IoU, centroid error, TSR, FPS, Kalman benefit.
13. **Analysis Plots** &mdash; IoU/CE/FPS time-series, trajectory overlay, summary bars.
14. **Qualitative Strip** &mdash; per-frame side-by-side detection comparison.
15. **Discussion** &mdash; accuracy, illumination, scale, distractors, efficiency, application fit.
16. **Final Conclusions** &mdash; pros/cons and real-world application guidance.

---

## Measured Results (150 frames, real highway clip, 960&times;540)

| Tracker                       | Mean IoU | Median IoU | Mean CE (px) | TSR @ IoU&gt;0.5 | Mean FPS |
|-------------------------------|:--------:|:----------:|:------------:|:----------------:|:--------:|
| Template (multi-scale)        |  0.97    |  1.00      |  2.7         |  97 %            |  ~12     |
| Lucas-Kanade Sparse           |  0.55    |  0.54      | 10.6         |  50 %            |  ~12     |
| Farneb&auml;ck Dense (motion ROI) |    -     |    -       |     -        |    -             |  ~12     |

The 2&times; scale change of the approaching car is the dominant factor: multi-scale template
matching grows its bbox to match, whereas Lucas-Kanade can only translate its fixed-size box.

---

## Real-World Applications

- **Template Matching** &mdash; logo / sign detection, factory part location, document tracking.
- **Sparse Optical Flow** &mdash; visual-inertial odometry, drone navigation, SLAM front-ends.
- **Dense Optical Flow** &mdash; video stabilization, action recognition, crowd-flow analytics.

---

## Tested With

- Python 3.10
- OpenCV 4.8+ (`opencv-contrib-python`)
- NumPy 1.24+, Matplotlib 3.7+, Pandas 2.0+
- gdown 4.7+ (for the Google Drive download)

---

## License

Released under the MIT License for educational and research purposes.
