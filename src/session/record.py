"""Record a study session: attention and gaze logged to a CSV file.

Run directly (from the project root):

    python -m src.session.record                 # attention + gaze if calibrated
    python -m src.session.record --no-gaze       # attention only
    python -m src.session.record --output sessions/lecture1.csv

While recording, press **m** in the preview window to mark the start of a new
section (for example when the lecture moves to a new topic) and **q** to stop.
Analyse the result with `python -m src.session.analyse`.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from ..attention.attention_tracker import AttentionTracker
from ..gaze import gaze_calibration as gc
from ..gaze.gaze_tracker import DEFAULT_CALIBRATION_PATH, GazeTracker
from .session_logger import SessionLogger

OFF_SCREEN_MARGIN = 0.05  # gaze this far beyond the screen edge counts as off


def _loadMapper(calibrationPath):
    path = Path(calibrationPath)

    if not path.exists():
        print(f"No gaze calibration at {path}; recording attention only.")
        return None

    return gc.GazeMapper.load(path)


def _gazeSample(gazeTracker, mapper, smoother, frame):
    """Return (gaze point or None, off_screen flag) for one frame."""

    features = gazeTracker.process(frame)

    if features is None or features.blink:
        return None, False

    raw = smoother.update(mapper.predict(features.vector()))
    low, high = -OFF_SCREEN_MARGIN, 1 + OFF_SCREEN_MARGIN
    offScreen = bool(np.any(raw < low) or np.any(raw > high))

    return np.clip(raw, 0.0, 1.0), offScreen


def _drawStatus(frame, elapsed, result, gaze, markers):
    prediction = result["prediction"]
    label = prediction["attention"] if prediction else "unknown"
    gazeText = "gaze: n/a" if gaze is None else f"gaze: {gaze[0]:.2f}, {gaze[1]:.2f}"

    lines = [
        f"REC {int(elapsed // 60):02d}:{int(elapsed % 60):02d}  "
        f"section {markers + 1}",
        f"{label} | {gazeText}",
        "m = new section, q = stop"
    ]

    for index, text in enumerate(lines):
        cv2.putText(
            frame, text, (20, 75 + 28 * index),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )


def runSession(
    cameraIndex=0,
    calibrationPath=DEFAULT_CALIBRATION_PATH,
    outputPath=None,
    logHz=10.0,
    useGaze=True,
    smoothing=0.35,
    capture=None,
    show=True
):
    """Record until 'q' is pressed; returns the CSV path."""

    mapper = _loadMapper(calibrationPath) if useGaze else None
    ownsCapture = capture is None

    if ownsCapture:
        capture = cv2.VideoCapture(cameraIndex)

    if not capture.isOpened():
        raise RuntimeError("Error starting webcam")

    attentionTracker = AttentionTracker()
    gazeTracker = GazeTracker() if mapper is not None else None
    smoother = gc.GazeSmoother(smoothing)

    # The first inference is slow (model warm-up); keep it out of the timeline
    ret, frame = capture.read()
    if ret:
        attentionTracker.process(frame)
        if gazeTracker is not None:
            gazeTracker.process(frame)

    logger = SessionLogger(
        outputPath,
        metadata={
            "camera": cameraIndex,
            "gaze": mapper is not None,
            "screen": list(mapper.screenSize) if mapper and mapper.screenSize else None,
            "log_hz": logHz
        }
    )

    start = time.monotonic()
    lastLogged = -1e9
    frames = 0
    markers = 0
    pendingMarker = ""

    print(f"Recording to {logger.path}")

    try:
        while True:
            ret, frame = capture.read()

            if not ret:
                break

            elapsed = time.monotonic() - start
            frames += 1

            result = attentionTracker.process(frame)
            gaze, offScreen = (None, False)

            if gazeTracker is not None:
                gaze, offScreen = _gazeSample(
                    gazeTracker, mapper, smoother, frame
                )

            if pendingMarker or elapsed - lastLogged >= 1.0 / logHz:
                features = result["features"] or {}
                face = features.get("face") or {}
                prediction = result["prediction"]

                logger.log(
                    t=elapsed,
                    attention=prediction["label"] if prediction else None,
                    faces=features.get("no_of_face", 0),
                    hands=features.get("no_of_hand", 0),
                    phone="phone" in features,
                    pose=(face.get("pose") or {}).get("pose"),
                    gaze=gaze,
                    offScreen=offScreen,
                    marker=pendingMarker
                )
                lastLogged = elapsed
                pendingMarker = ""

            if show:
                attentionTracker.annotate(frame, result)
                _drawStatus(frame, elapsed, result, gaze, markers)
                cv2.imshow("Session recording", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("m"):
                markers += 1
                pendingMarker = f"section {markers + 1}"
                print(f"{elapsed:7.1f}s  marker: {pendingMarker}")
    finally:
        duration = time.monotonic() - start
        logger.finish(
            duration_s=round(duration, 2),
            frames_processed=frames,
            processing_fps=round(frames / duration, 2) if duration else None,
            markers=markers
        )
        logger.close()
        attentionTracker.close()
        if gazeTracker is not None:
            gazeTracker.close()
        if ownsCapture:
            capture.release()
        if show:
            cv2.destroyAllWindows()

    fps = frames / duration if duration else 0
    print(
        f"Saved {logger.rows} rows ({duration:.0f} s, {fps:.1f} fps "
        f"processing) to {logger.path}"
    )
    return logger.path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Record attention and gaze during a study session."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None,
                        help="CSV path (default: sessions/session_<time>.csv)")
    parser.add_argument("--calibration", type=Path,
                        default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--no-gaze", action="store_true",
                        help="Record attention only")
    parser.add_argument("--log-hz", type=float, default=10.0,
                        help="Rows written per second (default: 10)")
    parser.add_argument("--smoothing", type=float, default=0.35)
    args = parser.parse_args(argv)

    runSession(
        cameraIndex=args.camera,
        calibrationPath=args.calibration,
        outputPath=args.output,
        logHz=args.log_hz,
        useGaze=not args.no_gaze,
        smoothing=args.smoothing
    )


if __name__ == "__main__":
    main()
