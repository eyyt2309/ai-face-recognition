"""Webcam gaze tracking.

Estimates where a student is looking from the iris position inside each eye,
the head pose, and the face position, using MediaPipe's face landmarker.
Features are mapped to screen coordinates by a calibrated GazeMapper.

Usage as a module:

    from src.gaze.gaze_tracker import GazeTracker
    from src.gaze.gaze_calibration import GazeMapper

    mapper = GazeMapper.load("models/gaze/gaze_calibration.npz")
    with GazeTracker() as tracker:
        features = tracker.process(frame)
        if features is not None and not features.blink:
            x, y = mapper.predict(features.vector())  # normalised 0..1

Run directly to test with the default webcam:

    python -m src.gaze.gaze_tracker                    # raw eye features, no screen
    python -m src.gaze.gaze_tracker --mode calibrate   # calibrate, save, then track
    python -m src.gaze.gaze_tracker --mode track       # track with saved calibration
    python -m src.gaze.gaze_tracker --mode validate    # measure accuracy in pixels
"""

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv

from ..config import PROJECT_ROOT, resolveModelPath
from . import gaze_calibration as gc

DEFAULT_CALIBRATION_PATH = (
    PROJECT_ROOT / "models" / "gaze" / "gaze_calibration.npz"
)

# Face landmarker indices. Each eye is (left corner, right corner, upper lid,
# lower lid, iris centre), with corners ordered by image x so a positive
# horizontal offset means "iris towards the image right" for both eyes.
RIGHT_EYE = (33, 133, 159, 145, 468)
LEFT_EYE = (362, 263, 386, 374, 473)
NOSE_TIP = 1

FEATURE_NAMES = ["h", "v", "yaw", "pitch", "face_x", "face_y"]


@dataclass
class GazeFeatures:
    """Gaze-related measurements for one frame.

    h, v:      mean iris offset from the eye centre, in eye widths, measured
               along / across the eye axis (so it is roll-invariant)
    yaw/pitch: head rotation in degrees (sign convention is arbitrary but
               consistent, which is all the calibration needs)
    face_x/y:  nose position as a fraction of the frame
    openness:  mean eyelid gap relative to eye width
    blink:     True when the eyes are (nearly) closed; iris data is unreliable
    landmarks: all landmarks in pixel coordinates, for drawing
    """

    h: float
    v: float
    yaw: float
    pitch: float
    face_x: float
    face_y: float
    openness: float
    blink: bool
    landmarks: np.ndarray

    def vector(self):
        return np.array(
            [self.h, self.v, self.yaw, self.pitch, self.face_x, self.face_y],
            dtype=np.float64
        )


def _eyeMetrics(points, eye):
    left, right, upper, lower, iris = eye

    axis = points[right] - points[left]
    width = np.linalg.norm(axis)

    if width < 1e-6:
        raise ValueError("Degenerate eye")

    along = axis / width
    across = np.array([-along[1], along[0]])  # points down in image space

    offset = points[iris] - (points[left] + points[right]) / 2
    openness = np.linalg.norm(points[lower] - points[upper]) / width

    return (
        float(offset @ along / width),
        float(offset @ across / width),
        float(openness)
    )


def _headPose(matrix):
    """Yaw and pitch in degrees from the face transformation matrix."""

    rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
    rotation = rotation / np.linalg.norm(rotation, axis=0)  # remove scale

    pitch = np.arctan2(rotation[2, 1], rotation[2, 2])
    yaw = np.arctan2(
        -rotation[2, 0],
        np.hypot(rotation[2, 1], rotation[2, 2])
    )

    return float(np.degrees(yaw)), float(np.degrees(pitch))


class GazeTracker:
    """Extracts gaze features from webcam frames.

    The landmarker model defaults to LANDMARKER_MODEL_PATH in the project .env.
    """

    def __init__(self, landmarkerModelPath=None, blinkThreshold=0.15):
        load_dotenv(PROJECT_ROOT / ".env")

        modelPath = resolveModelPath(
            landmarkerModelPath or os.getenv("LANDMARKER_MODEL_PATH"),
            "LANDMARKER_MODEL_PATH"
        )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(modelPath)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True
        )

        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            options
        )
        self.blinkThreshold = blinkThreshold
        self.startTime = time.monotonic()
        self.lastTimeStamp = -1

    def process(self, frame):
        """Return GazeFeatures for a BGR frame, or None if no face is found."""

        # MediaPipe video mode requires strictly increasing timestamps
        timeStamp = int((time.monotonic() - self.startTime) * 1000)
        timeStamp = max(timeStamp, self.lastTimeStamp + 1)
        self.lastTimeStamp = timeStamp

        height, width = frame.shape[:2]

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        result = self.landmarker.detect_for_video(image, timeStamp)

        if not result.face_landmarks or not result.facial_transformation_matrixes:
            return None

        landmarks = result.face_landmarks[0]

        if len(landmarks) <= max(RIGHT_EYE + LEFT_EYE):
            return None  # model without iris landmarks

        # Pixel coordinates keep eye geometry undistorted by the aspect ratio
        points = np.array(
            [(p.x * width, p.y * height) for p in landmarks],
            dtype=np.float64
        )

        try:
            rightH, rightV, rightOpen = _eyeMetrics(points, RIGHT_EYE)
            leftH, leftV, leftOpen = _eyeMetrics(points, LEFT_EYE)
        except ValueError:
            return None

        yaw, pitch = _headPose(result.facial_transformation_matrixes[0])
        openness = (rightOpen + leftOpen) / 2

        return GazeFeatures(
            h=(rightH + leftH) / 2,
            v=(rightV + leftV) / 2,
            yaw=yaw,
            pitch=pitch,
            face_x=float(landmarks[NOSE_TIP].x),
            face_y=float(landmarks[NOSE_TIP].y),
            openness=float(openness),
            blink=bool(openness < self.blinkThreshold),
            landmarks=points
        )

    def annotate(self, frame, features):
        """Draw the eye landmarks and feature values onto the frame."""

        if features is None:
            cv2.putText(
                frame, "No face", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )
            return frame

        for eye in (RIGHT_EYE, LEFT_EYE):
            for index in eye[:4]:
                cv2.circle(
                    frame,
                    tuple(features.landmarks[index].astype(int)),
                    2, (0, 255, 255), -1
                )
            cv2.circle(
                frame,
                tuple(features.landmarks[eye[4]].astype(int)),
                3, (0, 255, 0), -1
            )

        color = (0, 0, 255) if features.blink else (0, 255, 0)
        lines = [
            f"h {features.h:+.3f}  v {features.v:+.3f}",
            f"yaw {features.yaw:+.1f}  pitch {features.pitch:+.1f}",
            f"openness {features.openness:.2f}"
            + ("  BLINK" if features.blink else ""),
        ]

        for i, line in enumerate(lines):
            cv2.putText(
                frame, line, (20, 40 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

        return frame

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, excType, excValue, traceback):
        self.close()


def _openCamera(cameraIndex):
    capture = cv2.VideoCapture(cameraIndex)

    if not capture.isOpened():
        raise RuntimeError("Error starting webcam")

    return capture


def runPreview(cameraIndex=0):
    """Show the raw eye features on the webcam feed (no calibration)."""

    capture = _openCamera(cameraIndex)

    try:
        with GazeTracker() as tracker:
            while True:
                ret, frame = capture.read()

                if not ret:
                    break

                tracker.annotate(frame, tracker.process(frame))
                cv2.imshow("Gaze Preview", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def runTracking(
    mapper,
    cameraIndex=0,
    screenSize=None,
    smoothing=0.35,
    tracker=None,
    capture=None,
    windowName="Gaze"
):
    """Show the estimated gaze point on a full-screen canvas until q/Esc."""

    screenSize = screenSize or mapper.screenSize or gc.getScreenSize()
    width, height = screenSize

    ownsCapture = capture is None
    capture = capture or _openCamera(cameraIndex)
    ownsTracker = tracker is None
    tracker = tracker or GazeTracker()

    smoother = gc.GazeSmoother(smoothing)
    gaze = None

    if ownsCapture:
        gc.openFullscreen(windowName)

    try:
        while True:
            ret, frame = capture.read()

            if not ret:
                break

            features = tracker.process(frame)

            if features is not None and not features.blink:
                point = mapper.predict(features.vector())
                gaze = np.clip(smoother.update(point), 0.0, 1.0)

            canvas = np.zeros((height, width, 3), dtype=np.uint8)

            if gaze is not None:
                center = (int(gaze[0] * width), int(gaze[1] * height))
                color = (0, 255, 0) if features is not None else (90, 90, 90)
                cv2.circle(canvas, center, 18, color, -1)

            thumbnail = cv2.resize(
                tracker.annotate(frame, features), (320, 240)
            )
            canvas[10:250, width - 330:width - 10] = thumbnail

            cv2.imshow(windowName, canvas)

            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        if ownsTracker:
            tracker.close()
        if ownsCapture:
            capture.release()
            cv2.destroyAllWindows()


def _parseScreen(value):
    try:
        width, height = value.lower().split("x")
        return int(width), int(height)
    except ValueError:
        raise argparse.ArgumentTypeError("use WIDTHxHEIGHT, e.g. 1920x1080")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Test webcam gaze tracking."
    )
    parser.add_argument(
        "--mode",
        choices=["preview", "calibrate", "track", "validate"],
        default="preview",
        help="preview: raw eye features; calibrate: calibrate, save and "
             "track; track: use the saved calibration; validate: measure "
             "accuracy (default: preview)"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--calibration",
        default=str(DEFAULT_CALIBRATION_PATH),
        help="Calibration file to save/load"
    )
    parser.add_argument(
        "--screen", type=_parseScreen,
        help="Screen size WIDTHxHEIGHT (default: detect)"
    )
    parser.add_argument(
        "--grid", type=int, default=4,
        help="Calibration grid size N for N x N points (default: 4)"
    )
    parser.add_argument(
        "--degree", type=int, choices=[1, 2], default=2,
        help="Mapping polynomial degree (default: 2)"
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.35,
        help="Gaze smoothing factor, 1 = none (default: 0.35)"
    )
    args = parser.parse_args(argv)

    if args.mode == "preview":
        runPreview(args.camera)
        return

    windowName = "Gaze"
    screenSize = args.screen or gc.getScreenSize()

    if args.mode == "calibrate":
        capture = _openCamera(args.camera)
        tracker = GazeTracker()

        try:
            gc.openFullscreen(windowName)
            mapper = gc.calibrate(
                tracker, capture, screenSize, windowName,
                grid=args.grid, degree=args.degree
            )
            mapper.save(args.calibration)
            print(f"Saved calibration to {args.calibration}")

            runTracking(
                mapper, screenSize=screenSize, smoothing=args.smoothing,
                tracker=tracker, capture=capture, windowName=windowName
            )
        except gc.CalibrationAborted:
            print("Calibration cancelled")
        finally:
            tracker.close()
            capture.release()
            cv2.destroyAllWindows()
        return

    calibrationPath = Path(args.calibration)

    if not calibrationPath.is_file():
        parser.error(
            f"No calibration at {calibrationPath}; "
            "run with --mode calibrate first"
        )

    mapper = gc.GazeMapper.load(calibrationPath)

    if args.mode == "track":
        runTracking(
            mapper, args.camera, screenSize, args.smoothing,
            windowName=windowName
        )
        return

    capture = _openCamera(args.camera)

    try:
        with GazeTracker() as tracker:
            gc.openFullscreen(windowName)
            gc.validate(tracker, capture, mapper, screenSize, windowName)
    except gc.CalibrationAborted:
        print("Validation cancelled")
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
