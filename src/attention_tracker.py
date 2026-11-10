"""Reusable real-time attention tracking.

Usage as a module:

    from src.attention_tracker import AttentionTracker

    with AttentionTracker() as tracker:
        result = tracker.process(frame)
        tracker.annotate(frame, result)

Run directly to test with the default webcam:

    python -m src.attention_tracker [--camera 0] [--debug]
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

if __package__:
    from . import draw_features, get_features, predict_attention
else:
    import draw_features
    import get_features
    import predict_attention

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolveModelPath(value, variableName):
    """Resolve a model path, treating relative paths as project-root relative."""

    if not value:
        raise RuntimeError(
            f"{variableName} is not set in the project .env file"
        )

    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(
            f"{variableName} points to a missing file: {path}"
        )
    return path


class AttentionTracker:
    """Detects face, hands and phone in a frame and predicts attention.

    Model paths default to the DET_MODEL_PATH, FACE_MODEL_PATH,
    SVM_MODEL_PATH and SVM_NORMALISATION_PATH values in the project .env.
    """

    def __init__(
        self,
        detModelPath=None,
        faceModelPath=None,
        svmModelPath=None,
        scalerPath=None,
        debug=False
    ):
        load_dotenv(PROJECT_ROOT / ".env")

        def pick(explicit, variableName):
            return resolveModelPath(
                explicit or os.getenv(variableName),
                variableName
            )

        self.debug = debug

        self.detModel = YOLO(
            str(pick(detModelPath, "DET_MODEL_PATH"))
        )

        faceOptions = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(
                    pick(faceModelPath, "FACE_MODEL_PATH")
                )
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_detection_confidence=0.5
        )
        self.faceDetector = mp.tasks.vision.FaceDetector.create_from_options(
            faceOptions
        )

        self.svm = cv2.ml.SVM_load(
            str(pick(svmModelPath, "SVM_MODEL_PATH"))
        )

        scaler = np.load(pick(scalerPath, "SVM_NORMALISATION_PATH"))
        self.mean = scaler["mean"]
        self.std = scaler["std"]

        self.startTime = time.monotonic()
        self.lastTimeStamp = -1

    def process(self, frame):
        """Analyse one BGR frame.

        Returns a dict with:
            features:   detected face/hand/phone features
            prediction: {"label": 0|1, "attention": str}, or None
        """

        # MediaPipe video mode requires strictly increasing timestamps
        timeStamp = int((time.monotonic() - self.startTime) * 1000)
        timeStamp = max(timeStamp, self.lastTimeStamp + 1)
        self.lastTimeStamp = timeStamp

        features = get_features.getFeatures(
            frame=frame,
            timeStamp=timeStamp,
            detModel=self.detModel,
            faceDetector=self.faceDetector
        )

        prediction = predict_attention.predictAttention(
            svm=self.svm,
            mean=self.mean,
            std=self.std,
            features=features,
            debug=self.debug
        )

        return {"features": features, "prediction": prediction}

    def annotate(self, frame, result):
        """Draw detections and the attention label onto the frame in place."""

        draw_features.drawFeatures(frame, result["features"])

        prediction = result["prediction"]
        if prediction is None:
            return frame

        color = (0, 255, 0) if prediction["label"] == 0 else (0, 0, 255)

        cv2.putText(
            frame,
            f"Attention: {prediction['attention']}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )
        return frame

    def close(self):
        self.faceDetector.close()

    def __enter__(self):
        return self

    def __exit__(self, excType, excValue, traceback):
        self.close()


def runWebcam(cameraIndex=0, debug=False):
    """Run the tracker on a webcam until 'q' is pressed."""

    with AttentionTracker(debug=debug) as tracker:
        print("SVM input features:", tracker.svm.getVarCount())

        capture = cv2.VideoCapture(cameraIndex)

        if not capture.isOpened():
            raise RuntimeError("Error starting webcam")

        try:
            while True:
                ret, frame = capture.read()

                if not ret:
                    break

                result = tracker.process(frame)
                tracker.annotate(frame, result)

                cv2.imshow("Webcam Feed", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test attention tracking with a webcam."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the feature table for every frame"
    )
    args = parser.parse_args()

    runWebcam(cameraIndex=args.camera, debug=args.debug)
