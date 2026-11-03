import cv2
import numpy as np
import get_features as gf
import draw_features as df
import predict_attention as pa
from ultralytics import YOLO
import mediapipe as mp
import os
import time
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# LOAD MODELS
# ==========================================

# YOLO model for hand and phone detection
detModel = YOLO(str(os.getenv("DET_MODEL_PATH")))

# BlazeFace face detector
modelPath = os.getenv("FACE_MODEL_PATH")

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=modelPath),
    running_mode=VisionRunningMode.VIDEO,
    min_detection_confidence=0.5
)

# SVM attention classifier
svm = cv2.ml.SVM_load(
    "models/attention_model/attention_svm_model.yml"
)

# Load normalization values
scaler = np.load(
    "models/attention_model/attention_scaler.npz"
)

mean = scaler["mean"]
std = scaler["std"]

print("SVM input features:", svm.getVarCount())
print("Mean:", mean)
print("Std:", std)

# ==========================================
# INITIALIZE WEBCAM
# ==========================================

capture = cv2.VideoCapture(0)

if not capture.isOpened():
    raise RuntimeError("Error starting webcam")

startTime = time.monotonic()

# ==========================================
# MAIN LOOP
# ==========================================

with FaceDetector.create_from_options(options) as faceDetector:
    while True:
        ret, frame = capture.read()

        if not ret:
            break

        timeStamp = int(
            (time.monotonic() - startTime) * 1000
        )

        # Extract face, hand and phone features
        features = gf.getFeatures(
            frame=frame,
            timeStamp=timeStamp,
            detModel=detModel,
            faceDetector=faceDetector
        )

        # Predict attention using the SVM
        prediction = pa.predictAttention(
            svm=svm,
            mean=mean,
            std=std,
            features=features,
            debug=True
        )

        # Draw detected objects
        df.drawFeatures(frame, features)

        # Display attention prediction
        if prediction is not None:
            label = prediction["label"]
            attention = prediction["attention"]

            color = (
                (0, 255, 0)
                if label == 0
                else (0, 0, 255)
            )

            cv2.putText(
                frame,
                f"Attention: {attention}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        cv2.imshow("Webcam Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

capture.release()
cv2.destroyAllWindows()
