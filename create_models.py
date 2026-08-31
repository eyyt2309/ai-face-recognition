from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

def createFPDetector():
    face_phone_model_path = os.getenv('FACE_PHONE_MODEL_PATH')

    valid_classes = ["person", "cell phone"]

    base_options = python.BaseOptions(
        model_asset_path=face_phone_model_path
    )
    options = vision.ObjectDetectorOptions(
        running_mode=vision.RunningMode.VIDEO,
        base_options=base_options,
        score_threshold=0.3,
        max_results=-1,
        category_allowlist=valid_classes
    )
    detector = vision.ObjectDetector.create_from_options(options)
    return detector

def createHDetector():
    hand_model_path = os.getenv('HAND_MODEL_PATH')

    base_options = python.BaseOptions(
        model_asset_path=hand_model_path,
    )
    options = vision.HandLandmarkerOptions(
        running_mode=vision.RunningMode.VIDEO,
        base_options=base_options,
        num_hands=2,
        min_hand_presence_confidence=0.0
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return detector