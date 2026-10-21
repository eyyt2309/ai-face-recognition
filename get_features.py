import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2

def getFeatures(frame, timeStamp, detModel, landmarker):
    detResult = detModel(frame, verbose=False)[0]
    features = {}

    for box in detResult.boxes:
        cls_id = int(box.cls.item())
        if detResult.names[cls_id] == "person":
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            person = {
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1,
                "class": "person",
                "confidence": float(box.conf.item())
            }
            if "person" not in features:
                features["person"] = person
            elif features["person"]["confidence"] < person["confidence"]:
                features["person"] = person
        if detResult.names[cls_id] == "cell phone" or detResult.names[cls_id] == "remote":
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            phone = {
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1,
                "class": "phone",
                "confidence": float(box.conf.item())
            }
            if "phone" not in features:
                features["phone"] = phone
            elif features["phone"]["confidence"] < phone["confidence"]:
                features["phone"] = phone

    # Crop image and run pose estimation
    if "person" in features:
        x1 = int(features["person"]["x"])
        y1 = int(features["person"]["y"])
        x2 = int(features["person"]["x"] + features["person"]["w"])
        y2 = int(features["person"]["y"] + features["person"]["h"])
        cropImage = frame[y1:y2, x1:x2]
        height, width = cropImage.shape[:2]

        rgbImage = cv2.cvtColor(cropImage, cv2.COLOR_BGR2RGB)
        mpImage = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgbImage
        )


        landmarkerResult = landmarker.detect_for_video(mpImage, timeStamp)

        if landmarkerResult.face_landmarks:
            faceLandmarks = landmarkerResult.face_landmarks[0]

            frameHeight, frameWidth = frame.shape[:2]

            features["person"]["pose"] = getPose(
                faceLandmarks,
                height,
                width,
                frameHeight,
                frameWidth,
                x1,
                y1
            )

        return features



def getPose(faceLandmarks, cropHeight, cropWidth,
            frameHeight, frameWidth, offsetX, offsetY):

    indices = [1, 152, 33, 263, 61, 291]

    imagePoints = np.array([
        [
            faceLandmarks[i].x * cropWidth + offsetX,
            faceLandmarks[i].y * cropHeight + offsetY
        ]
        for i in indices
    ], dtype=np.float64)

    modelPoints = np.array([
        [0.0, 0.0, 0.0],          # Nose
        [0.0, -330.0, -65.0],    # Chin
        [-225.0, 170.0, -135.0], # Eye (image left)
        [225.0, 170.0, -135.0],  # Eye (image right)
        [-150.0, -150.0, -125.0],# Mouth (image left)
        [150.0, -150.0, -125.0]  # Mouth (image right)
    ], dtype=np.float64)

    # Convert model coordinate convention
    modelPoints[:, 1] *= -1
    modelPoints[:, 2] *= -1

    focalLength = frameWidth

    cameraMatrix = np.array([
        [focalLength, 0, frameWidth / 2],
        [0, focalLength, frameHeight / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    distCoeffs = np.zeros((4, 1))

    success, rotationVector, translationVector = cv2.solvePnP(
        modelPoints,
        imagePoints,
        cameraMatrix,
        distCoeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rotationMatrix, _ = cv2.Rodrigues(rotationVector)

    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotationMatrix)

    pitch, yaw, roll = angles

    pitch, yaw, roll = float(pitch), float(yaw), float(roll)

    # not using arbitrary pose estimation from paper

    # compare absolute values for pose_x and pose_y
    if abs(pitch) > abs(yaw):
        if pitch > 15:
            pose = 'right'
        elif pitch < -15:
            pose = 'left'
        else:
            pose = 'forward'
    else:
        if yaw < -10:
            pose = 'down'
        else:
            pose = 'forward'

    return {
        "pose_y": float(pitch),
        "pose_x": float(yaw),
        "roll": float(roll),
        "pose": pose,
    }

