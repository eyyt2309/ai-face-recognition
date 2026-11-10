
import mediapipe as mp
import numpy as np
import cv2


def getFeatures(frame, timeStamp, detModel, faceDetector):
    features = {}

    frameHeight, frameWidth = frame.shape[:2]

    # ==================================================
    # YOLO HAND AND PHONE DETECTION
    # ==================================================

    # New YOLO26n model:
    # Class 0 = hand
    # Class 1 = cell phone

    detResult = detModel(
        frame,
        conf=0.25,
        verbose=False
    )[0]

    features["no_of_hand"] = 0
    features["hands"] = []

    for box in detResult.boxes:
        cls_id = int(box.cls.item())
        confidence = float(box.conf.item())

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detectedObject = {
            "x": float(x1),
            "y": float(y1),
            "w": float(x2 - x1),
            "h": float(y2 - y1),
            "confidence": confidence
        }

        if cls_id == 0:
            # Hand detection
            hand = {
                **detectedObject,
                "class": "hand"
            }

            features["hands"].append(hand)

        elif cls_id == 1:
            # Phone detection
            phone = {
                **detectedObject,
                "class": "phone"
            }

            # Keep the highest-confidence phone
            if (
                "phone" not in features
                or confidence > features["phone"]["confidence"]
            ):
                features["phone"] = phone

    features["no_of_hand"] = len(features["hands"])

    # ==================================================
    # MEDIAPIPE FACE DETECTION
    # ==================================================

    # Convert BGR to RGB
    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mpImage = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgbImage
    )

    faceResult = faceDetector.detect_for_video(
        mpImage,
        timeStamp
    )

    features["no_of_face"] = len(faceResult.detections)

    if faceResult.detections:
        # Select highest-confidence face
        detection = max(
            faceResult.detections,
            key=lambda d: d.categories[0].score
        )

        box = detection.bounding_box

        pose = getPose(
            detection.keypoints,
            frameHeight,
            frameWidth
        )

        features["face"] = {
            "x": float(box.origin_x),
            "y": float(box.origin_y),
            "w": float(box.width),
            "h": float(box.height),
            "class": "face",
            "confidence": float(detection.categories[0].score),
            "pose": pose,
            "keypoints": [
                {
                    "x": float(p.x),
                    "y": float(p.y)
                }
                for p in detection.keypoints
            ]
        }

    return features


def getPose(keypoints, frameHeight, frameWidth):
    # BlazeFace keypoint order:
    # 0: Right eye
    # 1: Left eye
    # 2: Nose tip
    # 3: Mouth center
    # 4: Right ear tragion
    # 5: Left ear tragion

    imagePoints = np.array([
        [
            point.x * frameWidth,
            point.y * frameHeight
        ]
        for point in keypoints
    ], dtype=np.float64)

    # Approximate 3D face geometry
    modelPoints = np.array([
        [-32.0, -25.0,   0.0],
        [ 32.0, -25.0,   0.0],
        [  0.0,   0.0, -35.0],
        [  0.0,  35.0,  -5.0],
        [-75.0,   0.0,  15.0],
        [ 75.0,   0.0,  15.0]
    ], dtype=np.float64)

    # Approximate camera intrinsics
    focalLength = frameWidth

    cameraMatrix = np.array([
        [focalLength, 0, frameWidth / 2],
        [0, focalLength, frameHeight / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    distCoeffs = np.zeros((4, 1), dtype=np.float64)

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

    angles, _, _, _, _, _ = cv2.RQDecomp3x3(
        rotationMatrix
    )

    pitch, yaw, roll = map(float, angles)

    # Preserve existing pitch convention
    pitch = -pitch

    pose = classifyPose(pitch, yaw)

    return {
        "pose_x": pitch,
        "pose_y": yaw,
        "roll": roll,
        "pose": pose
    }


def classifyPose(pitch, yaw):
    # TODO: Verify against the original dataset convention

    if yaw > 15:
        return "right"
    elif yaw < -10:
        return "left"
    elif pitch < -10:
        return "down"
    else:
        return "forward"
