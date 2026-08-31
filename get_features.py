import mediapipe as mp

def getFeatures(frame, timestamp_ms, fp_detector, h_detector):
    """uses image obtained from opencv webcam and returns a list of features for attention model"""

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame
    )

    """gets """
    fp_features = getFPFeature(image, fp_detector, timestamp_ms)
    h_features  = getHandFeature(image, h_detector, timestamp_ms)

    return fp_features, h_features

def getFPFeature(image, detector, timestamp_ms):
    """returns dict containing face & phone features"""

    detection_result = detector.detect_for_video(image, timestamp_ms)

    features = {}

    for detection in detection_result.detections:
        box = detection.bounding_box
        x, y, w, h = box.origin_x, box.origin_y, box.width, box.height

        categories = detection.categories[0]
        score = categories.score
        category_name = categories.category_name

        obj_features = {
            "box": {
                "x": x,
                "y": y,
                "w": w,
                "h": h
            },
            "score": score,
            "category_name": category_name
        }

        if category_name not in features:
            features[category_name] = obj_features

    return features

def getHandFeature(image, detector, timestamp_ms) -> int:
    "return hands feature result"

    detection_result = detector.detect_for_video(image, timestamp_ms)

    num_hands = len(detection_result.handedness)

    print(num_hands)


    

