import cv2

def drawFeatures(frame, features):
    """draws the bounding boxes for person, phone and hands"""
    personFeatures = features.get("person", None)
    phoneFeatures = features.get("cell phone", None)

    if personFeatures != None:
        pt1 = (
            int(personFeatures["x"]),
            int(personFeatures["y"])
        )
        pt2 = (
            int(personFeatures["x"] + personFeatures["w"]),
            int(personFeatures["y"] + personFeatures["h"])
        )
        cv2.rectangle(frame, pt1, pt2, (0,0,255), 2)

        pose = personFeatures.get("pose") or {}

        lines = [
            f"Confidence: {personFeatures['confidence']:.2f}",
            f"Class: {personFeatures['class']}",
            f"Pose_y: {pose.get('pose_y', 0):.2f}",
            f"Pose_x: {pose.get('pose_x', 0):.2f}",
            f"Roll: {pose.get('roll', 0):.2f}",
            f"Pose: {pose.get('pose', None)}",
        ]

        x = pt1[0]
        y = max(pt1[1] - 10, 20)

        for i, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (x, y + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

    if phoneFeatures != None:
        pt1 = (
            int(phoneFeatures["x"]),
            int(phoneFeatures["y"])
        )
        pt2 = (
            int(phoneFeatures["x"] + phoneFeatures["w"]),
            int(phoneFeatures["y"] + phoneFeatures["h"])
        )
        cv2.rectangle(frame, pt1, pt2, (255,0,0), 2)
        cv2.putText(
            frame, 
            f"Confidence: {phoneFeatures["confidence"]}, Category: {phoneFeatures["class"]}",
            (pt1[0], pt1[1] + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,0,0),
            2
        )

