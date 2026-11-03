
import cv2


def drawFeatures(frame, features):
    """Draw face, hands, phone, and facial keypoints."""

    if features is None:
        return

    height, width = frame.shape[:2]

    faceFeatures = features.get("face")
    phoneFeatures = features.get("phone")
    handFeatures = features.get("hands", [])

    # Colors (BGR)
    FACE_COLOR = (0, 255, 0)      # Green
    HAND_COLOR = (0, 255, 255)    # Yellow
    PHONE_COLOR = (255, 0, 0)     # Blue
    KEYPOINT_COLOR = (0, 0, 255)  # Red

    # ==================================================
    # DRAW FACE
    # ==================================================

    if faceFeatures is not None:
        pt1 = (
            int(faceFeatures["x"]),
            int(faceFeatures["y"])
        )
        pt2 = (
            int(faceFeatures["x"] + faceFeatures["w"]),
            int(faceFeatures["y"] + faceFeatures["h"])
        )

        cv2.rectangle(frame, pt1, pt2, FACE_COLOR, 2)

        pose = faceFeatures.get("pose") or {}

        lines = [
            f"Confidence: {faceFeatures['confidence']:.2f}",
            f"Class: {faceFeatures['class']}"
        ]

        if pose:
            lines.extend([
                f"Pose_x: {pose.get('pose_x', 0):.2f}",
                f"Pose_y: {pose.get('pose_y', 0):.2f}",
                f"Roll: {pose.get('roll', 0):.2f}",
                f"Pose: {pose.get('pose', 'N/A')}"
            ])

        x = pt1[0]
        y = max(pt1[1] - 10, 20)

        for i, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (x, y + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                FACE_COLOR,
                2
            )

        # Draw BlazeFace's six facial keypoints
        for point in faceFeatures.get("keypoints", []):
            px = int(point["x"] * width)
            py = int(point["y"] * height)

            cv2.circle(
                frame,
                (px, py),
                3,
                KEYPOINT_COLOR,
                -1
            )

    # ==================================================
    # DRAW HANDS
    # ==================================================

    for i, hand in enumerate(handFeatures):
        pt1 = (
            int(hand["x"]),
            int(hand["y"])
        )
        pt2 = (
            int(hand["x"] + hand["w"]),
            int(hand["y"] + hand["h"])
        )

        cv2.rectangle(
            frame,
            pt1,
            pt2,
            HAND_COLOR,
            2
        )

        cv2.putText(
            frame,
            f"Hand {i + 1}: {hand['confidence']:.2f}",
            (pt1[0], max(pt1[1] - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            HAND_COLOR,
            2
        )

    # ==================================================
    # DRAW PHONE
    # ==================================================

    if phoneFeatures is not None:
        pt1 = (
            int(phoneFeatures["x"]),
            int(phoneFeatures["y"])
        )
        pt2 = (
            int(phoneFeatures["x"] + phoneFeatures["w"]),
            int(phoneFeatures["y"] + phoneFeatures["h"])
        )

        cv2.rectangle(
            frame,
            pt1,
            pt2,
            PHONE_COLOR,
            2
        )

        cv2.putText(
            frame,
            f"Phone: {phoneFeatures['confidence']:.2f}",
            (pt1[0], max(pt1[1] - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            PHONE_COLOR,
            2
        )

    # ==================================================
    # DRAW DETECTION COUNTS
    # ==================================================

    cv2.putText(
        frame,
        f"Faces: {features.get('no_of_face', 0)} | "
        f"Hands: {features.get('no_of_hand', 0)} | "
        f"Phone: {int(phoneFeatures is not None)}",
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
