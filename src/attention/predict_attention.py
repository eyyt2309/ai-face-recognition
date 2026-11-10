
import numpy as np


FEATURE_COLS = [
    "no_of_face",
    "face_x",
    "face_y",
    "face_w",
    "face_h",
    "face_con",
    "no_of_hand",
    "pose",
    "pose_x",
    "pose_y",
    "phone",
    "phone_x",
    "phone_y",
    "phone_w",
    "phone_h",
    "phone_con"
]


def predictAttention(svm, mean, std, features, debug=False):
    """
    Predict student attention using a trained OpenCV SVM.

    Labels:
        0 = Attentive
        1 = Inattentive
    """

    if features is None:
        return None

    face = features.get("face") or {}
    phone = features.get("phone") or {}
    poseData = face.get("pose") or {}

    # Must match the pose encoding during SVM training
    poseMapping = {
        "forward": 0,
        "left": 1,
        "down": 2,
        "right": 3
    }

    poseName = str(
        poseData.get("pose", "forward")
    ).strip().lower()

    if poseName not in poseMapping:
        raise ValueError(f"Unknown pose: {poseName}")

    poseValue = poseMapping[poseName]

    # Construct feature vector in exact training order
    featureVector = np.array([
        features.get("no_of_face", 0),

        face.get("x", 0.0),
        face.get("y", 0.0),
        face.get("w", 0.0),
        face.get("h", 0.0),
        face.get("confidence", 0.0) * 100, # rescale to 100 %

        features.get(
            "no_of_hands",
            features.get("no_of_hand", 0)
        ),

        poseValue,
        poseData.get("pose_x", 0.0),
        poseData.get("pose_y", 0.0),

        int(bool(phone)),
        phone.get("x", 0.0),
        phone.get("y", 0.0),
        phone.get("w", 0.0),
        phone.get("h", 0.0),
        phone.get("confidence", 0.0)
    ], dtype=np.float32)

    # Validate scaler
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)

    if (
        featureVector.size != mean.size
        or mean.size != std.size
    ):
        raise ValueError(
            f"Dimension mismatch: "
            f"features={featureVector.size}, "
            f"mean={mean.size}, std={std.size}"
        )

    if not np.isfinite(featureVector).all():
        raise ValueError("Invalid feature values")

    safeStd = np.where(std == 0, 1.0, std)

    scaledFeatures = (
        (featureVector - mean) / safeStd
    ).astype(np.float32)

    sample = scaledFeatures.reshape(1, -1)

    _, prediction = svm.predict(sample)

    label = int(prediction[0, 0])

    if debug:
        print("\n" + "=" * 75)
        print("SVM ATTENTION PREDICTION")
        print("=" * 75)

        print(
            f"{'Feature':<16}"
            f"{'Raw':>12}"
            f"{'Mean':>12}"
            f"{'Std':>12}"
            f"{'Scaled':>12}"
        )

        print("-" * 75)

        for name, raw, mu, sigma, scaled in zip(
            FEATURE_COLS,
            featureVector,
            mean,
            safeStd,
            scaledFeatures
        ):
            warning = " <-- CHECK" if abs(scaled) > 3 else ""

            print(
                f"{name:<16}"
                f"{raw:>12.3f}"
                f"{mu:>12.3f}"
                f"{sigma:>12.3f}"
                f"{scaled:>12.3f}"
                f"{warning}"
            )

        print("-" * 75)
        print(f"Prediction: {label}")
        print(
            "Attention:",
            "Attentive" if label == 0 else "Inattentive"
        )
        print("=" * 75)

    return {
        "label": label,
        "attention": (
            "Attentive" if label == 0
            else "Inattentive"
        )
    }
