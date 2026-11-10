
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    ROOT
    / "datasets"
    / "attention"
    / "attention_detection_dataset_v1.csv"
)

OUTPUT_DIR = ROOT / "models/attention_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(CSV_PATH)

# Convert categorical pose into integers
pose_map = {
    "forward": 0,
    "left": 1,
    "down": 2,
    "right": 3
}

df["pose"] = (
    df["pose"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(pose_map)
)

# Define the exact feature order
feature_cols = [
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

# Check for missing or invalid values
required_cols = feature_cols + ["label"]

if df[required_cols].isna().any().any():
    missing = df[required_cols].isna().sum()
    raise ValueError(
        f"Dataset contains missing or invalid values:\n"
        f"{missing[missing > 0]}"
    )

X = df[feature_cols].to_numpy(dtype=np.float32)
y = df["label"].to_numpy(dtype=np.int32)

if not np.isfinite(X).all():
    raise ValueError("Features contain non-finite values")

# Reproducible, stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Normalize using training data only
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

std[std == 0] = 1.0

X_train_scaled = (
    (X_train - mean) / std
).astype(np.float32)

X_test_scaled = (
    (X_test - mean) / std
).astype(np.float32)

# Create linear SVM
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(1.0)

# Train
svm.train(
    X_train_scaled,
    cv2.ml.ROW_SAMPLE,
    y_train
)

# Evaluate
_, predictions = svm.predict(X_test_scaled)
predictions = predictions.flatten().astype(np.int32)

accuracy = np.mean(predictions == y_test)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Attentive", "Inattentive"],
        zero_division=0
    )
)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

# Save model
svm.save(
    str(OUTPUT_DIR / "attention_svm_model.yml")
)

# Save preprocessing configuration
np.savez(
    OUTPUT_DIR / "attention_scaler.npz",
    mean=mean,
    std=std,
    feature_cols=np.array(feature_cols)
)

print("\nModel and scaler saved successfully.")
