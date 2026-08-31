import cv2
import numpy as np
import pandas as pd

csv_path = "./Students Attention Detection Dataset/attention_detection_dataset_v1.csv"

df = pd.read_csv(csv_path)

# convert pose text to numbers
pose_map = {
    "forward": 0,
    "left": 1,
    "down": 2,
    "right": 3
}

df["pose"] = df["pose"].map(pose_map)

# features
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

X = df[feature_cols].values.astype(np.float32)
y = df["label"].values.astype(np.int32)

indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

# train test split
split = int(0.8 * len(X))
X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

# normalisation
mean = X_train.mean(axis = 0)
std = X_train.std(axis = 0)

std[std == 0] = 1

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# create svm model
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(1.0)

# train model
svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

# test
_, predictions = svm.predict(X_test)
predictions = predictions.flatten().astype(np.int32)

accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}")

# save model
svm.save("models/attention_model/attention_svm_model.yml")
np.savez("models/attention_model/attention_scaler.npz", mean = mean, std = std)

