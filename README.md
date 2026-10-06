# AI Face Recognition

A Python webcam prototype that runs MediaPipe person/cell-phone object detection and hand landmark detection. The repository also includes a separate script for training an SVM attention classifier from the included attention dataset.

> **Current scope:** the webcam script displays the camera feed and runs the detectors, but it does not yet draw detections on the video or run the trained attention classifier. The live hand feature function currently prints the number of detected hands rather than returning features. This is not an identity-recognition or facial-expression-recognition application.

## Requirements

- Python with the packages pinned in [`requirements.txt`](requirements.txt)
- A working webcam
- The model assets included in `models/`

## Setup

From the project root, create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root with paths to the included detector models:

```dotenv
FACE_PHONE_MODEL_PATH=models/face_recognition_model/efficientdet_lite0.tflite
HAND_MODEL_PATH=models/hand_detection_model/hand_landmarker.task
```

These variables are read by `create_models.py` when `main.py` creates the MediaPipe detectors. The paths above are relative to the project root; run the scripts from that directory.

## Run the webcam prototype

```powershell
python main.py
```

The program opens the default camera and displays its feed. Press **q** in the video window to stop. If the camera cannot be opened, check that it is connected and available to the Python process.

## Train the attention classifier

The included dataset is `Students Attention Detection Dataset/attention_detection_dataset_v1.csv`. From the project root, run:

```powershell
python train_attention_model.py
```

The script shuffles the rows, uses an 80/20 train/test split, standardizes the features, trains a linear SVM, prints test accuracy, and saves the classifier and scaler to:

- `models/attention_model/attention_svm_model.yml`
- `models/attention_model/attention_scaler.npz`

Running the training script overwrites those model files. Training is currently separate from the webcam application.

## Project layout

| Path | Purpose |
| --- | --- |
| `main.py` | Opens the webcam and invokes the detector/feature functions for each frame |
| `create_models.py` | Configures MediaPipe person/cell-phone and hand detectors |
| `get_features.py` | Extracts person/cell-phone detections and reports the hand count |
| `train_attention_model.py` | Trains and evaluates the attention SVM using the included CSV |
| `models/face_recognition_model/` | Included EfficientDet Lite 0 model asset |
| `models/hand_detection_model/` | Included MediaPipe hand landmark model |
| `models/attention_model/` | Saved attention SVM and feature scaler |

## Acknowledgements

- Face/phone detector and hand landmark model assets are from [MediaPipe](https://developers.google.com/mediapipe).
- The attention model and dataset are based on the *Students Attention Detection Dataset 2023* by Muhammad Kamal Hossen and Mohammad Shorif Uddin.
