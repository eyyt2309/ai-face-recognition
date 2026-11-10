# Student Attention Detection

A webcam prototype that estimates student attention from face presence, head
pose, hand activity, and mobile-phone detections. It combines MediaPipe face
detection, a YOLO hand/phone detector, and a trained OpenCV SVM. This project
does not identify people or classify facial expressions.

## Setup

Use Python with the packages in [`requirements.txt`](requirements.txt), a
working webcam, and the model/data assets referenced below. Create and activate
a virtual environment from the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a root-level `.env` with paths to the detector and attention-model files.
Relative paths are resolved from the project root:

```dotenv
DET_MODEL_PATH=models/customPhoneHand/best.pt
FACE_MODEL_PATH=models/blazface/blaze_face_short_range.tflite
SVM_MODEL_PATH=models/attention_model/attention_svm_model.yml
SVM_NORMALISATION_PATH=models/attention_model/attention_scaler.npz
```

## Run the webcam app

From the project root:

```powershell
python -m src.main
```

The tracking logic lives in `src/attention_tracker.py` (`AttentionTracker`),
which can be tested directly (`--camera N` selects a camera, `--debug` prints
per-frame features):

```powershell
python -m src.attention_tracker
```

`src/main.py` is a thin wrapper around it and can be run directly from either
the project root or `src/`:

```powershell
python .\src\main.py
cd .\src
python main.py
```

The app opens the default webcam, draws detections, and displays the attention
prediction. Press **q** in the video window to stop.

## Train the attention classifier

Place `attention_detection_dataset_v1.csv` in
`datasets/attention/attention_detection_dataset_v1.csv`, then run:

```powershell
python -m model_training.train_attention_model
```

The script uses a reproducible, stratified 80/20 split, standardizes features
using training-set statistics, trains a linear SVM, and reports accuracy,
per-class metrics, and a confusion matrix. It writes the model and scaler to
`models/attention_model/`; rerunning it overwrites those files.

## Train or inspect the hand/phone detector

Detector datasets belong under `datasets/`, and training runs are kept under
`model_training/runs/attention/`. Run the training scripts from the project
root:

```powershell
python -m model_training.train
python -m model_training.train_v2
python -m model_training.train_v3
```

See [`model_training/HAND_PREANNOTATION_README.md`](model_training/HAND_PREANNOTATION_README.md)
for the reviewed hand-label workflow. `python model_training/test_yolo.py`
opens the webcam to inspect the v3 detector.

## Project layout

| Path | Purpose |
| --- | --- |
| `src/` | Attention tracking module (`attention_tracker.py`), its feature/drawing/prediction helpers, and the webcam entry point |
| `model_training/` | Model training, dataset utilities, diagnostics, and run outputs |
| `datasets/` | Training datasets and the attention CSV |
| `models/` | Detector and attention-model assets |
| `docs/` | Project planning documents |

The attention feature vector includes face count/bounding box/confidence, hand
count, encoded head pose and pose angles, and phone presence/bounding
box/confidence. Labels are `0` (Attentive) and `1` (Inattentive).

## Acknowledgements

The attention dataset and baseline are based on the _Students Attention
Detection Dataset 2023_ by Muhammad Kamal Hossen and Mohammad Shorif Uddin.
MediaPipe and Ultralytics provide the detection components used by the project.
