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

The tracking logic lives in `src/attention/attention_tracker.py` (`AttentionTracker`),
which can be tested directly (`--camera N` selects a camera, `--debug` prints
per-frame features):

```powershell
python -m src.attention.attention_tracker
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

## Eye tracking (optional extension)

`src/gaze/gaze_tracker.py` estimates where on the screen you are looking from iris
position and head pose (MediaPipe face landmarker), mapped to screen
coordinates by a short per-user calibration (`src/gaze/gaze_calibration.py`).

```powershell
python -m src.gaze.gaze_tracker                      # preview: landmarks and gaze features
python -m src.gaze.gaze_tracker --mode calibrate     # look at each dot; saves models/gaze/gaze_calibration.npz
python -m src.gaze.gaze_tracker --mode validate      # reports pixel error on fresh points
python -m src.gaze.gaze_tracker --mode track         # live gaze point overlay
```

Options: `--camera`, `--calibration`, `--screen WxH`, `--grid`, `--degree {1,2}`,
`--smoothing`. Keep your head and the screen position steady between
calibration and use; press **q** or **Esc** to quit. The mapper uses only the
iris offsets (head pose features overfit during calibration). In a first
single-user test the mean validation error was about 130 px on a 1920x1080
screen; each calibration also saves raw samples next to the calibration file
for offline comparison.

## Record and analyse a study session

```powershell
python -m src.session.record                 # attention + gaze (if calibrated)
python -m src.session.record --no-gaze       # attention only
python -m src.session.analyse                # analyse the newest session
python -m src.session.analyse sessions/session_20261110_150000.csv --window 60
```

While recording, press **m** in the preview window to mark the start of a new
section (for example a new lecture topic) and **q** to stop. Recordings are
saved under `sessions/` (git-ignored, since they are personal data) as a CSV
with a `.meta.json` file. The analysis prints and saves a report with overall
attention, sustained inattention periods and their likely cause (no face,
phone, looking away), attention per time window and per marked section, and
where the gaze fell on the screen, plus a timeline chart (`*_timeline.png`).
Run the tests with `python -m unittest discover -s tests -t .`.

## Project layout

| Path | Purpose |
| --- | --- |
| `src/attention/` | Attention tracking module (`attention_tracker.py`) and its feature, drawing and prediction helpers |
| `src/gaze/` | Eye-tracking modules (`gaze_tracker.py`, `gaze_calibration.py`) |
| `src/session/` | Session recording (`record.py`, `session_logger.py`) and post-session analysis (`analyse.py`) |
| `tests/` | Unit tests |
| `src/` | Shared `config.py` and the `main.py` webcam entry point |
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
