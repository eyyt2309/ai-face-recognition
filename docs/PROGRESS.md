# Project Progress Tracker

Tracks what is implemented and what remains, aligned with [FYP-PLAN-UPDATED.md](FYP-PLAN-UPDATED.md).
Update the checkboxes and the "Last updated" date as work progresses.

**Last updated:** 2026-11-10

Legend: `[x]` done, `[~]` partly done / needs validation, `[ ]` not started.

## Implemented

### Core pipeline
- [x] Webcam capture and display loop (`src/attention/attention_tracker.py`, `runWebcam`)
- [x] Reusable `AttentionTracker` module: `process(frame)`, `annotate(frame, result)`, context-manager support
- [x] Direct-test entry point: `python -m src.attention.attention_tracker [--camera N] [--debug]`
- [x] Thin wrapper `src/main.py`, runnable from the project root or `src/`
- [x] Model paths configured through `.env` (relative paths resolve from the project root)

### Feature extraction (`src/attention/get_features.py`)
- [x] Face detection (MediaPipe BlazeFace) with bounding box and confidence
- [x] Head pose estimation via `solvePnP` from face keypoints (pitch, yaw, roll) and pose class
- [x] Hand and phone detection via the custom YOLO model, with counts, boxes and confidence
- [x] On-screen drawing of faces, keypoints, hands, phone and counts (`src/attention/draw_features.py`)

### Attention classifier
- [x] Attention dataset in `datasets/attention/` (4,000 samples; 2,314 attentive / 1,686 inattentive)
- [x] SVM training script with stratified 80/20 split, scaling, and metrics report (`model_training/train_attention_model.py`)
- [x] Trained SVM and scaler saved in `models/attention_model/`
- [x] Real-time SVM inference and on-screen label (`src/attention/predict_attention.py`)

### Detector training (hands and phones)
- [x] Hand and phone datasets merged into `datasets/hands-phones` (`model_training/merge_datasets.py`)
- [x] Label sanity-check and inspection scripts (`check_labels.py`, `check_hand_labels.py`, `inspect_phone_dataset.py`)
- [x] Reviewed pseudo-labelling workflow producing `hands-phones-v2` (1,503 accepted proposals)
- [x] Three YOLO training runs (`train.py`, `train_v2.py`, `train_v3.py`); deployed model is `models/customPhoneHand/best.pt`
- [x] Webcam detector diagnostic (`model_training/test_yolo.py`)

### Repository organisation
- [x] Code reorganised into `src/` and `model_training/`; datasets under `datasets/`; docs under `docs/`
- [x] README updated for the new layout and commands
- [x] Session recording (`python -m src.session.record`): attention, hands, phone, head pose and optional gaze logged to `sessions/*.csv` with a metadata file; **m** marks a new section, **q** stops
- [x] Post-session analysis (`python -m src.session.analyse`): overall attention, sustained inattention periods with a likely cause (no face / phone / looking away), attention per time window and per marked section, gaze off-screen share, 3x3 gaze distribution, text report and timeline chart
- [x] Unit tests for session logging and analysis (`python -m unittest discover -s tests -t .`)

## Needs validation (implemented but unverified)

- [~] Session recording and analysis are verified with a scripted synthetic session and a mocked camera, but not yet on a real recorded study session.
- [~] Record baseline SVM metrics (accuracy, precision, recall, F1, confusion matrix) and compare against simple baselines. The training script prints them, but no results are saved.
- [~] Confirm the head-pose thresholds match the original dataset's convention (`classifyPose` in `src/attention/get_features.py` still carries a TODO).
- [~] Confirm live features match training features. This covers units and scale (for example, face confidence is multiplied by 100 at inference), pose encoding and feature order.
- [~] Audit validation/test labels of the hand/phone dataset. Metrics are unreliable while they are incomplete (see `model_training/HAND_PREANNOTATION_README.md`).
- [~] Validate on independent webcam sessions rather than the dataset split alone.

## To do

### Attention (core)
- [ ] Temporal smoothing of predictions over multiple frames (only if it improves stability)
- [ ] Test across lighting, head poses, webcam quality and hand/phone visibility; document failure cases
- [ ] Record real study sessions and review the analysis output; tune the cause rules and thresholds (`src/session/analyse.py`)
- [ ] Measure end-to-end latency and CPU/GPU usage; check stability over longer sessions (recording already stores `processing_fps` in `sessions/*.meta.json`)
- [ ] Simple instructor-facing view (for example a single HTML report) built on the existing report and timeline chart
- [ ] Extend the automated tests to the trackers and mapper (only session analysis is covered so far, in `tests/`)

### Housekeeping
- [ ] Re-save `requirements.txt` as UTF-8 (it is currently UTF-16 encoded, which some tools can't read)
- [ ] Commit the reorganisation and the `docs/` folder (currently staged or untracked)

### Optional extensions
- [~] Eye tracking (`src/gaze/gaze_tracker.py`, `src/gaze/gaze_calibration.py`): iris-based gaze features, ridge-regression calibration, preview/calibrate/track/validate modes. First real-webcam test (one user, 1920x1080): mean validation error about 129 px; the tracked dot follows the eyes. Still to do: more users, sessions and lighting, plus head-movement tolerance.
- [ ] Eye tracking: measure real screen-point and region accuracy across several users/sessions
- [ ] Page and content analysis: define regions of interest and map gaze and fixation statistics to them
- [ ] Interest-based recommendation prototype (stretch goal)

### Final
- [ ] Run final experiments and write up limitations and privacy considerations
- [ ] Final report and demonstration materials

## Out of scope

- Facial-expression and emotion recognition (removed from the plan)
