# Real-Time Student Attention Analysis for Online Learning

**Ethan Yew Yi Teng**

| Project Title | Real-Time Student Attention Analysis for Online Learning |
| --- | --- |
| Project Summary | This project aims to develop a real-time computer-vision system that uses a student's webcam feed to estimate attention during online learning. The core system combines behavioural cues, including face presence, head orientation, hand activity, and mobile-phone usage, into an interpretable attention estimate. If time permits, a webcam-based eye-tracking and page-analysis extension will associate gaze with learning-page regions. The system will present attention trends through a live view and/or session summary for instructors. |

## 1. Project Objectives

1. Develop a real-time webcam pipeline that extracts visual and behavioural cues relevant to student attention.
2. Train, evaluate, and integrate an attention classifier using face presence, head orientation, hand activity, and mobile-phone usage.
3. Improve the stability and interpretability of real-time attention estimates and evaluate the system under realistic webcam conditions.
4. If time permits, develop an eye-tracking module and correlate gaze with regions or content shown on a learning page.
5. Present attention information through a real-time view and/or post-session summary that is useful to an instructor.
6. If the core objectives are complete, investigate a proof-of-concept recommendation feature based on attention and gaze/content signals.

## 2. Current Implementation and Progress

The repository contains a working baseline attention pipeline. The core webcam application and attention model are implemented; collecting real-world performance measurements and validating predictions on separate webcam sessions remain important next steps.

| Component | Current Status | Evidence in Current Codebase |
| --- | --- | --- |
| Webcam acquisition | Implemented | `src/attention/attention_tracker.py` (`runWebcam`, launched via `src/main.py`) captures frames using OpenCV. |
| Face detection and head pose | Implemented baseline | `src/attention/get_features.py` uses MediaPipe face detection, estimates head pose from face keypoints, and extracts face bounding-box and confidence features. |
| Hand and phone detection | Implemented baseline | `src/attention/get_features.py` uses the configured YOLO model to detect hands and phones and extracts counts, bounding boxes, and confidence values. |
| Attention dataset | Integrated | `datasets/attention/attention_detection_dataset_v1.csv` contains 4,000 samples with attention-related features and binary labels. |
| Attention classifier training | Implemented | `model_training/train_attention_model.py` trains a linear OpenCV SVM with a stratified 80/20 split and training-set feature normalization. It reports accuracy, per-class metrics, and a confusion matrix. |
| Real-time attention inference | Implemented | `src/attention/attention_tracker.py` and `predict_attention.py` load the SVM and scaler, predicts attention for each webcam frame, draws detections, and displays the result. |
| Eye tracking | Prototype working, limited validation | `src/gaze/` provides iris-based gaze features, regression-based screen calibration, and preview/calibrate/track/validate modes. A first single-user webcam test gave about 129 px mean error on a 1920x1080 screen; multi-user and head-movement evaluation is still needed. |
| Page analysis | Planned extension | Page-region mapping and gaze-derived content statistics are not implemented. |
| Session logging and reporting | Prototype implemented, unvalidated | `src/session/` records attention and gaze per frame to CSV and `analyse.py` reports sustained inattention periods with likely causes, attention per time window and marked section, and gaze off-screen statistics, with a timeline chart. Verified on synthetic and mocked data only; not yet on real study sessions. An instructor-facing view is still to be developed. |

The dataset labels are `0` (Attentive) and `1` (Inattentive). Dataset metrics are a baseline; they do not by themselves establish performance on live webcam footage.

## 3. Proposed System Scope and Architecture

The core system is a behavioural-attention pipeline. Optional extensions add gaze/page analysis and instructor-oriented reporting.

### 3.1 Behavioural Attention Module

- Use webcam frames to detect the face, hands, and mobile-phone presence.
- Estimate head orientation and extract relevant spatial features such as bounding-box location, size, and detection confidence.
- Apply the trained attention classifier during real-time operation and output an attention class for each frame or analysis window.
- Investigate temporal smoothing so that brief detection noise does not cause large changes in the displayed estimate.

### 3.2 Optional Eye-Tracking and Learning-Page Analysis

- Estimate gaze direction or screen gaze point from webcam-based eye and facial landmarks, with user calibration where necessary.
- Divide the learning page into regions of interest, such as text, diagrams, videos, questions, or navigation elements.
- If reliable gaze estimates are achievable, aggregate fixation duration and revisit frequency to describe which page regions receive attention.
- Where feasible, associate gaze behaviour with learning-material categories rather than recording only screen coordinates.

### 3.3 Attention Visualisation and Reporting

- Display live attention output and relevant behavioural cues.
- Where session logging is implemented, generate summaries of attention trends and periods of likely distraction.
- If the eye-tracking extension is completed, include page-region attention statistics in the visualisation or summary.

## 4. Methodology

**Data preparation and privacy:** Use the included attention dataset for baseline model development. Check feature definitions, label meaning, class balance, and data quality. Evaluate webcam performance using separate recorded or controlled sessions where consent and privacy requirements are addressed.

**Baseline attention modelling:** Reproduce and record the SVM baseline metrics, compare with simple baselines, and verify that feature extraction and preprocessing at inference match the training feature order and scaling.

**Real-time system validation:** Test the pipeline with different lighting, head poses, webcam quality, and hand/phone visibility. Measure prediction stability, frame rate, end-to-end latency, and failure cases. Add temporal smoothing only if it improves stability without masking meaningful changes.

**Eye tracking and page mapping (optional):** If time permits, implement gaze estimation, calibrate against known screen points, and map gaze positions to page regions of interest. Measure whether the gaze module can reliably identify the region being viewed.

**Interface and reporting:** Develop an instructor-facing view that emphasizes trends and interpretable information rather than exposing raw model outputs.

## 5. Evaluation Plan

Evaluate the attention model and end-to-end webcam pipeline separately. Attention is an estimate based on observable cues and should not be treated as a direct measurement of a student's internal state.

| Evaluation Area | Proposed Measures |
| --- | --- |
| Attention classifier | Accuracy, precision, recall, F1-score, and confusion matrix; compare against simple baselines. |
| Real-world attention behaviour | Performance across lighting, pose, webcam quality, and hand/phone visibility; document common failure cases. |
| Optional eye-tracking module | Screen-point or region-of-interest accuracy after calibration; stability under natural head movement. |
| Real-time system | Frames per second, end-to-end latency, CPU/GPU usage, and stability during longer sessions. |
| Usability of output | Assess whether the live view and summaries communicate attention trends clearly without overwhelming the instructor. |

## 6. Optional Extension: Interest-Based Recommendation

Recommendation is a stretch goal and should not dilute the core attention-analysis work. If the attention system and optional gaze/page-analysis module are completed and evaluated successfully, a proof-of-concept recommendation layer could rank educational content using attention and gaze-derived interest signals. The recommendation system is not a core deliverable.

## 7. Expected Deliverables

- A working real-time webcam application that extracts behavioural attention features and produces an attention estimate.
- A trained and evaluated attention classifier, with documented limitations and real-time performance measurements.
- A live view and/or session summary presenting attention trends in an interpretable way.
- If time permits, an eye-tracking prototype that maps gaze to learning-page regions and produces content-interest statistics.
- If the eye-tracking extension is completed, an exploratory method for combining attention and gaze/page signals.
- Project report, source code, experimental results, and demonstration materials.

## 8. Proposed Work Plan

| Phase | Main Activities |
| --- | --- |
| Phase 1 - Baseline implementation | Implement webcam feature extraction, train the SVM, and integrate real-time attention inference. Core implementation is present; baseline results and independent webcam validation still need to be recorded. |
| Phase 2 - Attention validation and stability (In Progress) | Verify feature/preprocessing consistency, record baseline and live-session metrics, test difficult webcam conditions, and assess temporal smoothing. |
| Phase 3 - Optional eye tracking | If time permits, implement gaze estimation and calibration; evaluate gaze and region accuracy under webcam conditions. |
| Phase 4 - Optional page/content analysis | If gaze estimation is reliable, define page regions and map gaze/fixation statistics to learning-page elements. |
| Phase 5 - Visualisation and reporting | Improve live attention presentation, add session logging and summaries, and optimize latency. Include gaze information only if the optional extension is successful. |
| Phase 6 - Final evaluation and report | Run experiments, document limitations and privacy considerations, and prepare the final report and demonstration. |
| Stretch phase | If the core milestones are complete, investigate a small recommendation prototype based on attention and, if available, gaze/content signals. |
