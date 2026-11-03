# Repairing missing hand labels in the merged phone dataset

The existing YOLO model confuses hands with phones. Many phone-source images have phone annotations but lack hand annotations. This tool **proposes** MediaPipe hand boxes and requires explicit review before modifying a copy of the merged dataset.

## Setup

Run commands from the **project root**, not `model_training`.

MediaPipe in the project's requirements is the Tasks API version, and the archived project includes `face_landmarker.task` but **not** `hand_landmarker.task`. Download the official hand model to `models/landmarker/hand_landmarker.task`:

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "models/landmarker/hand_landmarker.task"
```

## Preview a small sample

```powershell
python model_training/preannotate_phone_hands.py propose --limit 100
```

The script reads existing `datasets/hands-phones/images/train/phone_*` files. It does not modify source datasets or any labels during proposal generation. Review images in `datasets/hand_review/previews/train/` (green = candidate hand, red = existing phone). Open `datasets/hand_review/review.csv` and mark the `status` column as `ACCEPT` only when **all proposed hand boxes in the image are correct**. Use `REJECT` for wrong proposals; leave `PENDING` if uncertain. The green boxes are estimated from hand landmarks and can omit parts of wrists/fingers. Ideally correct such cases in an annotation editor before accepting; `ACCEPT` only endorses the generated proposal file as it stands. The script does not visually detect whether the model missed a hand altogether.

## Full pass

Delete or move the sample `datasets/hand_review` folder first (only if you do not need its prior review decisions), then:

```powershell
python model_training/preannotate_phone_hands.py propose
```

After reviewing preview files, change selected `status` entries in `review.csv` from `PENDING` to `ACCEPT`. You can edit candidate YOLO annotations directly in `datasets/hand_review/proposals/train/*.txt` before accepting them.

## Create a new dataset version

```powershell
python model_training/preannotate_phone_hands.py apply
```

This copies `datasets/hands-phones` to `datasets/hands-phones-v2` (including images, which may require substantial free disk space), preserves existing phone boxes, appends reviewed hand boxes, and updates `data.yaml` `path`. It refuses to overwrite a preexisting v2 directory. **The original dataset is untouched.**

Only `train` receives proposals by default. The untouched validation/test labels remain incomplete for the two-class task. They must be independently audited/corrected before relying on metrics. Source hand images also need a review for missing phones. Unlabelled hands in images without proposals remain missing, and `PENDING` proposals remain unmerged. Do not use unreviewed pseudo-labels as ground truth.

If using the v2 dataset for a comparative training experiment, use a separate fully annotated webcam test set, start with `runs/attention/hands_phones/weights/best.pt`, and evaluate both classes before/after fine-tuning.
