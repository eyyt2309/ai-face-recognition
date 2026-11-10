
from pathlib import Path
from multiprocessing import freeze_support
from ultralytics import YOLO


def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    TRAINING_ROOT = Path(__file__).resolve().parent

    model = YOLO(
        str(TRAINING_ROOT / "runs/attention/hands_phones_v2/weights/best.pt")
    )

    model.train(
        data=str(PROJECT_ROOT / "datasets/hands-phones-v2/data.yaml"),
        epochs=25,
        imgsz=640,
        batch=8,
        patience=10,
        device=0,
        workers=4,
        optimizer="AdamW",
        lr0=0.0001,
        resume=False,
        project=str(TRAINING_ROOT / "runs/attention"),
        name="hands_phones_v3"
    )


if __name__ == "__main__":
    freeze_support()
    main()
