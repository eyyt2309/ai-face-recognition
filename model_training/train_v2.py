
from pathlib import Path
from multiprocessing import freeze_support
from ultralytics import YOLO


def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    TRAINING_ROOT = Path(__file__).resolve().parent

    # Start from your previously trained model
    model = YOLO(
        str(TRAINING_ROOT / "runs/attention/hands_phones/weights/best.pt")
    )

    # Fine-tune on the corrected dataset
    model.train(
        data=str(PROJECT_ROOT / "datasets/hands-phones-v2/data.yaml"),
        epochs=30,
        imgsz=640,
        batch=8,
        patience=10,
        device=0,
        workers=4,
        optimizer="AdamW",
        lr0=0.0005,
        resume=False,
        project=str(TRAINING_ROOT / "runs/attention"),
        name="hands_phones_v2"
    )


if __name__ == "__main__":
    freeze_support()
    main()
