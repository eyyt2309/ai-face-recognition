
from pathlib import Path
from multiprocessing import freeze_support
from ultralytics import YOLO


def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    TRAINING_ROOT = Path(__file__).resolve().parent

    model = YOLO(str(PROJECT_ROOT / "models/yolo/yolo26n.pt"))

    model.train(
        data=str(PROJECT_ROOT / "datasets/hands-phones/data.yaml"),
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        device=0,
        workers=4,
        project=str(TRAINING_ROOT / "runs/attention"),
        name="hands_phones"
    )


if __name__ == "__main__":
    freeze_support()
    main()
