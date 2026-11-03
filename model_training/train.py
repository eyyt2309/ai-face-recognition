
from pathlib import Path
from multiprocessing import freeze_support
from ultralytics import YOLO


def main():
    ROOT = Path(__file__).resolve().parent

    model = YOLO("yolo26n.pt")

    model.train(
        data=str(ROOT / "datasets/hands-phones/data.yaml"),
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        device=0,
        workers=4,
        project=str(ROOT / "runs/attention"),
        name="hands_phones"
    )


if __name__ == "__main__":
    freeze_support()
    main()
