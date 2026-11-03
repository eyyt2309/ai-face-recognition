
import cv2
from pathlib import Path
from collections import Counter

dataset = Path("datasets/phones-detection")

class_counts = Counter()
shown = 0

for split in ["train", "valid", "test"]:
    image_dir = dataset / split / "images"
    label_dir = dataset / split / "labels"

    for image_path in image_dir.iterdir():
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        annotations = []

        for line in label_path.read_text().splitlines():
            values = line.split()

            if len(values) != 5:
                continue

            cls_id = int(values[0])
            class_counts[cls_id] += 1

            if cls_id == 1:
                annotations.append(values)

        # Skip images without class 1
        if not annotations:
            continue

        frame = cv2.imread(str(image_path))

        if frame is None:
            continue

        height, width = frame.shape[:2]

        for values in annotations:
            _, xc, yc, w, h = map(float, values)

            x1 = int((xc - w / 2) * width)
            y1 = int((yc - h / 2) * height)
            x2 = int((xc + w / 2) * width)
            y2 = int((yc + h / 2) * height)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                "Class 1",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        cv2.imshow("Class 1 Inspection", frame)

        print(f"{split}: {image_path.name}")

        shown += 1

        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

        if shown >= 20:
            break

    if shown >= 20:
        break

cv2.destroyAllWindows()

print("Class counts:", dict(class_counts))
