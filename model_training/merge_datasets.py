
from pathlib import Path
from collections import Counter
import numpy as np
import shutil

ROOT = Path(__file__).resolve().parents[1]

HAND_DIR = ROOT / "datasets/hands-detection"
PHONE_DIR = ROOT / "datasets/phones-detection"
OUTPUT_DIR = ROOT / "datasets/hands-phones"

# Final class IDs
# 0 = hand
# 1 = cell phone

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def convertAnnotation(values):
    """
    Convert YOLO detection or segmentation annotation
    into a valid normalized YOLO bounding box.

    Clips bounding boxes to image boundaries.
    """

    if len(values) == 5:
        # Standard YOLO bounding box
        xc, yc, w, h = map(float, values[1:])

        if not np.isfinite([xc, yc, w, h]).all():
            raise ValueError("Non-finite bounding box")

        if w <= 0 or h <= 0:
            raise ValueError("Invalid bounding box dimensions")

        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2

    elif len(values) >= 7 and len(values) % 2 == 1:
        # Polygon annotation
        coords = np.array(
            list(map(float, values[1:])),
            dtype=np.float64
        )

        if not np.isfinite(coords).all():
            raise ValueError("Non-finite polygon coordinates")

        xs = coords[0::2]
        ys = coords[1::2]

        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs)
        y2 = max(ys)

    else:
        raise ValueError(
            f"Unsupported annotation length: {len(values)}"
        )

    # Clip to image boundaries
    x1 = np.clip(x1, 0.0, 1.0)
    y1 = np.clip(y1, 0.0, 1.0)
    x2 = np.clip(x2, 0.0, 1.0)
    y2 = np.clip(y2, 0.0, 1.0)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            "Bounding box has no area after clipping"
        )

    # Convert back to YOLO format
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    return [xc, yc, w, h]


def mergeDataset(source, prefix, layout, targetClass):
    """
    Merge a dataset into the combined dataset.

    layout = "hand":
        images/train, images/val
        labels/train, labels/val

    layout = "phone":
        train/images, valid/images, test/images
        train/labels, valid/labels, test/labels
    """

    if layout == "hand":
        splits = [
            ("train", "train"),
            ("val", "val")
        ]
    elif layout == "phone":
        splits = [
            ("train", "train"),
            ("valid", "val"),
            ("test", "test")
        ]
    else:
        raise ValueError(f"Unknown dataset layout: {layout}")

    for sourceSplit, targetSplit in splits:

        if layout == "hand":
            imageDir = source / "images" / sourceSplit
            labelDir = source / "labels" / sourceSplit
        else:
            imageDir = source / sourceSplit / "images"
            labelDir = source / sourceSplit / "labels"

        if not imageDir.is_dir():
            raise FileNotFoundError(imageDir)

        if not labelDir.is_dir():
            raise FileNotFoundError(labelDir)

        targetImages = OUTPUT_DIR / "images" / targetSplit
        targetLabels = OUTPUT_DIR / "labels" / targetSplit

        targetImages.mkdir(parents=True, exist_ok=True)
        targetLabels.mkdir(parents=True, exist_ok=True)

        imageCount = 0
        annotationCount = 0
        polygonCount = 0

        for imagePath in sorted(imageDir.iterdir()):

            if imagePath.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            labelPath = labelDir / f"{imagePath.stem}.txt"

            newStem = f"{prefix}_{imagePath.stem}"
            newImage = targetImages / f"{newStem}{imagePath.suffix}"
            newLabel = targetLabels / f"{newStem}.txt"

            convertedLines = []

            if labelPath.exists():
                for lineNumber, line in enumerate(
                    labelPath.read_text(
                        encoding="utf-8-sig"
                    ).splitlines(),
                    start=1
                ):
                    values = line.split()

                    if not values:
                        continue

                    try:
                        sourceClass = int(values[0])

                        # Both cleaned source datasets
                        # should contain only class 0.
                        if sourceClass != 0:
                            raise ValueError(
                                f"Unexpected source class: {sourceClass}"
                            )

                        if len(values) != 5:
                            polygonCount += 1

                        bbox = convertAnnotation(values)

                        # Remap source class to final class
                        outputValues = [
                            str(targetClass),
                            *[f"{v:.8f}" for v in bbox]
                        ]

                        convertedLines.append(
                            " ".join(outputValues)
                        )

                        annotationCount += 1

                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid annotation in {labelPath}, "
                            f"line {lineNumber}: {exc}"
                        ) from exc

            # Only write the image after its labels
            # have been successfully validated.
            if newImage.exists() or newLabel.exists():
                raise FileExistsError(
                    f"Output already exists: {newImage}"
                )

            shutil.copy2(imagePath, newImage)

            newLabel.write_text(
                "\n".join(convertedLines),
                encoding="utf-8"
            )

            imageCount += 1

        print(f"\n{prefix.upper()} - {targetSplit}")
        print(f"  Images: {imageCount}")
        print(f"  Annotations: {annotationCount}")
        print(f"  Polygons converted: {polygonCount}")


def verifyDataset():
    """Check the final merged annotations."""

    print("\n========== VERIFICATION ==========")

    totalCounts = Counter()

    for split in ["train", "val", "test"]:
        labelDir = OUTPUT_DIR / "labels" / split

        counts = Counter()
        invalid = 0

        if not labelDir.exists():
            continue

        for labelPath in labelDir.glob("*.txt"):
            for line in labelPath.read_text().splitlines():
                values = line.split()

                if not values:
                    continue

                if len(values) != 5:
                    invalid += 1
                    continue

                classID = int(values[0])
                counts[classID] += 1

        totalCounts.update(counts)

        print(f"\n{split.upper()}")
        print(f"  Hands: {counts[0]}")
        print(f"  Phones: {counts[1]}")
        print(f"  Invalid annotations: {invalid}")

    print("\nTOTAL")
    print(f"  Hands: {totalCounts[0]}")
    print(f"  Phones: {totalCounts[1]}")


def createYaml():
    """Create Ultralytics dataset configuration."""

    yamlText = f"""path: {OUTPUT_DIR.as_posix()}

train: images/train
val: images/val
test: images/test

nc: 2

names:
  0: hand
  1: cell phone
"""

    (OUTPUT_DIR / "data.yaml").write_text(
        yamlText,
        encoding="utf-8"
    )


def main():
    # Prevent accidentally mixing old and new outputs
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {OUTPUT_DIR}\n"
            "Delete the previous generated hands-phones "
            "directory before rerunning."
        )

    mergeDataset(
        source=HAND_DIR,
        prefix="hand",
        layout="hand",
        targetClass=0
    )

    mergeDataset(
        source=PHONE_DIR,
        prefix="phone",
        layout="phone",
        targetClass=1
    )

    createYaml()
    verifyDataset()

    print("\nDataset merging complete!")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
