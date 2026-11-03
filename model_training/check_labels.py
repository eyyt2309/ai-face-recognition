
from pathlib import Path
from collections import Counter

dataset = Path("datasets/phones-detection")

formats = Counter()
invalid_files = []

for split in ["train", "valid", "test"]:
    label_dir = dataset / split / "labels"

    for label_path in label_dir.glob("*.txt"):
        for line_num, line in enumerate(
            label_path.read_text().splitlines(), start=1
        ):
            values = line.split()

            if not values:
                continue

            formats[len(values)] += 1

            if len(values) != 5:
                invalid_files.append(
                    (str(label_path), line_num, len(values))
                )

print("Annotation lengths:", dict(formats))
print("Non-standard annotations:", len(invalid_files))

for path, line_num, count in invalid_files[:10]:
    print(f"{path} | line {line_num} | {count} values")
