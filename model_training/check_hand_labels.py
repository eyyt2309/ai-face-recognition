
from pathlib import Path
from collections import Counter

root = Path("datasets/hands-detection/labels")
issues = Counter()

for label_path in root.rglob("*.txt"):
    for line_num, line in enumerate(label_path.read_text().splitlines(), 1):
        values = line.split()

        if not values:
            continue

        if len(values) != 5:
            issues["wrong_length"] += 1
            print(f"Wrong length: {label_path}:{line_num}")
            continue

        try:
            x, y, w, h = map(float, values[1:])
        except ValueError:
            issues["non_numeric"] += 1
            print(f"Non-numeric: {label_path}:{line_num}")
            continue

        if not all(0 <= v <= 1 for v in (x, y, w, h)):
            issues["out_of_range"] += 1
            print(f"Out of range: {label_path}:{line_num}: {values}")

        elif (x - w / 2 < 0 or x + w / 2 > 1 or
              y - h / 2 < 0 or y + h / 2 > 1):
            issues["crosses_boundary"] += 1

        if w <= 0 or h <= 0:
            issues["invalid_size"] += 1

print("\nSummary:", dict(issues))
