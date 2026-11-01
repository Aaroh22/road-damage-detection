
# Phase 1: Bounding-Box Size Audit
# Measure how large road-damage annotations become after
# resizing images to fit within a 640x640 YOLO input.
# Compare box sizes across classes, domains, and dataset splits.
# No data is modified and no model training is performed.

from pathlib import Path
from collections import defaultdict
from statistics import median
from PIL import Image
import csv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/processed/main_split"
OUTPUT = ROOT / "runs/dataset_audit/box_sizes_640.csv"

IMAGE_SIZE = 640
SPLITS = ["train", "val"]

CLASSES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
}

DOMAINS = [
    "China_Drone",
    "China_MotorBike",
    "Czech",
    "India",
    "Japan",
    "Norway",
    "United_States",
]

stats = defaultdict(list)


def get_domain(filename):
    for domain in DOMAINS:
        if filename.startswith(domain + "_"):
            return domain

    raise ValueError(f"Unknown domain: {filename}")


def analyze_split(split):
    image_dir = DATA / "images" / split
    label_dir = DATA / "labels" / split

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.is_file():
            raise FileNotFoundError(label_path)

        lines = label_path.read_text().splitlines()

        if not lines:
            continue

        with Image.open(image_path) as image:
            width, height = image.size

        # Resize while keeping the original aspect ratio
        scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
        domain = get_domain(image_path.name)

        for line in lines:
            if not line.strip():
                continue

            values = line.split()

            if len(values) != 5:
                raise ValueError(f"Invalid annotation: {label_path}")

            cls = int(values[0])

            if cls not in CLASSES:
                raise ValueError(f"Unknown class {cls}: {label_path}")

            # YOLO labels contain normalized width and height
            box_width = float(values[3]) * width * scale
            box_height = float(values[4]) * height * scale

            short_side = min(box_width, box_height)
            box_area = box_width * box_height

            # Store results for overall, class and domain summaries
            for group, name in [
                ("Overall", "All"),
                ("Class", CLASSES[cls]),
                ("Domain", domain),
            ]:
                stats[(split, group, name)].append(
                    (short_side, box_area)
                )


def main():
    for split in SPLITS:
        analyze_split(split)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for (split, group, name), boxes in stats.items():
        total = len(boxes)

        short_sides = [box[0] for box in boxes]
        areas = [box[1] for box in boxes]

        rows.append({
            "split": split,
            "group": group,
            "name": name,
            "boxes": total,
            "median_short_side_px": round(median(short_sides), 2),
            "under_8px_pct": round(
                100 * sum(x < 8 for x in short_sides) / total, 2
            ),
            "under_16px_pct": round(
                100 * sum(x < 16 for x in short_sides) / total, 2
            ),
            "area_under_1024_pct": round(
                100 * sum(x < 1024 for x in areas) / total, 2
            ),
        })

    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for split in SPLITS:
        print(f"\n===== {split.upper()} =====")

        for row in rows:
            if row["split"] != split:
                continue

            print(
                f"{row['group']:7} | {row['name']:20} | "
                f"Boxes: {row['boxes']:6} | "
                f"<8px: {row['under_8px_pct']:5.1f}% | "
                f"<16px: {row['under_16px_pct']:5.1f}%"
            )

    print(f"\nResults saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
