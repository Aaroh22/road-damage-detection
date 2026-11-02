
# Phase 3: Norway Crop Feasibility Audit
# Check whether an 1824x1824 lower-left crop preserves most
# Norwegian road-damage boxes and makes them larger at 640px.
# No images or labels are changed.

from pathlib import Path
from statistics import median
from collections import Counter
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/processed/main_split"

CROP_SIZE = 1824
INPUT_SIZE = 640
SPLITS = ["train", "val"]


def inspect_split(split):
    image_dir = DATA / "images" / split
    label_dir = DATA / "labels" / split

    images = sorted(image_dir.glob("Norway_*.jpg"))

    if not images:
        raise ValueError(f"No Norway images found in {split}")

    counts = Counter()
    image_sizes = Counter()
    original_sides = []
    cropped_sides = []

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.is_file():
            raise FileNotFoundError(label_path)

        with Image.open(image_path) as image:
            width, height = image.size

        image_sizes[(width, height)] += 1

        # Lower-left crop, following the RDD2022 research
        crop_width = min(CROP_SIZE, width)
        crop_height = min(CROP_SIZE, height)

        left = 0
        top = height - crop_height
        right = crop_width
        bottom = height

        original_scale = min(
            INPUT_SIZE / width,
            INPUT_SIZE / height
        )
        crop_scale = min(
            INPUT_SIZE / crop_width,
            INPUT_SIZE / crop_height
        )

        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue

            values = line.split()

            if len(values) != 5:
                raise ValueError(f"Invalid label in {label_path}")

            cls, xc, yc, bw, bh = map(float, values)

            # Convert normalized YOLO coordinates to pixels
            center_x = xc * width
            center_y = yc * height
            box_w = bw * width
            box_h = bh * height

            x1 = center_x - box_w / 2
            y1 = center_y - box_h / 2
            x2 = center_x + box_w / 2
            y2 = center_y + box_h / 2

            counts["total"] += 1

            # Check whether the box center stays inside the crop
            if not (
                left <= center_x < right
                and top <= center_y < bottom
            ):
                counts["outside"] += 1
                continue

            counts["retained"] += 1

            # Clip boxes crossing the crop boundary
            clipped_w = max(0, min(x2, right) - max(x1, left))
            clipped_h = max(0, min(y2, bottom) - max(y1, top))

            if (
                x1 < left or y1 < top
                or x2 > right or y2 > bottom
            ):
                counts["partially_clipped"] += 1
            else:
                counts["fully_inside"] += 1

            # Compare box sizes before and after cropping
            original_sides.append(
                min(box_w, box_h) * original_scale
            )
            cropped_sides.append(
                min(clipped_w, clipped_h) * crop_scale
            )

    print(f"\n===== NORWAY {split.upper()} =====")
    print(f"Images checked: {len(images)}")
    print(f"Image sizes: {dict(image_sizes)}")
    print(f"Total damage boxes: {counts['total']}")

    print(f"Boxes retained: {counts['retained']}")
    print(f"Fully inside: {counts['fully_inside']}")
    print(f"Partially clipped: {counts['partially_clipped']}")
    print(f"Outside crop: {counts['outside']}")

    if not original_sides:
        print("No boxes retained for size comparison.")
        return

    retained_pct = (
        100 * counts["retained"] / counts["total"]
    )

    before_under16 = (
        100 * sum(x < 16 for x in original_sides)
        / len(original_sides)
    )

    after_under16 = (
        100 * sum(x < 16 for x in cropped_sides)
        / len(cropped_sides)
    )

    print(f"\nBoxes retained: {retained_pct:.2f}%")
    print(
        f"Median short side before crop: "
        f"{median(original_sides):.2f}px"
    )
    print(
        f"Median short side after crop: "
        f"{median(cropped_sides):.2f}px"
    )
    print(f"Under 16px before: {before_under16:.2f}%")
    print(f"Under 16px after: {after_under16:.2f}%")


def main():
    for split in SPLITS:
        inspect_split(split)


if __name__ == "__main__":
    main()
