# This script creates extra cropped images from Norway's training data
# to make small road cracks easier for YOLO to learn.
# It updates bounding boxes, skips badly cut damage, and saves
# the new images and labels separately without changing the original dataset.


from pathlib import Path
from collections import Counter
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

SOURCE = ROOT / "data/processed/main_split"
OUTPUT = ROOT / "data/processed/norway_crops"

CROP_SIZE = 1824
MIN_VISIBLE = 0.25
MIN_BOUNDARY_SIDE = 2

stats = Counter()


def process_image(image_path):
    label_path = SOURCE / "labels/train" / f"{image_path.stem}.txt"

    if not label_path.is_file():
        raise FileNotFoundError(label_path)

    lines = [
        line for line in label_path.read_text().splitlines()
        if line.strip()
    ]

    # Only create extra crops from images containing damage
    if not lines:
        stats["no_damage"] += 1
        return

    with Image.open(image_path) as image:
        width, height = image.size

        if width < CROP_SIZE or height < CROP_SIZE:
            raise ValueError(f"Image too small: {image_path}")

        # Lower-left 1824x1824 crop
        left = 0
        top = height - CROP_SIZE
        right = CROP_SIZE
        bottom = height

        new_labels = []
        clipped_count = 0

        for line in lines:
            values = line.split()

            if len(values) != 5:
                raise ValueError(f"Invalid label: {label_path}")

            cls, xc, yc, bw, bh = map(float, values)

            if int(cls) != cls or int(cls) not in range(4):
                raise ValueError(f"Invalid class: {label_path}")

            # Convert YOLO coordinates into original pixels
            x1 = (xc - bw / 2) * width
            y1 = (yc - bh / 2) * height
            x2 = (xc + bw / 2) * width
            y2 = (yc + bh / 2) * height

            original_area = (x2 - x1) * (y2 - y1)

            if original_area <= 0:
                raise ValueError(f"Invalid bounding box: {label_path}")

            # Find the part of the box inside the crop
            nx1 = max(x1, left)
            ny1 = max(y1, top)
            nx2 = min(x2, right)
            ny2 = min(y2, bottom)

            new_width = nx2 - nx1
            new_height = ny2 - ny1

            # Completely outside the crop
            if new_width <= 0 or new_height <= 0:
                continue

            fully_inside = (
                x1 >= left and y1 >= top
                and x2 <= right and y2 <= bottom
            )

            if not fully_inside:
                visible_fraction = (
                    new_width * new_height / original_area
                )

                short_side_640 = (
                    min(new_width, new_height)
                    * 640 / CROP_SIZE
                )

                # Avoid training on unusable boundary fragments.
                # Skip the entire crop rather than leave damage
                # visible without a valid annotation.
                if (
                    visible_fraction < MIN_VISIBLE
                    or short_side_640 < MIN_BOUNDARY_SIDE
                ):
                    stats["boundary_skipped"] += 1
                    return

                clipped_count += 1

            # Convert box to coordinates relative to the crop
            new_xc = ((nx1 + nx2) / 2 - left) / CROP_SIZE
            new_yc = ((ny1 + ny2) / 2 - top) / CROP_SIZE
            new_bw = new_width / CROP_SIZE
            new_bh = new_height / CROP_SIZE

            new_labels.append(
                f"{int(cls)} {new_xc:.6f} {new_yc:.6f} "
                f"{new_bw:.6f} {new_bh:.6f}"
            )

        if not new_labels:
            stats["no_retained_damage"] += 1
            return

        image_output = (
            OUTPUT / "images/train"
            / f"NorwayCrop_{image_path.stem}.jpg"
        )

        label_output = (
            OUTPUT / "labels/train"
            / f"NorwayCrop_{image_path.stem}.txt"
        )

        image.crop((left, top, right, bottom)).convert(
            "RGB"
        ).save(image_output, quality=92)

        label_output.write_text("\n".join(new_labels) + "\n")

        stats["created"] += 1
        stats["boxes"] += len(new_labels)
        stats["clipped_boxes"] += clipped_count


def main():
    (OUTPUT / "images/train").mkdir(
        parents=True, exist_ok=True
    )
    (OUTPUT / "labels/train").mkdir(
        parents=True, exist_ok=True
    )

    images = sorted(
        (SOURCE / "images/train").glob("Norway_*.jpg")
    )

    for image_path in images:
        process_image(image_path)

    print("\n===== NORWAY CROP GENERATION =====")
    print(f"Original images checked: {len(images)}")
    print(f"Cropped images created: {stats['created']}")
    print(f"New bounding boxes: {stats['boxes']}")
    print(f"Clipped boxes retained: {stats['clipped_boxes']}")
    print(f"Empty images skipped: {stats['no_damage']}")
    print(f"Boundary-problem crops skipped: {stats['boundary_skipped']}")
    print(f"No retained damage: {stats['no_retained_damage']}")
    print(f"\nSaved to: {OUTPUT}")


if __name__ == "__main__":
    main()
