
# Preview proposed Norway crops and annotation boundaries.
# Does not modify the dataset.

from pathlib import Path
from random import Random
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data/processed/main_split/images/train"
LABELS = ROOT / "data/processed/main_split/labels/train"
OUTPUT = ROOT / "runs/dataset_audit/norway_crop_previews"
OUTPUT.mkdir(parents=True, exist_ok=True)

CROP_SIZE = 1824


def get_boxes(path, width, height):
    label_path = LABELS / f"{path.stem}.txt"
    boxes = []

    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue

        cls, xc, yc, bw, bh = map(float, line.split())

        boxes.append((
            (xc - bw / 2) * width,
            (yc - bh / 2) * height,
            (xc + bw / 2) * width,
            (yc + bh / 2) * height,
        ))

    return boxes


def status(box, crop):
    x1, y1, x2, y2 = box
    left, top, right, bottom = crop

    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2

    center_inside = (
        left <= xc < right and top <= yc < bottom
    )

    fully_inside = (
        x1 >= left and y1 >= top
        and x2 <= right and y2 <= bottom
    )

    intersects = (
        min(x2, right) > max(x1, left)
        and min(y2, bottom) > max(y1, top)
    )

    if center_inside:
        return "full" if fully_inside else "partial"

    return "edge" if intersects else "outside"


def main():
    groups = {
        "edge": [],
        "partial": [],
        "full": [],
    }

    for path in sorted(IMAGES.glob("Norway_*.jpg")):
        with Image.open(path) as image:
            width, height = image.size

        crop = (
            0,
            height - min(CROP_SIZE, height),
            min(CROP_SIZE, width),
            height,
        )

        statuses = [
            status(box, crop)
            for box in get_boxes(path, width, height)
        ]

        for category in ("edge", "partial", "full"):
            if category in statuses:
                groups[category].append(path)
                break

    rng = Random(42)
    selected = []

    for category, paths in groups.items():
        for path in rng.sample(paths, min(2, len(paths))):
            selected.append((category, path))

    for category, path in selected:
        with Image.open(path) as source:
            original = source.convert("RGB")

        width, height = original.size
        crop_box = (
            0,
            height - min(CROP_SIZE, height),
            min(CROP_SIZE, width),
            height,
        )

        cropped = original.crop(crop_box)

        draw_original = ImageDraw.Draw(original)
        draw_cropped = ImageDraw.Draw(cropped)

        draw_original.rectangle(
            crop_box, outline="yellow", width=10
        )

        for box in get_boxes(path, width, height):
            kind = status(box, crop_box)

            color = {
                "full": "lime",
                "partial": "orange",
                "edge": "red",
                "outside": "red",
            }[kind]

            draw_original.rectangle(
                box, outline=color, width=8
            )

            left, top, right, bottom = crop_box
            x1, y1, x2, y2 = box

            if (
                min(x2, right) > max(x1, left)
                and min(y2, bottom) > max(y1, top)
            ):
                clipped_box = (
                    max(x1, left) - left,
                    max(y1, top) - top,
                    min(x2, right) - left,
                    min(y2, bottom) - top,
                )

                draw_cropped.rectangle(
                    clipped_box, outline=color, width=8
                )

        original.thumbnail((580, 520))
        cropped.thumbnail((580, 520))

        canvas = Image.new("RGB", (1200, 610), "white")
        canvas.paste(original, (10, 40))
        canvas.paste(cropped, (610, 40))

        draw = ImageDraw.Draw(canvas)
        draw.text((10, 15), "Original", fill="black")
        draw.text((610, 15), "Proposed crop", fill="black")
        draw.text(
            (10, 570),
            "Green: inside | Orange: clipped | Red: center outside",
            fill="black"
        )

        canvas.save(
            OUTPUT / f"{path.stem}_{category}.jpg"
        )

    print(f"Saved {len(selected)} previews to: {OUTPUT}")
    print("Image groups:", {
        key: len(value) for key, value in groups.items()
    })


if __name__ == "__main__":
    main()
