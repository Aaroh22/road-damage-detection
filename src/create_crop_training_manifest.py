
# This script combines the original training images with our new Norway crops.
# It checks that the cropped images have valid YOLO labels.
# It will create a .txt file containing every training image path for YOLO training
# will contain roughly 30k original training images and additional 2k cropped training
# images from Norway domain.

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/processed"
ORIGINAL = DATA / "main_split"
CROPS = DATA / "norway_crops"

OUTPUT = ORIGINAL / "train_with_norway_crops.txt"


def get_images(folder):
    return sorted(
        path for path in folder.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def check_labels(images, label_dir, check_boxes=False):
    total_boxes = 0

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.is_file():
            raise FileNotFoundError(label_path)

        if not check_boxes:
            continue

        lines = [
            line for line in label_path.read_text().splitlines()
            if line.strip()
        ]

        if not lines:
            raise ValueError(f"Empty crop label: {label_path}")

        for line in lines:
            values = line.split()

            if len(values) != 5:
                raise ValueError(f"Invalid label: {label_path}")

            cls, x, y, w, h = map(float, values)

            if cls not in (0, 1, 2, 3):
                raise ValueError(f"Invalid class: {label_path}")

            if not (
                0 < w <= 1 and 0 < h <= 1
                and x - w / 2 >= -0.00001
                and y - h / 2 >= -0.00001
                and x + w / 2 <= 1.00001
                and y + h / 2 <= 1.00001
            ):
                raise ValueError(f"Invalid box: {label_path}")

            total_boxes += 1

    return total_boxes


def main():
    original_images = get_images(ORIGINAL / "images/train")
    cropped_images = get_images(CROPS / "images/train")

    if len(original_images) != 30704:
        raise ValueError("Original training image count changed.")

    if not cropped_images:
        raise ValueError("No Norway crops found.")

    check_labels(
        original_images,
        ORIGINAL / "labels/train"
    )

    cropped_boxes = check_labels(
        cropped_images,
        CROPS / "labels/train",
        check_boxes=True
    )

    all_images = original_images + cropped_images

    OUTPUT.write_text(
        "\n".join(str(path.resolve()) for path in all_images)
        + "\n"
    )

    print("\n===== COMBINED TRAINING DATA =====")
    print(f"Original images: {len(original_images)}")
    print(f"Norway crops: {len(cropped_images)}")
    print(f"Total training images: {len(all_images)}")
    print(f"Cropped damage boxes: {cropped_boxes}")
    print("Validation and test sets: unchanged")
    print(f"\nSaved to: {OUTPUT}")


if __name__ == "__main__":
    main()
