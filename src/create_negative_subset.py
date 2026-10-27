
# Experiment 4: Reduced Negative Images
# Keep all positive training images and limit negative images
# to approximately 10% per domain.
# Save a reproducible training image list without modifying the dataset.

from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data/processed/main_split"
IMAGES = DATA / "images/train"
LABELS = DATA / "labels/train"
OUTPUT = DATA / "train_neg10.txt"

DOMAINS = [
    "China_Drone",
    "China_MotorBike",
    "Czech",
    "India",
    "Japan",
    "Norway",
    "United_States",
]

NEGATIVE_FRACTION = 0.10
SEED = 42

rng = random.Random(SEED)


def is_negative(image_path):
    label_path = LABELS / f"{image_path.stem}.txt"

    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label: {label_path}")

    return not label_path.read_text().strip()


def main():
    all_images = sorted(
        p for p in IMAGES.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    selected_images = []
    total_positive = 0
    total_negative = 0

    for domain in DOMAINS:
        images = [
            p for p in all_images
            if p.name.startswith(domain + "_")
        ]

        positives = [p for p in images if not is_negative(p)]
        negatives = [p for p in images if is_negative(p)]

        max_negatives = round(
            len(positives) * NEGATIVE_FRACTION
            / (1 - NEGATIVE_FRACTION)
        )

        keep_count = min(len(negatives), max_negatives)
        kept_negatives = rng.sample(negatives, keep_count)

        selected_images.extend(positives)
        selected_images.extend(kept_negatives)

        total_positive += len(positives)
        total_negative += len(kept_negatives)

        print(
            f"{domain}: "
            f"{len(positives)} positives, "
            f"{keep_count}/{len(negatives)} negatives retained"
        )

    if len(all_images) != 30704:
        raise ValueError("Original training image count has changed.")

    if total_positive != 18984:
        raise ValueError("Unexpected positive image count.")

    selected_images.sort()

    OUTPUT.write_text(
        "\n".join(
            f"./images/train/{p.name}"
            for p in selected_images
        ) + "\n"
    )

    total = len(selected_images)

    print(f"\nOriginal training images: {len(all_images)}")
    print(f"Selected training images: {total}")
    print(f"Positive images: {total_positive}")
    print(f"Negative images: {total_negative}")
    print(f"Negative percentage: {100 * total_negative / total:.2f}%")
    print(f"Saved training list: {OUTPUT}")


if __name__ == "__main__":
    main()
