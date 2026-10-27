# Phase 1: Negative image audit
# For each domain (country) in the dataset , this script checks how many images are present that have empty label files(negative images) and then
# randomly picks 12 such images from each of the 7 domains. It creates a sheet for each domain that containes these 12 images, on seeing them we 
# can visally check if the roads in those images are actually undamaged or if the label files are missing.


from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent.parent

IMAGES = ROOT / "data/processed/main_split/images/train"
LABELS = ROOT / "data/processed/main_split/labels/train"
OUTPUT = ROOT / "runs/dataset_audit/negative_samples"

DOMAINS = [
    "China_Drone",
    "China_MotorBike",
    "Czech",
    "India",
    "Japan",
    "Norway",
    "United_States",
]

SAMPLES_PER_DOMAIN = 12
SEED = 42

OUTPUT.mkdir(parents=True, exist_ok=True)
rng = random.Random(SEED)


def is_negative(image_path):
    label_path = LABELS / f"{image_path.stem}.txt"

    if not label_path.exists():
        raise FileNotFoundError(f"Missing label: {label_path}")

    return not label_path.read_text().strip()


def create_sheet(images, output_path):
    tile_w, tile_h = 600, 370
    cols = 4
    rows = (len(images) + cols - 1) // cols

    sheet = Image.new(
        "RGB", (cols * tile_w, rows * tile_h), "white"
    )
    draw = ImageDraw.Draw(sheet)

    for i, path in enumerate(images):
        with Image.open(path) as img:
            img = ImageOps.contain(
                img.convert("RGB"), (tile_w - 20, tile_h - 45)
            )

            x = (i % cols) * tile_w
            y = (i // cols) * tile_h

            sheet.paste(img, (x + 10, y + 10))
            draw.text(
                (x + 10, y + tile_h - 25),
                path.name,
                fill="black"
            )

    sheet.save(output_path, quality=95)


def main():
    all_images = sorted(
        p for p in IMAGES.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    for domain in DOMAINS:
        candidates = [
            p for p in all_images
            if p.name.startswith(domain + "_")
        ]

        negatives = [
            p for p in candidates if is_negative(p)
        ]

        print(
            f"{domain}: {len(negatives)} negative images "
            f"out of {len(candidates)} total"
        )

        selected = rng.sample(
            negatives,
            min(SAMPLES_PER_DOMAIN, len(negatives))
        )

        # Skip domains with no negative images
        if not selected:
            print(f"No negative samples for {domain}. Skipping sheet.")
            continue

        output_path = OUTPUT / f"{domain}_negatives.jpg"
        create_sheet(selected, output_path)

    print(f"\nSaved visual samples to: {OUTPUT}")


if __name__ == "__main__":
    main()