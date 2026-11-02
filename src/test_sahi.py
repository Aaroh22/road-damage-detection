
# Phase 2: Sliced Inference Pilot
# Compare normal YOLO26s predictions with SAHI sliced predictions.
# Test 3 Norway validation images containing small damage boxes
# and 1 negative image.
# No training or dataset modifications are performed.

from pathlib import Path
from time import perf_counter

from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction

ROOT = Path(__file__).resolve().parent.parent

IMAGES = ROOT / "data/processed/main_split/images/val"
LABELS = ROOT / "data/processed/main_split/labels/val"

WEIGHTS = (
    ROOT / "runs/YOLO26s_baseline_model/weights/best.pt"
)

OUTPUT = ROOT / "runs/sahi_pilot"
OUTPUT.mkdir(parents=True, exist_ok=True)


def select_images():
    small_images = []
    negative_images = []

    for path in sorted(IMAGES.glob("Norway_*.jpg")):
        label_path = LABELS / f"{path.stem}.txt"
        lines = [
            line for line in label_path.read_text().splitlines()
            if line.strip()
        ]

        if not lines:
            negative_images.append(path)
            continue

        with Image.open(path) as image:
            width, height = image.size

        scale = min(640 / width, 640 / height)

        # Find images containing boxes with a side below 16px
        for line in lines:
            cls, x, y, w, h = map(float, line.split())

            short_side = min(
                w * width * scale,
                h * height * scale
            )

            if short_side < 16:
                small_images.append(path)
                break

    if len(small_images) < 3 or not negative_images:
        raise ValueError("Not enough images for the pilot.")

    return small_images[:3] + negative_images[:1]


def main():
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(WEIGHTS),
        confidence_threshold=0.25,
        image_size=640,
        device="cpu"
    )

    images = select_images()

    for image_path in images:
        print(f"\nImage: {image_path.name}")

        # Normal inference
        start = perf_counter()
        normal = get_prediction(
            str(image_path), model
        )
        normal_time = perf_counter() - start

        # Sliced inference
        start = perf_counter()
        sliced = get_sliced_prediction(
            str(image_path),
            model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            perform_standard_pred=False,
            verbose=0
        )
        sliced_time = perf_counter() - start

        normal.export_visuals(
            export_dir=str(OUTPUT / "normal"),
            file_name=image_path.stem
        )

        sliced.export_visuals(
            export_dir=str(OUTPUT / "sliced"),
            file_name=image_path.stem
        )

        print(
            f"Normal: {len(normal.object_prediction_list)} "
            f"detections | {normal_time:.2f}s"
        )

        print(
            f"Sliced: {len(sliced.object_prediction_list)} "
            f"detections | {sliced_time:.2f}s"
        )

    print(f"\nResults saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
