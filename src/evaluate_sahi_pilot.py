
# Phase 2: Quantitative SAHI Evaluation
# Compare normal YOLO26s and sliced inference on 40 fixed
# Norway validation images using ground-truth annotations.
# Measure correct detections, missed damage, false positives,
# precision, recall, and inference time.
# No training or dataset modifications.

from pathlib import Path
from time import perf_counter
import csv
import random

from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction

ROOT = Path(__file__).resolve().parent.parent

IMAGES = ROOT / "data/processed/main_split/images/val"
LABELS = ROOT / "data/processed/main_split/labels/val"

WEIGHTS = ROOT / "runs/YOLO26s_baseline_model/weights/best.pt"
OUTPUT = ROOT / "runs/sahi_evaluation"
OUTPUT.mkdir(parents=True, exist_ok=True)

SEED = 42
CONFIDENCE = 0.25
IOU_THRESHOLD = 0.5

POSITIVE_SAMPLES = 25
NEGATIVE_SAMPLES = 15


def get_ground_truth(image_path):
    label_path = LABELS / f"{image_path.stem}.txt"

    if not label_path.is_file():
        raise FileNotFoundError(label_path)

    with Image.open(image_path) as image:
        width, height = image.size

    boxes = []

    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue

        cls, x, y, w, h = map(float, line.split())

        boxes.append((
            int(cls),
            [
                (x - w / 2) * width,
                (y - h / 2) * height,
                (x + w / 2) * width,
                (y + h / 2) * height,
            ]
        ))

    return boxes


def select_images():
    positives = []
    negatives = []

    for path in sorted(IMAGES.glob("Norway_*.jpg")):
        gt = get_ground_truth(path)

        if not gt:
            negatives.append(path)
            continue

        with Image.open(path) as image:
            width, height = image.size

        scale = min(640 / width, 640 / height)

        # Select positive images containing thin bounding boxes
        if any(
            min(
                (box[2] - box[0]) * scale,
                (box[3] - box[1]) * scale
            ) < 16
            for _, box in gt
        ):
            positives.append(path)

    if len(positives) < POSITIVE_SAMPLES:
        raise ValueError("Not enough small-damage images.")

    if len(negatives) < NEGATIVE_SAMPLES:
        raise ValueError("Not enough negative images.")

    rng = random.Random(SEED)

    selected_positives = rng.sample(
        positives, POSITIVE_SAMPLES
    )
    selected_negatives = rng.sample(
        negatives, NEGATIVE_SAMPLES
    )

    return [
        (path, "positive") for path in selected_positives
    ] + [
        (path, "negative") for path in selected_negatives
    ]


def calculate_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0


def evaluate_predictions(ground_truth, result):
    predictions = [
        (
            int(pred.category.id),
            pred.bbox.to_xyxy(),
            pred.score.value
        )
        for pred in result.object_prediction_list
    ]

    # Evaluate higher-confidence predictions first
    predictions.sort(key=lambda p: p[2], reverse=True)

    matched_gt = set()
    true_positives = 0

    for pred_class, pred_box, score in predictions:
        best_iou = IOU_THRESHOLD
        best_index = None

        for i, (gt_class, gt_box) in enumerate(ground_truth):
            if i in matched_gt or pred_class != gt_class:
                continue

            overlap = calculate_iou(pred_box, gt_box)

            if overlap >= best_iou:
                best_iou = overlap
                best_index = i

        if best_index is not None:
            matched_gt.add(best_index)
            true_positives += 1

    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - true_positives

    return true_positives, false_positives, false_negatives


def main():
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(WEIGHTS),
        confidence_threshold=CONFIDENCE,
        image_size=640,
        device="cpu"
    )

    selected = select_images()

    # Warm up the model before measuring inference time
    get_prediction(str(selected[0][0]), model)

    rows = []

    for number, (image_path, image_type) in enumerate(
        selected, start=1
    ):
        ground_truth = get_ground_truth(image_path)

        for method in ("normal", "sahi"):
            start = perf_counter()

            if method == "normal":
                result = get_prediction(
                    str(image_path), model
                )
            else:
                result = get_sliced_prediction(
                    str(image_path),
                    model,
                    slice_height=640,
                    slice_width=640,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2,
                    perform_standard_pred=False,
                    verbose=0
                )

            elapsed = perf_counter() - start

            tp, fp, fn = evaluate_predictions(
                ground_truth, result
            )

            rows.append({
                "image": image_path.name,
                "image_type": image_type,
                "method": method,
                "ground_truth": len(ground_truth),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "time_seconds": round(elapsed, 3),
            })

        print(f"Completed {number}/{len(selected)}: {image_path.name}")

    csv_path = OUTPUT / "comparison.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== FINAL COMPARISON =====")

    for method in ("normal", "sahi"):
        method_rows = [
            row for row in rows if row["method"] == method
        ]

        tp = sum(row["tp"] for row in method_rows)
        fp = sum(row["fp"] for row in method_rows)
        fn = sum(row["fn"] for row in method_rows)

        total_time = sum(
            row["time_seconds"] for row in method_rows
        )

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        negative_rows = [
            row for row in method_rows
            if row["image_type"] == "negative"
        ]

        negative_fp = sum(row["fp"] for row in negative_rows)

        print(f"\nMethod: {method.upper()}")
        print(f"True positives: {tp}")
        print(f"False positives: {fp}")
        print(f"False negatives: {fn}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"False positives on negative images: {negative_fp}")
        print(
            f"Average time/image: "
            f"{total_time / len(method_rows):.2f}s"
        )

    print(f"\nSaved detailed results to: {csv_path}")


if __name__ == "__main__":
    main()
