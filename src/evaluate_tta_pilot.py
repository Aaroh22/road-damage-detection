
# Phase 2: Test-Time Augmentation Evaluation
# Compare normal YOLO26s inference against TTA on the same
# 40 Norway validation images used in the SAHI experiment.
# Measure TP, FP, FN, precision, recall, and inference speed.
# No training or dataset modifications.

from pathlib import Path
from time import perf_counter
import csv

from ultralytics import YOLO
from evaluate_sahi_pilot import (
    select_images,
    get_ground_truth,
    calculate_iou,
)

ROOT = Path(__file__).resolve().parent.parent

WEIGHTS = ROOT / "runs/YOLO26s_baseline_model/weights/best.pt"
OUTPUT = ROOT / "runs/tta_evaluation/comparison.csv"

CONFIDENCE = 0.25
IOU_THRESHOLD = 0.5


def evaluate_predictions(ground_truth, result):
    predictions = []

    for box in result.boxes:
        predictions.append((
            int(box.cls.item()),
            box.xyxy[0].tolist(),
            float(box.conf.item())
        ))

    predictions.sort(key=lambda p: p[2], reverse=True)

    matched_gt = set()
    tp = 0

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
            tp += 1

    fp = len(predictions) - tp
    fn = len(ground_truth) - tp

    return tp, fp, fn


def main():
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)

    model = YOLO(str(WEIGHTS))
    selected = select_images()

    def predict(image_path, augment):
        return model.predict(
            source=str(image_path),
            imgsz=640,
            conf=CONFIDENCE,
            device="cpu",
            augment=augment,
            nms=True,
            verbose=False
        )[0]

    # Warm up both inference methods
    predict(selected[0][0], augment=False)
    predict(selected[0][0], augment=True)

    rows = []

    for number, (image_path, image_type) in enumerate(
        selected, start=1
    ):
        ground_truth = get_ground_truth(image_path)

        for method in ("normal", "tta"):
            start = perf_counter()

            result = predict(
                image_path,
                augment=(method == "tta")
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

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== FINAL TTA COMPARISON =====")

    for method in ("normal", "tta"):
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

        negative_fp = sum(
            row["fp"] for row in method_rows
            if row["image_type"] == "negative"
        )

        print(f"\nMethod: {method.upper()}")
        print(f"True positives: {tp}")
        print(f"False positives: {fp}")
        print(f"False negatives: {fn}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"False positives on negative images: {negative_fp}")
        print(
            f"Average time/image: "
            f"{total_time / len(method_rows):.3f}s"
        )

    print(f"\nResults saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
