# This script helps us understand where our trained YOLO model makes mistakes.
# It randomly selects 50 validation images from each of the 7 domains,
# It runs our trained model on these images and compares its predictions
# with the actual labels using IoU to count missed detections and false positives.
# Finally, it saves the 3 images with the most errors from each domain (21 total),
# showing the actual labels (green) and predictions (orange) side by side.
# This helps us identify weaknesses and decide how to improve our model.
# No training happens here.




from pathlib import Path
import random
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data/processed/main_split/images/val"
LABELS = ROOT / "data/processed/main_split/labels/val"

# Separate folder so Round 1 results remain untouched
OUTPUT = ROOT / "runs/error_analysis_round2"
OUTPUT.mkdir(parents=True, exist_ok=True)

DOMAINS = [
    "China_Drone", "China_MotorBike", "Czech",
    "India", "Japan", "Norway", "United_States"
]

model = YOLO(
    str(ROOT / "runs/YOLO26s_baseline_model/weights/best.pt")
)

# Reproduce the original 12 images, then exclude them
rng = random.Random(42)

# Different seed for the new 50-image samples
new_rng = random.Random(2026)


def read_labels(image_path, width, height):
    label_path = LABELS / f"{image_path.stem}.txt"
    boxes = []

    for line in label_path.read_text().splitlines():
        cls, x, y, w, h = map(float, line.split())

        boxes.append((
            int(cls),
            [
                (x - w / 2) * width,
                (y - h / 2) * height,
                (x + w / 2) * width,
                (y + h / 2) * height
            ]
        ))

    return boxes


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    return intersection / (area_a + area_b - intersection + 1e-6)


def count_errors(ground_truth, predictions):
    possible_matches = []

    for i, (gt_class, gt_box) in enumerate(ground_truth):
        for j, (pred_class, pred_box) in enumerate(predictions):

            overlap = iou(gt_box, pred_box)

            if gt_class == pred_class and overlap >= 0.5:
                possible_matches.append((overlap, i, j))

    matched_gt = set()
    matched_pred = set()

    for _, i, j in sorted(possible_matches, reverse=True):
        if i not in matched_gt and j not in matched_pred:
            matched_gt.add(i)
            matched_pred.add(j)

    missed = len(ground_truth) - len(matched_gt)
    false_positives = len(predictions) - len(matched_pred)

    return missed, false_positives


def draw_boxes(image, boxes, color):
    result = image.copy()

    for cls, box in boxes:
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(
            result, (x1, y1), (x2, y2), color, 2
        )

        cv2.putText(
            result,
            f"Class {cls}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return result


for domain in DOMAINS:
    images = sorted(IMAGES.glob(f"{domain}_*.jpg"))

    # Reconstruct exactly which 12 images Round 1 used
    original = set(rng.sample(images, min(12, len(images))))

    # Exclude the original images
    remaining = [
        image for image in images if image not in original
    ]

    # Select 50 completely new images
    selected = new_rng.sample(
        remaining, min(50, len(remaining))
    )

    cases = []

    for image_path in selected:
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]

        ground_truth = read_labels(
            image_path, width, height
        )

        result = model.predict(
            source=image,
            imgsz=640,
            conf=0.25,
            device="cpu",
            verbose=False
        )[0]

        predictions = [
            (int(box.cls.item()), box.xyxy[0].tolist())
            for box in result.boxes
        ]

        missed, false_positives = count_errors(
            ground_truth, predictions
        )

        cases.append((
            missed + false_positives,
            image_path,
            image,
            ground_truth,
            predictions,
            missed,
            false_positives
        ))

    # Highest-error images first
    cases.sort(key=lambda case: case[0], reverse=True)

    # Save the 3 highest-error images for this domain
    for rank, case in enumerate(cases[:3], start=1):
        _, image_path, image, gt, pred, missed, fp = case

        left = draw_boxes(image, gt, (0, 255, 0))
        right = draw_boxes(image, pred, (0, 165, 255))

        left = cv2.resize(left, (640, 480))
        right = cv2.resize(right, (640, 480))

        comparison = cv2.hconcat([left, right])

        filename = OUTPUT / f"{domain}_{rank}.jpg"
        cv2.imwrite(str(filename), comparison)

        print(
            f"{domain}: {image_path.name} | "
            f"Missed: {missed}, False positives: {fp}"
        )

    print(f"Finished analyzing {len(selected)} images from {domain}")

print(f"\nRound 2 complete! Results saved in: {OUTPUT}")
