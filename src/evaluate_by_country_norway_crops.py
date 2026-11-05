
# This script evaluates our Norway crop-trained YOLO26s model on all 7 RDD2022 domains.
# It checks detection accuracy separately for each country using validation images.
# It saves precision, recall, and mAP results in a CSV file for comparison with our baseline.
# No training happens here.

from pathlib import Path
import csv
import yaml
from ultralytics import YOLO

from train import prepare_data_yaml, project_root


domains = [
    "China_Drone",
    "China_MotorBike",
    "Czech",
    "India",
    "Japan",
    "Norway",
    "United_States"
]


def main():
    with open(prepare_data_yaml()) as f:
        config = yaml.safe_load(f)

    val_dir = Path(config["path"]) / "images/val"

    output = project_root / "runs/country_evaluation_norway_crops"
    output.mkdir(parents=True, exist_ok=True)

    checkpoint = (
        project_root
        / "runs/YOLO26s_norway_crops_ft10/weights/best.pt"
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint missing: {checkpoint}")

    model = YOLO(str(checkpoint))
    results = []

    for domain in domains:
        images = sorted(val_dir.glob(f"{domain}_*.jpg"))

        if not images:
            raise ValueError(f"No validation images found for {domain}")

        image_list = output / f"{domain}.txt"
        image_list.write_text(
            "\n".join(str(p.resolve()) for p in images) + "\n"
        )

        config["val"] = str(image_list.resolve())

        country_yaml = output / f"{domain}.yaml"
        country_yaml.write_text(
            yaml.safe_dump(config, sort_keys=False)
        )

        print(f"\nEvaluating {domain} ({len(images)} images)...")

        metrics = model.val(
            data=str(country_yaml),
            imgsz=640,
            batch=8,
            device=0,
            plots=False,
            project=str(output),
            name=domain
        )

        results.append([
            domain,
            len(images),
            round(metrics.box.mp, 4),
            round(metrics.box.mr, 4),
            round(metrics.box.map50, 4),
            round(metrics.box.map, 4)
        ])

    csv_path = output / "country_metrics.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Domain",
            "Images",
            "Precision",
            "Recall",
            "mAP50",
            "mAP50-95"
        ])

        writer.writerows(results)

    print("\nCountry-wise evaluation complete!")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
