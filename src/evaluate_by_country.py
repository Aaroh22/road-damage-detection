
from pathlib import Path
import csv
import yaml
from ultralytics import YOLO
from train import prepare_data_yaml, project_root

domains = [
    "China_Drone", "China_MotorBike", "Czech",
    "India", "Japan", "Norway", "United_States"
]

# Evaluate our trained YOLO model separately on each of the 7 RDD2022 domains (evaluation by countries).
# For each domain, create a .txt file containing the list of its validation image paths
# and a YAML file telling YOLO to use only those images.
# Calculate precision, recall, and mAP, seperately for each country then save all results in one CSV.
# No model training happens here.


def main():
    with open(prepare_data_yaml()) as f:
        config = yaml.safe_load(f)

    val_dir = Path(config["path"]) / "images/val"
    output = project_root / "runs/country_evaluation"
    output.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(project_root / "runs/YOLO26s_baseline_model/weights/best.pt"))
    results = []

    for domain in domains:
        images = sorted(val_dir.glob(f"{domain}_*.jpg"))

        if not images:
            raise ValueError(f"No images found for {domain}")

        image_list = output / f"{domain}.txt"
        image_list.write_text("\n".join(str(p.resolve()) for p in images) + "\n")

        config["val"] = str(image_list.resolve())
        country_yaml = output / f"{domain}.yaml"
        country_yaml.write_text(yaml.safe_dump(config))

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
            domain, len(images),
            round(metrics.box.mp, 4),
            round(metrics.box.mr, 4),
            round(metrics.box.map50, 4),
            round(metrics.box.map, 4)
        ])

    with (output / "country_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Domain", "Images", "Precision", "Recall", "mAP50", "mAP50-95"])
        writer.writerows(results)

    print("Country-wise evaluation complete!")

if __name__ == "__main__":
    main()
