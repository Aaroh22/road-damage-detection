
# This script fine-tunes our existing YOLO26s model using the original
# training images plus the new Norwegian crops.
# It uses a lower learning rate and saves results in a separate run.
# The original dataset and validation set remain unchanged.

from pathlib import Path
import yaml
from ultralytics import YOLO

from train import prepare_data_yaml, project_root


RUN_NAME = "YOLO26s_norway_crops_ft10"
RESUME = False


def main():
    manifest = (
        project_root
        / "data/processed/main_split/train_with_norway_crops.txt"
    )

    baseline = (
        project_root
        / "runs/YOLO26s_baseline_model/weights/best.pt"
    )

    if not manifest.is_file():
        raise FileNotFoundError(f"Training list missing: {manifest}")

    paths = [
        Path(line)
        for line in manifest.read_text().splitlines()
        if line.strip()
    ]

    if len(paths) != 32845:
        raise ValueError(f"Expected 32845 images, got {len(paths)}")

    missing = next((p for p in paths if not p.is_file()), None)
    if missing:
        raise FileNotFoundError(f"Training image missing: {missing}")

    # Start with the original dataset configuration
    original_yaml = prepare_data_yaml()

    with original_yaml.open("r") as file:
        data = yaml.safe_load(file)

    # Only change the training images
    data["train"] = str(manifest.resolve())

    experiment_yaml = project_root / "data_norway_crops_temp.yaml"

    with experiment_yaml.open("w") as file:
        yaml.safe_dump(data, file, sort_keys=False)

    # Resume only if this experiment was interrupted
    if RESUME:
        checkpoint = (
            project_root / "runs"
            / RUN_NAME / "weights/last.pt"
        )

        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

        model = YOLO(str(checkpoint))

        return model.train(
            resume=True,
            data=str(experiment_yaml)
        )

    if not baseline.is_file():
        raise FileNotFoundError(f"Baseline model missing: {baseline}")

    # Fine-tune our already-trained road damage model
    model = YOLO(str(baseline))

    return model.train(
        data=str(experiment_yaml),
        epochs=10,
        imgsz=640,
        batch=8,
        device=0,
        seed=42,
        optimizer="AdamW",
        lr0=0.0005,
        project=str(project_root / "runs"),
        name=RUN_NAME
    )


if __name__ == "__main__":
    main()
