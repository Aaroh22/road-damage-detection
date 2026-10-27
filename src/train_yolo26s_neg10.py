
# Experiment 4: Reduced Negative Images
# Train YOLO26s using all positive images and fewer negative images.
# Test whether reducing background images improves overall accuracy.
# Original train, validation, and test folders remain unchanged.

from pathlib import Path
import yaml
from ultralytics import YOLO

from train import prepare_data_yaml, project_root
from create_negative_subset import main as create_subset


RUN_NAME = "YOLO26s_neg10_full50"
RESUME = False


def main():
    # Recreate the same training subset using seed 42
    create_subset()

    # Prepare the original dataset configuration
    original_yaml = prepare_data_yaml()

    with open(original_yaml, "r") as file:
        data = yaml.safe_load(file)

    # Change only the training source
    data["train"] = "train_neg10.txt"

    # Save a separate YAML for this experiment
    experiment_yaml = project_root / "data_neg10_temp.yaml"

    with open(experiment_yaml, "w") as file:
        yaml.safe_dump(data, file, sort_keys=False)

    # Confirm the training subset is correct
    subset_file = (
        project_root
        / "data/processed/main_split/train_neg10.txt"
    )

    selected_count = len(subset_file.read_text().splitlines())

    if selected_count != 20532:
        raise ValueError(
            f"Expected 20532 training images, got {selected_count}"
        )

    print(f"Training images: {selected_count}")
    print("Validation and test sets remain unchanged.")

    # Resume from the last saved checkpoint if needed
    if RESUME:
        checkpoint = (
            project_root / "runs"
            / RUN_NAME / "weights" / "last.pt"
        )

        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

        model = YOLO(str(checkpoint))
        return model.train(
            resume=True,
            data=str(experiment_yaml)
        )

    # Start a fresh controlled experiment
    model = YOLO("yolo26s.pt")

    return model.train(
        data=str(experiment_yaml),
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        seed=42,
        project=str(project_root / "runs"),
        name=RUN_NAME
    )


if __name__ == "__main__":
    main()
