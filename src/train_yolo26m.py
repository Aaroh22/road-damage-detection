
# Experiment 2: Full YOLO26m training
# Train a larger YOLO model on the same dataset and settings as our
# YOLO26s baseline to see if model capacity improves overall accuracy.

from ultralytics import YOLO
from train import prepare_data_yaml, project_root


def main():
    # Prepare dataset configuration with the correct absolute paths
    data_yaml = prepare_data_yaml()

    # Load pretrained YOLO26m
    model = YOLO("yolo26m.pt")

    # Train for 50 epochs
    results = model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=8,
        seed=42,
        optimizer="auto",
        close_mosaic=10,
        val=True,
        plots=True,
        project=str(project_root / "runs"),
        name="YOLO26m_full50"
    )

    return results


if __name__ == "__main__":
    main()
