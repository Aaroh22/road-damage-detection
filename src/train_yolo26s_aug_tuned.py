

from ultralytics import YOLO
from train import prepare_data_yaml, project_root


def main():
    RESUME = False

    data_yaml = prepare_data_yaml()

    if RESUME:
        checkpoint = (
            project_root
            / "runs"
            / "YOLO26s_augmented_full50"
            / "weights"
            / "last.pt"
        )

        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {checkpoint}"
            )

        print(f"Resuming training from: {checkpoint}")

        model = YOLO(str(checkpoint))
        return model.train(
            resume=True,
            data=str(data_yaml)
        )

    # Start from pretrained YOLO26s, not our trained baseline
    model = YOLO("yolo26s.pt")

    print("Starting Experiment 3: Targeted Augmentation")

    # Changes from baseline:
    # mosaic: 1.0 -> 0.5
    # scale:  0.5 -> 0.25
    # All other training settings remain consistent with the baseline.

    results = model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        seed=42,
        mosaic=0.5,
        scale=0.25,
        project=str(project_root / "runs"),
        name="YOLO26s_augmentation_tuned_full50",
    )

    return results

if __name__ == "__main__":
    main()
