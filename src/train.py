from pathlib import Path

import yaml
from ultralytics import YOLO

project_root = Path(__file__).resolve().parent.parent
Data_yaml = project_root / "data.yaml"

# Set the dataset path correctly for the current environment before training.
# (Read the original data.yaml, convert the dataset path to an absolute path,
# and save it in a separate YAML file so training works on both Mac and Colab
# without changing the original configuration)

def prepare_data_yaml():
    with Data_yaml.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    dataset_path = project_root / data["path"]

    if not dataset_path.is_dir():
       raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    data["path"] = str(dataset_path)

    data_temp_yaml = project_root / "data_temp.yaml"
    with data_temp_yaml.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)

    return data_temp_yaml


def main():
    # Lets me resume an interrupeted training run
    RESUME = False

    if RESUME:
        checkpoint = (
            project_root / "runs"
            / "YOLO26s_baseline_model"
            / "weights" / "last.pt"
        )

        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        data_temp_yaml = prepare_data_yaml()

        print(f"Resuming training from {checkpoint}")
        model = YOLO(str(checkpoint))
        return model.train(resume=True, data=str(data_temp_yaml))

    print("Starting a new training run.")
    
    data_temp_yaml = prepare_data_yaml()

    # Add the baseline model : for the baseline model i am choosing s model(small).
    model = YOLO("yolo26s.pt")

    return model.train(
        data=str(data_temp_yaml),
        epochs=50,        
        imgsz=640,        
        batch=8,          
        device=0,         # "cpu", "mps", or GPU index such as 0
        seed=42,
        project=str(project_root/"runs"),        # Output directory (str)
        name="YOLO26s_baseline_model",           # Training run name (str)
    )


if __name__ == "__main__":
    main()
