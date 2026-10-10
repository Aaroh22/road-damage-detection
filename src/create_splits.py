# This file contains the python code to split the training data(both images and yolo labels) into Train ,test and validation sets 
# country by country and save them into new folders.


from pathlib import Path
import random
import shutil

RAW_DIR = Path("data/raw/RDD2022")
LABEL_DIR = Path("data/processed/RDD2022_YOLO")

SPLIT_DIR = Path("data/processed/main_split")

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42

random.seed(SEED) # Just remember that this enables us to reproduce the same test train validation splits each time.

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Find the matching image for each Yolo label file in one country's dataset and return the image and label pairs, or raise an error if an image is missing.

def get_image_label_pairs(country_name):
    image_dir = RAW_DIR / country_name / "train" / "images"
    label_dir = LABEL_DIR / country_name / "labels"

    pairs = []

    for label_path in sorted(label_dir.glob("*.txt")):

        image_path = None

        for extension in IMAGE_EXTENSIONS:
            candidate = image_dir / f"{label_path.stem}{extension}"

            if candidate.is_file():
                image_path = candidate
                break

        if image_path is None:
            raise FileNotFoundError(
                f"No matching image found for {label_path}"
            )

        pairs.append((image_path, label_path))

    return pairs

# Shuffle a copy of the pairs and then divide them into training, validation, and test set(Use 80% for training, 10% for validation and 10% pairs for testing).

def split_pairs(pairs):
    pairs = pairs.copy()  # creating copy
    random.shuffle(pairs) # shuffling

    total = len(pairs)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    return train_pairs, val_pairs, test_pairs

# Create the folders for a split and copy its images and labels into them. Keep each file's original name so images still match their labels.

def copy_pairs(pairs, split_name):
    image_output_dir = SPLIT_DIR / "images" / split_name
    label_output_dir = SPLIT_DIR / "labels" / split_name

    image_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    for image_path, label_path in pairs:
        shutil.copy2(
            image_path,
            image_output_dir / image_path.name
        )

        shutil.copy2(
            label_path,
            label_output_dir / label_path.name
        )

# Remove an old split before creating a fresh one
if SPLIT_DIR.exists():
    shutil.rmtree(SPLIT_DIR)

total_train = 0
total_val = 0
total_test = 0

# For each country run the above 3 functions and print the counts

for country_dir in sorted(RAW_DIR.iterdir()):

    if not country_dir.is_dir():
        continue

    country_name = country_dir.name

    pairs = get_image_label_pairs(country_name)

    train_pairs, val_pairs, test_pairs = split_pairs(pairs)

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")
    copy_pairs(test_pairs, "test")

    total_train += len(train_pairs)
    total_val += len(val_pairs)
    total_test += len(test_pairs)

    print(
        f"{country_name}: "
        f"train={len(train_pairs)}, "
        f"val={len(val_pairs)}, "
        f"test={len(test_pairs)}"
    )

print("\nFinal split:")
print("Train:", total_train)
print("Validation:", total_val)
print("Test:", total_test)
print("Total:", total_train + total_val + total_test)

# Check that each image has a matching label in every split.
# Print the counts and report any missing images or labels.

for split_name in ("train", "val", "test"):
    image_dir = SPLIT_DIR / "images" / split_name
    label_dir = SPLIT_DIR / "labels" / split_name

    image_stems = {path.stem for path in image_dir.iterdir()}
    label_stems = {path.stem for path in label_dir.glob("*.txt")}

    print(f"\n{split_name.upper()}")
    print("Images:", len(image_stems))
    print("Labels:", len(label_stems))
    print("Missing labels:", len(image_stems - label_stems))
    print("Missing images:", len(label_stems - image_stems))
