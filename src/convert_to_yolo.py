
# This file contains the python code to convert the data from .xml files into yolo readable format 
# and save that data in a seperate folder without altering the original xml files.

# The data in the xml consists of class name , xmin , ymin , xmax , ymax etc. (these 4 are the coordinates of
# the bottom left and top right corners of the bounding boxes). However yolo cannot needs a different format
# Yolo format: class_id, x_center, y_center, box_height, box_width

from pathlib import Path
import xml.etree.ElementTree as ET

RAW_DIR = Path("data/raw/RDD2022")
OUTPUT_DIR = Path("data/processed/RDD2022_YOLO")

CLASS_MAP = {
    "D00": 0,
    "D10": 1,
    "D20": 2,
    "D40": 3,
} 

#  1. function to convert co ordinates to yolo format ( x_center , y_center , box_height , box width)

def voc_to_yolo(xmin, ymin, xmax, ymax, image_width, image_height):
    box_width = xmax - xmin
    box_height = ymax - ymin

    x_center = xmin + box_width / 2
    y_center = ymin + box_height / 2

    # Normalize to 0-1
    x_center /= image_width
    y_center /= image_height
    box_width /= image_width
    box_height /= image_height

    return x_center, y_center, box_width, box_height

#  2. function to find the image associated with xml file using xml file path as input

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

def find_image(xml_path):
    image_dir = xml_path.parents[2] / "images"

    for extension in IMAGE_EXTENSIONS:
        image_path = image_dir / f"{xml_path.stem}{extension}"

        if image_path.is_file():
            return image_path

    return None


#  3. function to get xml_root (lets us extract values from xml files) , image_path(using find_image function) , image_width and image_height(both extracted from the images directly)

from PIL import Image

def load_annotation(xml_path):
    image_path = find_image(xml_path)

    if image_path is None:
        raise FileNotFoundError(f"No matching image found for {xml_path}")

    with Image.open(image_path) as image:
        image_width, image_height = image.size 

    xml_root = ET.parse(xml_path).getroot()

    return xml_root, image_path, image_width, image_height

#  4. function to get target_boxes list from a particular image and its xml file
#  target_boxes is a list that contains  a list of (class_name , xmin , ymin , xmax , ymax) for all the classes(eg potholes,cracks etc) that are present in a given image 

def get_target_boxes(xml_root, image_width, image_height):
    target_boxes = []

    for obj in xml_root.findall(".//object"):
        class_name = obj.findtext("name", "").strip()

        if class_name not in CLASS_MAP:
            continue

        box = obj.find("bndbox")

        if box is None:
            continue

        try:
            xmin = float(box.findtext("xmin"))
            ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax" ))
            ymax = float(box.findtext("ymax"))

        except (TypeError, ValueError):
            continue

        if not (
            0 <= xmin < xmax <= image_width
            and 0 <= ymin < ymax <= image_height
        ):
            continue

        target_boxes.append(
            (class_name, xmin, ymin, xmax, ymax)
        )

    return target_boxes


# 5. function to get convert the info in target_boxes to yolo format (class_id,x_center,y_center,box_width,box_height) and store it in yolo_labels list

def convert_boxes_to_yolo(target_boxes, image_width, image_height):
    yolo_labels = []

    for class_name, xmin, ymin, xmax, ymax in target_boxes:

        x_center, y_center, box_width, box_height = voc_to_yolo(
            xmin,
            ymin,
            xmax,
            ymax,
            image_width,
            image_height,
        )

        class_id = CLASS_MAP[class_name]

        yolo_labels.append(
            (class_id, x_center, y_center, box_width, box_height)
        )

    return yolo_labels

# 6. function to create a .txt file and write information into it from yolo_labels

def save_yolo_labels(yolo_labels, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as file:
        for class_id, x_center, y_center, box_width, box_height in yolo_labels:
            file.write(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}\n"
            )

# Convert all XML annotations in the dataset to YOLO format using all the functions above and save them in new files and folders.

xml_count = 0
label_count = 0

for country_dir in sorted(RAW_DIR.iterdir()):

    if not country_dir.is_dir():
        continue

    xml_dir = country_dir / "train" / "annotations" / "xmls"

    if not xml_dir.exists():
        continue

    output_label_dir = OUTPUT_DIR / country_dir.name / "labels"

    for xml_path in sorted(xml_dir.glob("*.xml")):

        xml_root, image_path, image_width, image_height = load_annotation(xml_path)

        target_boxes = get_target_boxes(
            xml_root,
            image_width,
            image_height
        )

        yolo_labels = convert_boxes_to_yolo(
            target_boxes,
            image_width,
            image_height
        )

        output_path = output_label_dir / f"{xml_path.stem}.txt"

        save_yolo_labels(
            yolo_labels,
            output_path
        )

        xml_count += 1
        label_count += len(yolo_labels)

print("XML files converted:", xml_count)
print("YOLO annotations created:", label_count)
print("Saved to:", OUTPUT_DIR)


# code to check if all the xml files were converted into yolo labels

label_files = list(OUTPUT_DIR.rglob("*.txt"))

total_annotations = 0

for label_file in label_files:
    with open(label_file, "r") as file:
        total_annotations += sum(1 for line in file if line.strip())

print("YOLO label files:", len(label_files))
print("Total YOLO annotations:", total_annotations)

# After running the above block i got the following results:
# YOLO label files: 38385
# Total YOLO annotations: 55006

# This matches the expectations perfectly hence XML to YOLO conversion is complete