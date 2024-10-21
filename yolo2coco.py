import os
import json
import shutil
from PIL import Image

from sarfusion.data.wisard import TRAIN_FOLDERS, VAL_FOLDERS, get_wisard_folders


def yolo_to_coco(datasets, output_file, output_image_dir, category_mapping):
    """
    Convert multiple YOLO datasets to a single COCO format dataset and copy all images to a single folder.

    Args:
        datasets (list): List of tuples, each containing (yolo_dir, images_dir),
                         where `yolo_dir` is the path to the YOLO labels folder,
                         and `images_dir` is the path to the images folder.
        output_file (str): Path to save the COCO JSON file.
        output_image_dir (str): Path to copy all images into a single folder.
        category_mapping (dict): A dictionary mapping YOLO class ids to COCO category ids.

    Returns:
        None
    """

    # Create output directory for images if it doesn't exist
    os.makedirs(output_image_dir, exist_ok=True)

    # Initialize COCO structure
    coco = {"images": [], "annotations": [], "categories": []}

    # Create category entries (ensure categories are added once)
    if not coco["categories"]:
        for id, name in category_mapping.items():
            coco["categories"].append({"id": id, "name": name, "supercategory": "none"})

    # Annotation and image id counters
    annotation_id = 1
    image_id = 1

    # Process each YOLO dataset
    for dataset_idx, (yolo_dir, images_dir) in enumerate(datasets):
        # Iterate over YOLO label files
        for label_file in os.listdir(yolo_dir):
            if not label_file.endswith(".txt"):
                continue

            # Get original image filename and path
            image_file = label_file.replace(".txt", ".jpg")  # Assume images are .jpg
            original_image_path = os.path.join(images_dir, image_file)

            # Skip if image doesn't exist
            if not os.path.exists(original_image_path):
                continue

            # Define a new unique filename for the copied image to avoid collisions
            new_image_file = f"dataset{dataset_idx}_{image_file}"
            new_image_path = os.path.join(output_image_dir, new_image_file)

            # Copy image to the output image directory
            shutil.copy(original_image_path, new_image_path)

            # Open image to get its width and height
            with Image.open(new_image_path) as img:
                image_width, image_height = img.size

            # Add image information to COCO JSON
            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": new_image_file,  # Use the new filename in the COCO annotation
                    "width": image_width,
                    "height": image_height,
                }
            )

            # Read YOLO label file
            label_path = os.path.join(yolo_dir, label_file)
            with open(label_path, "r") as f:
                for line in f.readlines():
                    yolo_data = line.strip().split()
                    class_id = int(yolo_data[0])
                    x_center, y_center, width, height = map(float, yolo_data[1:])

                    # Convert normalized YOLO bbox to COCO bbox
                    x_min = (x_center - width / 2) * image_width
                    y_min = (y_center - height / 2) * image_height
                    bbox_width = width * image_width
                    bbox_height = height * image_height

                    # Add annotation to COCO JSON
                    coco["annotations"].append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id,
                            "bbox": [x_min, y_min, bbox_width, bbox_height],
                            "area": bbox_width * bbox_height,
                            "iscrowd": 0,
                        }
                    )

                    annotation_id += 1

            image_id += 1

    # Write COCO JSON to file
    with open(output_file, "w") as outfile:
        json.dump(coco, outfile, indent=4)


root = "dataset/WiSARD"
folders = get_wisard_folders("vis")
train_folders = [
    os.path.join(root, folder) for folder in folders if folder in TRAIN_FOLDERS
]
val_folders = [
    os.path.join(root, folder) for folder in folders if folder in VAL_FOLDERS
]

datasets = [
    (os.path.join(folder, "labels"), os.path.join(folder, "images"))
    for folder in train_folders
]

output_file = "datasets/WiSARD_COCO/annotations/instances_train2017.json"
output_image_dir = "datasets/WiSARD_COCO/train2017"
category_mapping = {1: "stands", 2: "rests", 3: "not_defined"}
os.makedirs(output_image_dir, exist_ok=True)

yolo_to_coco(datasets, output_file, output_image_dir, category_mapping)

# Same for validation dataset

datasets = [
    (os.path.join(folder, "labels"), os.path.join(folder, "images"))
    for folder in val_folders
]

output_file = "datasets/WiSARD_COCO/annotations/instances_val2017.json"
output_image_dir = "datasets/WiSARD_COCO/val2017"
os.makedirs(output_image_dir, exist_ok=True)

yolo_to_coco(datasets, output_file, output_image_dir, category_mapping)
