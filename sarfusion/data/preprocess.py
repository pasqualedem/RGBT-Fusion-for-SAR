import os
from tqdm import tqdm
from torchvision import transforms
from PIL import Image, ImageDraw

from sarfusion.data.sard import YOLODataset, download_and_clean

NO_LABELS = [
    "210812_Hannegan_Enterprise_VIS_0055",
    "210924_FHL_Enterprise_VIS_0564",
    "210924_FHL_Enterprise_VIS_0566",
    "210924_FHL_Enterprise_IR_0410",
    "210924_FHL_Enterprise_VIS_0409",
    "210812_Hannegan_Enterprise_IR_0056",
    "210529_Carnation_Enterprise_IR_0026",
    "210812_Hannegan_Enterprise_IR_0054",
    "210812_Hannegan_Enterprise_VIS_0053",
    "210924_FHL_Enterprise_VIS_0403",
    "210924_FHL_Enterprise_IR_0408",
    "210924_FHL_Enterprise_IR_0127",
]


def crop_bboxes(image, targets, crop_size=224):
    crop_width = crop_height = crop_size
    cropped_images = []
    # Pad the image to make sure the bounding box is not out of the image
    image_width, image_height = image.size(2), image.size(1)
    image = transforms.Pad(padding=crop_size)(image)
    for target in targets:
        class_label, x_center, y_center, width, height = target
        
        # Calculate the center of the bounding box
        bbox_center_x = x_center * image_width + crop_width
        bbox_center_y = y_center * image_height + crop_height
        
        # Calculate the crop region
        xmin = round(bbox_center_x - crop_width / 2)
        xmax = round(bbox_center_x + crop_width / 2)
        ymin = round(bbox_center_y - crop_height / 2)
        ymax = round(bbox_center_y + crop_height / 2)        
        
        # Crop the image
        cropped_image = image[:, ymin:ymax, xmin:xmax]        
        cropped_images.append((cropped_image, class_label))
    
    return cropped_images


def generate_pose_classification_dataset(output_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset_location = download_and_clean()
    os.makedirs(output_dir, exist_ok=True)
    
    for subset in ['train', 'valid', 'test']:
        print(f"Processing {subset}...")
        os.makedirs(f"{output_dir}/{subset}", exist_ok=True)
        subset_location = f"{dataset_location}/{subset}"

        dataset = YOLODataset(subset_location, transform=transform)
        
        bar = tqdm(dataset, total=len(dataset))
        
        for i, (image, targets) in enumerate(bar):
            cropped_images = crop_bboxes(image, targets)
            for j, (cropped_image, class_label) in enumerate(cropped_images):
                save_path = f"{output_dir}/{subset}/{i}_{j}_{class_label}.png"
                cropped_image = cropped_image.mul(255).byte().permute(1, 2, 0).numpy()
                cropped_image = Image.fromarray(cropped_image)
                cropped_image.save(save_path)
                
                
def wisard_to_yolo_dataset(root):
    subfolders = [os.path.join(root, f) for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
    print(f"Found {len(subfolders)} subfolders.")
    for subfolder in subfolders:
        print(f"Processing {subfolder}...")
        os.makedirs(f"{subfolder}/images", exist_ok=True)
        os.makedirs(f"{subfolder}/labels", exist_ok=True)
        for image in tqdm(os.listdir(subfolder)):
            ext = os.path.splitext(image)[-1]
            if ext.lower() in ['.jpg', '.jpeg', '.png']:
                image_path = os.path.join(subfolder, image)
                label_path = image_path.replace(ext, ".txt")
                os.rename(image_path, f"{subfolder}/images/{image}")
                if os.path.exists(label_path):
                    os.rename(label_path, f"{subfolder}/labels/{image.replace(ext, '.txt')}") 