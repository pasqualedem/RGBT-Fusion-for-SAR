import os
import accelerate
from tqdm import tqdm
from torchvision import transforms
from PIL import Image, ImageDraw

from sarfusion.data.sard import YOLODataset, download_and_clean
from sarfusion.data.utils import build_preprocessor, is_annotation_valid
from sarfusion.data.wisard import MISSING_ANNOTATIONS, TEST_FOLDERS, TRAIN_FOLDERS, VAL_FOLDERS, VIS, IR, VIS_IR, generate_wisard_filelist, get_wisard_folders
from sarfusion.models import build_model
from sarfusion.utils.utils import load_yaml


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
        
        # Fix precision errors
        if xmax - xmin != crop_width:
            if (xmax - xmin) < crop_width:
                xmax += 1
            else:
                xmin += 1
        if ymax - ymin != crop_height:
            if (ymax - ymin) < crop_height:
                ymax += 1
            else:
                ymin += 1
        
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
        
        for i, data_dict in enumerate(bar):
            image = data_dict.images
            targets = data_dict.target
            cropped_images = crop_bboxes(image, targets)
            for j, (cropped_image, class_label) in enumerate(cropped_images):
                save_path = f"{output_dir}/{subset}/{i}_{j}_{class_label}.png"
                cropped_image = cropped_image.mul(255).byte().permute(1, 2, 0).numpy()
                cropped_image = Image.fromarray(cropped_image)
                cropped_image.save(save_path)
                
                
def annotate_rgb_wisard(root, model_yaml):
    accelerator = accelerate.Accelerator()
    vis = VIS + [f[0] for f in VIS_IR]
    params = load_yaml(model_yaml)
    model_params = params["model"]
    model = build_model(model_params)
    model.eval()
    model = accelerator.prepare(model)
    transform, detransform = build_preprocessor(params)
    
    
    print ("Placing -1 in all labels...")
    # Place -1 in all labels
    for subset in tqdm(os.listdir(root)):
        subset_location = f"{root}/{subset}"
        label_path = f"{subset_location}/labels"
        for label_file in os.listdir(label_path):
            with open(f"{label_path}/{label_file}", 'r') as file:
                lines = file.readlines()
            for i in range(len(lines)):
                row = lines[i].split(" ")
                row[0] = '-1'
                lines[i] = " ".join(row) 
            with open(f"{label_path}/{label_file}", 'w') as file:
                file.writelines(lines)
        
    for subset in vis:
        print(f"Processing {subset}...")
        subset_location = f"{root}/{subset}"        

        dataset = YOLODataset(subset_location, transform=transform, return_path=True)
        
        bar = tqdm(dataset, total=len(dataset))
        
        for i, data_dict in enumerate(bar):
            image = data_dict.images
            targets = data_dict.target
            path = data_dict.path
            cropped_images = crop_bboxes(image, targets)
            new_targets = []
            for (cropped_image, _), target in zip(cropped_images, targets):
                # Check if the annotation is valid

                if not is_annotation_valid(target):
                    print(f"Invalid annotation in {path}, {target}")
                    continue
                input_dict = {"pixel_values": cropped_image.unsqueeze(0).to(accelerator.device)}
                result = model(**input_dict)
                class_label = result.logits.argmax().item()
                new_targets.append((class_label, *target[1:]))
            # Replace the target file with the new one
            gt_path = path.replace("images", "labels")
            gt_path, ext = os.path.splitext(gt_path)
            gt_path += ".txt"
            with open(gt_path, 'w') as file:
                for target in new_targets:
                    file.write(" ".join(map(str, target)) + "\n")
                    
def simplify_wisard(root):
    label_map = {
        0: 0, # running
        1: 0, # walking
        2: 1, # laying_down
        3: 2, # not_defined
        4: 1, # seated
        5: 0, # stands
    }
    vis = VIS + [f[0] for f in VIS_IR]    
            
    for subset in vis:
        print(f"Processing {subset}...")
        subset_location = f"{root}/{subset}"        

        dataset = YOLODataset(subset_location, transform=lambda x: x, return_path=True)
        
        bar = tqdm(dataset, total=len(dataset))
        
        for i, data_dict in enumerate(bar):
            image = data_dict.images
            targets = data_dict.target
            path = data_dict.path
            new_targets = []
            for target in targets:
                # Check if the annotation is valid
                if not is_annotation_valid(target):
                    print(f"Invalid annotation in {path}, {target}")
                    continue
                class_label = label_map[target[0]]
                new_targets.append((class_label, *target[1:]))
            # Replace the target file with the new one
            gt_path = path.replace("images", "labels")
            gt_path, ext = os.path.splitext(gt_path)
            gt_path += ".txt"
            with open(gt_path, 'w') as file:
                for target in new_targets:
                    file.write(" ".join(map(str, target)) + "\n")
                
                
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
    for annotation in MISSING_ANNOTATIONS:
        print(f"Creating missing annotation {annotation}...")
        with open(f"{root}/{annotation}", 'w') as file:
            pass
        
    print("Generating VIS filelists...")
    folders = "vis"
    train_folders = [folder for folder in get_wisard_folders(folders) if  folder in TRAIN_FOLDERS]
    generate_wisard_filelist(root, train_folders, "train_vis.txt")
    val_folders = [folder for folder in get_wisard_folders(folders) if  folder in VAL_FOLDERS]
    generate_wisard_filelist(root, val_folders, "val_vis.txt")
    test_folders = [folder for folder in get_wisard_folders(folders) if  folder in TEST_FOLDERS]
    generate_wisard_filelist(root, test_folders, "test_vis.txt")
    
    folders = "ir"
    print("Generating IR filelists...")
    train_folders = [folder for folder in get_wisard_folders(folders) if  folder in TRAIN_FOLDERS]
    generate_wisard_filelist(root, train_folders, "train_ir.txt")
    val_folders = [folder for folder in get_wisard_folders(folders) if  folder in VAL_FOLDERS]
    generate_wisard_filelist(root, val_folders, "val_ir.txt")
    test_folders = [folder for folder in get_wisard_folders(folders) if  folder in TEST_FOLDERS]
    generate_wisard_filelist(root, test_folders, "test_ir.txt")
    
    folders = "vis_ir"
    print("Generating VIS_IR filelists...")
    train_folders = [folder for folder in get_wisard_folders(folders) if  folder in TRAIN_FOLDERS]
    generate_wisard_filelist(root, train_folders, "train_vis_ir.txt")
    val_folders = [folder for folder in get_wisard_folders(folders) if  folder in VAL_FOLDERS]
    generate_wisard_filelist(root, val_folders, "val_vis_ir.txt")
    test_folders = [folder for folder in get_wisard_folders(folders) if  folder in TEST_FOLDERS]
    generate_wisard_filelist(root, test_folders, "test_vis_ir.txt")
    
    folders = "vis_all_ir_sync"
    print("Generating VIS_ALL_IR_SYNC filelists...")
    train_folders = [folder for folder in get_wisard_folders(folders) if  folder in TRAIN_FOLDERS]
    generate_wisard_filelist(root, train_folders, "train_vis_all_ir_sync.txt")
    val_folders = [folder for folder in get_wisard_folders(folders) if  folder in VAL_FOLDERS]
    generate_wisard_filelist(root, val_folders, "val_vis_all_ir_sync.txt")
    test_folders = [folder for folder in get_wisard_folders(folders) if  folder in TEST_FOLDERS]
    generate_wisard_filelist(root, test_folders, "test_vis_all_ir_sync.txt")
    
    folders = "vis_sync_ir_all"
    print("Generating VIS_SYNC_IR_ALL filelists...")
    train_folders = [folder for folder in get_wisard_folders(folders) if  folder in TRAIN_FOLDERS]
    generate_wisard_filelist(root, train_folders, "train_vis_sync_ir_all.txt")
    val_folders = [folder for folder in get_wisard_folders(folders) if  folder in VAL_FOLDERS]
    generate_wisard_filelist(root, val_folders, "val_vis_sync_ir_all.txt")
    test_folders = [folder for folder in get_wisard_folders(folders) if  folder in TEST_FOLDERS]
    generate_wisard_filelist(root, test_folders, "test_vis_sync_ir_all.txt")
    
    folders = "all"
    print("Generating ALL filelists...")
    train_folders = [folder for folder in get_wisard_folders(folders) if  folder in TRAIN_FOLDERS]
    generate_wisard_filelist(root, train_folders, "train_all.txt")
    val_folders = [folder for folder in get_wisard_folders(folders) if  folder in VAL_FOLDERS]
    generate_wisard_filelist(root, val_folders, "val_all.txt")
    test_folders = [folder for folder in get_wisard_folders(folders) if  folder in TEST_FOLDERS]
    generate_wisard_filelist(root, test_folders, "test_all.txt")
    
    