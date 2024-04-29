import os

from roboflow import Roboflow
from PIL import Image
from torch.utils.data import Dataset

from sarfusion.data.utils import DataDict


def download_and_clean():
    api = os.environ.get("ROBOFLOW_API_KEY")
    rf = Roboflow(api_key=api)
    project = rf.workspace("sard").project("sardd")
    version = project.version(1)
    dataset = version.download("yolov9", location="dataset/sard")
    
    for subset in ["train", "valid", "test"]:
        subset_path = os.path.join(dataset.location, subset)
        types = ["images", "labels"]
        for t in types:
            subset_t = os.path.join(subset_path, t)
            subset_t_files = os.listdir(subset_t)
            for file in subset_t_files:
                parts = file.split(".")
                if "rf" in parts:
                    new_name = parts[0] + "." + parts[-1]
                    os.rename(os.path.join(subset_t, file), os.path.join(subset_t, new_name))
    
    return dataset.location


class YOLODataset(Dataset):
    def __init__(self, root, transform=None, return_path=False):
        image_path = os.path.join(root, "images")
        annotation_path = os.path.join(root, "labels")
        annotations = os.listdir(annotation_path)
        images = os.listdir(image_path)
        self.annotation_paths = sorted([os.path.join(annotation_path, ann) for ann in annotations])
        self.image_paths = sorted([os.path.join(image_path, img) for img in images])
        for i in range(len(self.annotation_paths)):
            assert self.annotation_paths[i].split("/")[-1].split(".")[0] == self.image_paths[i].split("/")[-1].split(".")[0]
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Load annotations
        with open(annotation_path, 'r') as file:
            annotations = file.readlines()
        
        # Parse annotations
        targets = []
        for annotation in annotations:
            annotation = annotation.strip().split()
            class_label = int(annotation[0])
            x_center, y_center, width, height = map(float, annotation[1:])
            targets.append([class_label, x_center, y_center, width, height])

        if self.transform:
            img = self.transform(img)
        data_dict = {
            DataDict.IMAGES: img,
            DataDict.TARGET: targets
        }
        if self.return_path:
            data_dict[DataDict.PATH] = img_path

        return data_dict
    
    
class PoseClassificationDataset(Dataset):
    def __init__(self, root, transform=None, return_path=False):
        self.image_paths = [os.path.join(root, img) for img in os.listdir(root)]
        self.transform = transform
        self.return_path = return_path
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        cls = int(img_path.split("_")[-1].split(".")[0])
        
        if self.transform:
            img = self.transform(img)
        
        data_dict = {
            DataDict.IMAGES: img,
            DataDict.TARGET: cls
        }
        if self.return_path:
            data_dict[DataDict.PATH] = img_path
            
        return data_dict