from copy import deepcopy
import os

import torch
import torchvision.transforms as T
from transformers import AutoProcessor

from sarfusion.utils.augmentations import denormalize
from sarfusion.utils.structures import DataDict


def dict_collate_fn(batch):
    d = {}
    keys = batch[0].keys()
    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            d[key] = torch.stack([sample[key] for sample in batch])
        elif isinstance(batch[0][key], int) or isinstance(batch[0][key], float):
            d[key] = torch.stack([torch.tensor(sample[key]) for sample in batch])
        elif isinstance(batch[0][key], list):
            d[key] = [sample[key] for sample in batch]
        else:
            raise ValueError(f"Unsupported type {type(batch[0][key])}")

    return d


def get_collate_fn(dataset):
    if hasattr(dataset, "collate_fn"):
        return dataset.collate_fn
    return dict_collate_fn


def build_preprocessor(params):
    params = deepcopy(params)
    preprocessor_params = params.pop("preprocessor")
    if "path" in preprocessor_params:
        path = preprocessor_params["path"]
        auto_processor = AutoProcessor.from_pretrained(path, **preprocessor_params)
        return (
            T.Compose(
                [
                    auto_processor,
                    lambda x: x['pixel_values'],
                    lambda x: torch.tensor(x),
                ]
            ),
            lambda x: x,
        )
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                mean=preprocessor_params["mean"], std=preprocessor_params["std"]
            ),
        ]
    ), lambda x: denormalize(x, preprocessor_params["mean"], preprocessor_params["std"])


def is_annotation_valid(annotation):
    bbox = annotation[1:]
    return all([0 <= x <= 1 for x in bbox])


def load_annotations(annotation_path):
    with open(annotation_path, "r") as file:
        annotations = file.readlines()

    # Parse annotations
    targets = []
    for annotation in annotations:
        annotation = annotation.strip().split()
        class_label = int(annotation[0])
        x_center, y_center, width, height = map(float, annotation[1:])
        targets.append([class_label, x_center, y_center, width, height])
    return targets


def process_image_annotation_folders(root):
    image_path = os.path.join(root, "images")
    annotation_path = os.path.join(root, "labels")
    annotations = os.listdir(annotation_path)
    images = os.listdir(image_path)
    annotation_paths = sorted(
        [os.path.join(annotation_path, ann) for ann in annotations]
    )
    image_paths = sorted([os.path.join(image_path, img) for img in images])
    for i in range(len(annotation_paths)):
        assert (
            annotation_paths[i].split("/")[-1].split(".")[0]
            == image_paths[i].split("/")[-1].split(".")[0]
        )
    return image_paths, annotation_paths
