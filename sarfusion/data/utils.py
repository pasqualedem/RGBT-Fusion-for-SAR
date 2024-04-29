from copy import deepcopy
from enum import StrEnum

import torch
import torchvision.transforms as T
from transformers import AutoProcessor


class DataDict(StrEnum):
    IMAGES = "pixel_values"
    TARGET = "label"    
    PATH = "path"
    
    
def dict_collate_fn(batch):
    d = {}
    keys = batch[0].keys()
    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            d[key] = torch.stack([sample[key] for sample in batch])
        elif isinstance(batch[0][key], int) or isinstance(batch[0][key], float):
            d[key] = torch.stack([torch.tensor(sample[key]) for sample in batch])
        
    return d


def build_preprocessor(params):
    params = deepcopy(params)
    preprocessor_params = params.pop("preprocessor")
    if "path" in preprocessor_params:
        path = preprocessor_params["path"]
        auto_processor =  AutoProcessor.from_pretrained(path, **preprocessor_params)
        return T.Compose([
            auto_processor,
            lambda x: x[DataDict.IMAGES][0],
            lambda x: torch.tensor(x)
        ])
    return T.Compose([
        T.Normalize(mean=preprocessor_params["mean"], std=preprocessor_params["std"]),
    ])
    
    
def is_annotation_valid(annotation):
    bbox = annotation[1:]
    return all([0 <= x <= 1 for x in bbox])
    
    