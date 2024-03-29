from enum import StrEnum

import torch


class DataDict(StrEnum):
    IMAGES = "pixel_values"
    TARGET = "label"    
    
    
def dict_collate_fn(batch):
    d = {}
    keys = batch[0].keys()
    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            d[key] = torch.stack([sample[key] for sample in batch])
        elif isinstance(batch[0][key], int) or isinstance(batch[0][key], float):
            d[key] = torch.stack([torch.tensor(sample[key]) for sample in batch])
        
    return d