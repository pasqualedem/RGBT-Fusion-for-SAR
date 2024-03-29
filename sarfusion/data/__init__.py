import os
import torch
import torchvision.transforms as T

from transformers import AutoProcessor

from sarfusion.data.sard import PoseClassificationDataset
from sarfusion.data.utils import DataDict, dict_collate_fn


DATASET_REGISTRY = {
    "sard_pose": PoseClassificationDataset,
}


def build_preprocessor(params):
    preprocessor_params = params.pop("preprocessor")
    if "path" in preprocessor_params:
        auto_processor =  AutoProcessor.from_pretrained(preprocessor_params["path"])
        return T.Compose([
            auto_processor,
            lambda x: x[DataDict.IMAGES][0],
            lambda x: torch.tensor(x)
        ])
    return T.Compose([
        T.Normalize(mean=preprocessor_params["mean"], std=preprocessor_params["std"]),
    ])


def get_dataloaders(dataset_params, dataloader_params):
    dataset_params = dataset_params.copy()
    
    transforms = build_preprocessor(dataset_params)
    
    root = dataset_params.pop("root")
    name = dataset_params.pop("name")
    dataclass = DATASET_REGISTRY[name]
    train_root = os.path.join(root, "train")
    val_root = os.path.join(root, "valid")
    test_root = os.path.join(root, "test")

    train_set = dataclass(
        train_root,
        transform=transforms,
        **dataset_params,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        collate_fn=dict_collate_fn,
        **dataloader_params,
    )
    val_set = dataclass(
        val_root,
        transform=transforms,
        **dataset_params,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        collate_fn=dict_collate_fn,
        **dataloader_params,
    )
    test_set = dataclass(
        test_root,
        transform=transforms,
        **dataset_params,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        collate_fn=dict_collate_fn,
        **dataloader_params,
    )                                         

    return train_loader, val_loader, test_loader