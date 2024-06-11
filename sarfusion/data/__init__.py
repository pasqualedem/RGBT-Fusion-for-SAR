import os
import torch


from sarfusion.data.sard import PoseClassificationDataset
from sarfusion.data.utils import dict_collate_fn, get_collate_fn
from sarfusion.data.utils import build_preprocessor
from sarfusion.data.wisard import (
    WiSARDDataset,
    TRAIN_FOLDERS,
    VAL_FOLDERS,
    TEST_FOLDERS,
    get_wisard_folders,
)


DATASET_REGISTRY = {"sard_pose": PoseClassificationDataset, "wisard": WiSARDDataset}


def get_train_val_test_params(name, dataset_params):
    if name == "sard_pose":
        train_dataset_params = {
            **dataset_params,
            "root": os.path.join(dataset_params["root"], "train"),
        }
        val_dataset_params = {
            **dataset_params,
            "root": os.path.join(dataset_params["root"], "valid"),
        }
        test_dataset_params = {
            **dataset_params,
            "root": os.path.join(dataset_params["root"], "test"),
        }
    elif name == "wisard":
        train_dataset_params = {
            **dataset_params,
            "folders": [
                folder
                for folder in get_wisard_folders(dataset_params["folders"])
                if folder in TRAIN_FOLDERS
            ],
        }
        val_dataset_params = {
            **dataset_params,
            "folders": [
                folder
                for folder in get_wisard_folders(dataset_params["folders"])
                if folder in VAL_FOLDERS
            ],
        }
        test_dataset_params = {
            **dataset_params,
            "folders": [
                folder
                for folder in get_wisard_folders(dataset_params["folders"])
                if folder in TEST_FOLDERS
            ],
        }
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    return train_dataset_params, val_dataset_params, test_dataset_params


def get_dataloaders(dataset_params, dataloader_params):
    dataset_params = dataset_params.copy()

    transforms = build_preprocessor(dataset_params)

    name = dataset_params.pop("name")
    dataclass = DATASET_REGISTRY[name]
    train_dataset_params, val_dataset_params, test_dataset_params = (
        get_train_val_test_params(name, dataset_params)
    )

    train_dataset_params.pop("preprocessor", None)
    val_dataset_params.pop("preprocessor", None)
    test_dataset_params.pop("preprocessor", None)
    train_set = dataclass(
        transform=transforms,
        **train_dataset_params,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        collate_fn=get_collate_fn(train_set),
        **dataloader_params,
    )
    val_set = dataclass(
        transform=transforms,
        **val_dataset_params,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        collate_fn=get_collate_fn(val_set),
        **dataloader_params,
    )
    test_set = dataclass(
        transform=transforms,
        **test_dataset_params,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        collate_fn=get_collate_fn(test_set),
        **dataloader_params,
    )

    return train_loader, val_loader, test_loader
