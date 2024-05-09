import os
from PIL import Image
import cv2
from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as tvF

from sarfusion.data.utils import (
    DataDict,
    load_annotations,
    process_image_annotation_folders,
)


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

VIS = [
    "200321_Baker_Phantom_VIS",
    "200402_Carnation_Inspire_VIS",
    "200402_Karen_Inspire_VIS",
    "200426_SkookumCreek_Mavic_Mini_VIS_0006",
    "200426_SkookumCreek_Mavic_Mini_VIS_0007",
    "200426_SkookumCreek_Mavic_Mini_VIS_0008",
    "200505_Bellingham_Mavic_Mini_VIS_0015",
    "200505_Bellingham_Mavic_Mini_VIS_0016",
    "200528_Everson_Mavic_Mini_VIS_0021",
    "200528_Everson_Mavic_Mini_VIS_0028",
    "200528_Everson_Mavic_Mini_VIS_0029",
    "200528_Everson_Mavic_Mini_VIS_0031",
    "200614_SuddenValley_Phantom_VIS_0005",
    "200614_SuddenValley_Phantom_VIS_0006",
    "200614_SuddenValley_Phantom_VIS_0012",
    "200614_SuddenValley_Phantom_VIS_0013",
    "200614_SuddenValley_Phantom_VIS_0014",
    "200717_Mission_FLIR_VIS",
    "210327_Airfield_FLIR_VIS_1",
    "210327_Airfield_FLIR_VIS_2",
    "210327_Airfield_FLIR_VIS_3",
    "210327_Airfield_FLIR_VIS_4",
]

IR = [
    "200704_Baker_FLIR_IR_1",
    "200704_Baker_FLIR_IR_2",
    "200910_Carnation_FLIR_IR_1",
    "200910_Carnation_FLIR_IR_2",
    "200910_Carnation_FLIR_IR_3",
    "200910_Carnation_FLIR_IR_4",
    "200910_Carnation_FLIR_IR_5",
    "200910_Carnation_FLIR_IR_6",
    "200910_Carnation_FLIR_IR_7",
    "200929_Karen_FLIR_IR_1",
    "200929_Karen_FLIR_IR_2",
    "200929_Karen_FLIR_IR_3",
    "200929_Karen_FLIR_IR_4",
    "200929_Karen_FLIR_IR_5",
    "200929_Karen_FLIR_IR_6",
    "220109_Baker_Enterprise_IR_2",
    "210327_Airfield_FLIR_IR_1",
    "210327_Airfield_FLIR_IR_2",
    "210327_Airfield_FLIR_IR_3",
    "210327_Airfield_FLIR_IR_4",
    "210327_Airfield_FLIR_IR_5",
    "210327_Airfield_FLIR_IR_6",
    "210327_Airfield_FLIR_IR_7",
    "210327_Airfield_FLIR_IR_8",
]

VIS_IR = [
    ("210417_MtErie_Enterprise_VIS_0003", "210417_MtErie_Enterprise_IR_0004"),
    ("210417_MtErie_Enterprise_VIS_0005", "210417_MtErie_Enterprise_IR_0006"),
    ("210417_MtErie_Enterprise_VIS_0007", "210417_MtErie_Enterprise_IR_0008"),
    ("210529_Carnation_Enterprise_VIS_0023", "210529_Carnation_Enterprise_IR_0024"),
    ("210529_Carnation_Enterprise_VIS_0025", "210529_Carnation_Enterprise_IR_0026"),
    ("210812_Hannegan_Enterprise_VIS_0053", "210812_Hannegan_Enterprise_IR_0054"),
    ("210812_Hannegan_Enterprise_VIS_0055", "210812_Hannegan_Enterprise_IR_0056"),
    ("210924_FHL_Enterprise_VIS_0126", "210924_FHL_Enterprise_IR_0127"),
    ("210924_FHL_Enterprise_VIS_0134", "210924_FHL_Enterprise_IR_0135"),
    ("210924_FHL_Enterprise_VIS_0401", "210924_FHL_Enterprise_IR_0402"),
    ("210924_FHL_Enterprise_VIS_0403", "210924_FHL_Enterprise_IR_0404"),
    ("210924_FHL_Enterprise_VIS_0405", "210924_FHL_Enterprise_IR_0406"),
    ("210924_FHL_Enterprise_VIS_0407", "210924_FHL_Enterprise_IR_0408"),
    ("210924_FHL_Enterprise_VIS_0409", "210924_FHL_Enterprise_IR_0410"),
    ("210924_FHL_Enterprise_VIS_0564", "210924_FHL_Enterprise_IR_0565"),
    ("210924_FHL_Enterprise_VIS_0566", "210924_FHL_Enterprise_IR_0567"),
    ("220109_Baker_Enterprise_VIS_1", "220109_Baker_Enterprise_IR_1"),
]

MISSING_ANNOTATIONS = [
    "210924_FHL_Enterprise_VIS_0126/labels/DJI_0126_00000211.txt",
    "210924_FHL_Enterprise_VIS_0126/labels/DJI_0126_00000212.txt",
]


def collate_rgb_ir(rgb, ir):
    ir = ir[0:1]  # All channels are the same
    # calculate h, w displacement
    rgb_h, rgb_w = rgb.shape[1:]
    ir_h, ir_w = ir.shape[1:]

    new_ir_h = rgb_h
    new_ir_w = int(ir_w * (rgb_h / ir_h))

    new_ir = tvF.resize(ir, (new_ir_h, new_ir_w))
    w_pad = (rgb_w - new_ir_w) // 2
    new_ir = tvF.pad(new_ir, (w_pad, 0, w_pad, 0))
    return torch.cat((rgb, new_ir), dim=0)


class WiSARDDataset(Dataset):
    RGB_ITEM = 0
    IR_ITEM = 1
    MULTI_MODALITY_ITEM = 2

    def __init__(
        self,
        root,
        folders,
        transform=None,
        ir_transform=None,
        return_path=False,
        augment=False,
        img_size=640,
    ):
        rgb_datasets = list(filter(lambda x: x in VIS, folders))
        ir_datasets = list(filter(lambda x: x in IR, folders))
        multi_modality_datasets = list(filter(lambda x: isinstance(x, tuple), folders))
        rgb_items = [
            list(zip(*process_image_annotation_folders(os.path.join(root, folder))))
            for folder in rgb_datasets
        ]
        rgb_items = [(self.RGB_ITEM, item) for dataset in rgb_items for item in dataset]
        ir_items = [
            list(zip(*process_image_annotation_folders(os.path.join(root, folder))))
            for folder in ir_datasets
        ]
        ir_items = [(self.IR_ITEM, item) for dataset in ir_items for item in dataset]
        multi_modality_rgb_items = [
            list(zip(*process_image_annotation_folders(os.path.join(root, folder[0]))))
            for folder in multi_modality_datasets
        ]
        multi_modality_ir_items = [
            list(zip(*process_image_annotation_folders(os.path.join(root, folder[1]))))
            for folder in multi_modality_datasets
        ]
        multi_modality_items = [
            (self.MULTI_MODALITY_ITEM, (rgb_item, ir_item))
            for rgb_dataset, ir_dataset in zip(
                multi_modality_rgb_items, multi_modality_ir_items
            )
            for rgb_item, ir_item in zip(rgb_dataset, ir_dataset)
        ]

        self.items = rgb_items + ir_items + multi_modality_items
        self.transform = transform
        self.ir_transform = ir_transform
        self.img_size = img_size
        self.augment = augment
        self.return_path = return_path

    def __len__(self):
        return len(self.items)

    def _load_rgb(self, img_path):
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img

    def _load_ir(self, img_path):
        img = Image.open(img_path).convert("RGB")
        img = self.ir_transform(img)
        return img

    def __getitem__(self, idx):
        item_type, item = self.items[idx]

        if item_type == self.RGB_ITEM:
            img_path, annotation_path = item
            img = self._load_rgb(img_path)
            targets = load_annotations(annotation_path)
            img_path_vis = img_path
        elif item_type == self.IR_ITEM:
            img_path, annotation_path = item
            img = self._load_ir(img_path)
            targets = load_annotations(annotation_path)
            img_path_vis = img_path
        else:
            (img_path_vis, annotation_path), (img_path_ir, annotation_path_ir) = item
            img_vis = self._load_rgb(img_path_vis)
            img_ir = self._load_ir(img_path_ir)
            img = collate_rgb_ir(img_vis, img_ir)
            targets = load_annotations(annotation_path)
            targets_ir = load_annotations(annotation_path_ir)

        data_dict = DataDict(images=img, target=targets, targets_ir=targets_ir)
        if self.return_path:
            data_dict.path = img_path_vis

        return data_dict
