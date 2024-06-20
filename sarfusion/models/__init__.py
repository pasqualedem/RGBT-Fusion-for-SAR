from copy import deepcopy
from enum import StrEnum
from transformers import AutoModel, ViTForImageClassification

from sarfusion.experiment.utils import WrapperModule
from sarfusion.models.experimental import attempt_load
from sarfusion.models.utils import torch_dict_load
from sarfusion.utils.utils import load_yaml


class AdditionalParams(StrEnum):
    PRETRAINED_PATH = "pretrained_path"


def build_model(params):
    """
    Build a model from a yaml file or a dictionary
    
    Args:
        params (dict or str): Dictionary or path to yaml file containing model parameters
        Additional parameters:
            pretrained_path: The path of the pretrained model
        
    """
    if isinstance(params, str):
        params = load_yaml(params)
    params = deepcopy(params)
    name = params["name"]
    params = params["params"]
    pretrained_path = params.pop(AdditionalParams.PRETRAINED_PATH, None)

    if name in MODEL_REGISTRY:
        model =  MODEL_REGISTRY[name](**params)
    else:
        model = AutoModel.from_pretrained(name)
      
    if pretrained_path:
        model.load_state_dict(torch_dict_load(pretrained_path))
        print(f"Loaded model from {pretrained_path}")
    return model


def backbone_learnable_params(self, train_params: dict):
    freeze_backbone = train_params.get("freeze_backbone", False)
    if freeze_backbone:
        for param in self.vit.parameters():
            param.requires_grad = False
        return [{"params": [x[1] for x in self.named_parameters() if x[1].requires_grad]}]
    return [{"params": list(self.parameters())}]


def build_vit_classifier(**params):
    params = deepcopy(params)
    labels = params.pop("labels")
    path = params.pop("path")
    num_labels = len(labels)
    id2label = {str(i): c for i, c in enumerate(labels)}
    label2id = {c: str(i) for i, c in enumerate(labels)}
    vit = ViTForImageClassification.from_pretrained(
        path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        **params
    )
    vit.get_learnable_params = backbone_learnable_params.__get__(
        vit, ViTForImageClassification
    )
    return vit


def build_yolo_v9(cfg, checkpoint=None, iou_t=0.2, conf_t=0.001):
    from sarfusion.models.yolo import Model as YOLOv9
    if checkpoint:
        return attempt_load(checkpoint)
    return YOLOv9(cfg)


MODEL_REGISTRY = {
    "vit_classifier": build_vit_classifier,
    "yolov9": build_yolo_v9,
}
