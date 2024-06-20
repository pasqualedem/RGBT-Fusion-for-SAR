import torch
from easydict import EasyDict


class ModelOutput(EasyDict):
    features: list[torch.Tensor]
    features_aux: list[torch.Tensor]
    logits: list[torch.Tensor]
    logits_aux: list[torch.Tensor]
    predictions: list[torch.Tensor]


class LossOutput(EasyDict):
    value: torch.Tensor
    components: dict


class WrapperModelOutput(ModelOutput):
    loss: LossOutput


class DataDict(EasyDict):
    images: torch.Tensor
    target: torch.Tensor
    dims: tuple
    path: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "images" not in self:
            self.images = None
        if "target" not in self:
            self.target = None