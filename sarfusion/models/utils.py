import torch

from easydict import EasyDict

from safetensors import safe_open
from safetensors.torch import save_file


def torch_dict_load(file_path):
    if file_path.endswith(".pth") or file_path.endswith(".pt") or file_path.endswith(".bin"):
        return torch.load(file_path)
    if file_path.endswith(".safetensors"):
        with safe_open(file_path, framework="pt") as f:
            d = {}
            for k in f.keys():
                d[k] = f.get_tensor(k)
        return d
    raise ValueError("File extension not supported")

        
def torch_dict_save(data, file_path):
    if file_path.endswith(".pth") or file_path.endswith(".pt") or file_path.endswith(".bin"):
        torch.save(data, file_path)
    elif file_path.endswith(".safetensors"):
        save_file(data, file_path)
    else:
        raise ValueError("File extension not supported")
    
    
class ModelOutput(EasyDict):
    features: list[torch.Tensor]
    features_aux: list[torch.Tensor]
    logits: list[torch.Tensor]
    logits_aux: list[torch.Tensor]


class LossOutput(EasyDict):
    value: torch.Tensor
    components: dict
    
    
class WrapperModelOutput(ModelOutput):
    loss: LossOutput