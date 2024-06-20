import torch.nn as nn
from sarfusion.utils.loss_tal_dual import ComputeLoss as YOLOLoss

LOSS_REGISTRY = {
    "yolo_loss": YOLOLoss,
}

def build_loss(params, model=None):
    if not params:
        return None
    
    name = params['name']
    params = params['params']
    if params['requires_model']:
        params['model'] = model
    
    if name in LOSS_REGISTRY:
        return LOSS_REGISTRY[name](**params)
    
    torch_losses = nn.__dict__
    if name in torch_losses:
        return torch_losses[name](**params)
    raise ValueError(f"Loss {name} not found in torch.nn or LOSS_REGISTRY")