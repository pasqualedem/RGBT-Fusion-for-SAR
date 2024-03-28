import torch.nn as nn

LOSS_REGISTRY = {}

def build_loss(params):
    if not params:
        return None
    
    name = params['name']
    params = params['params']
    
    if name in LOSS_REGISTRY:
        return LOSS_REGISTRY[name](**params)
    
    torch_losses = nn.__dict__
    if name in torch_losses:
        return torch_losses[name](**params)
    raise ValueError(f"Loss {name} not found in torch.nn or LOSS_REGISTRY")