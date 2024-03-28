from copy import deepcopy
from transformers import AutoModel, ViTForImageClassification


def build_model(params):
    name = params['name']
    params = params['params']
    
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name](**params)
    return AutoModel.from_pretrained(name)


def build_vit_classifier(**params):
    params = deepcopy(params)
    labels = params.pop('labels')
    path = params.pop('path')
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
    return vit
    

MODEL_REGISTRY = {
    "vit_classifier": build_vit_classifier,
}