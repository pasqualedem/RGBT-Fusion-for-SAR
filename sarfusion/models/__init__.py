from copy import deepcopy
from transformers import AutoModel, ViTForImageClassification


def build_model(params):
    name = params["name"]
    params = params["params"]

    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name](**params)
    return AutoModel.from_pretrained(name)


def backbone_learnable_params(self, train_params: dict):
    freeze_backbone = train_params.get("freeze_backbone", False)
    if freeze_backbone:
        for param in self.vit.parameters():
            param.requires_grad = False
        return self.classifier.parameters()
    return self.parameters()

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


MODEL_REGISTRY = {
    "vit_classifier": build_vit_classifier,
}
