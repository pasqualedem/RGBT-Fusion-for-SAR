import torch

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import YOLOv10DetectionModel as YOLOv10DetectionModelUltra
from ultralytics.models.yolov10.val import YOLOv10DetectionValidator
from ultralytics.models.yolov10.predict import YOLOv10DetectionPredictor
from ultralytics.models.yolov10.train import YOLOv10DetectionTrainer
from ultralytics.utils import ops

from huggingface_hub import PyTorchModelHubMixin
from ultralytics.models.yolov10.card import card_template_text


class YOLOv10DetectionModel(YOLOv10DetectionModelUltra):    
    def forward(self, images):
        if self.training:
            return super().forward(x=images)
        else:
            preds = super().forward(x=images)
            if isinstance(preds, dict):
                preds = preds["one2one"]
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            preds = preds.transpose(-1, -2)
            boxes, scores, labels = ops.v10postprocess(preds, self.args["max_det"], self.nc)
            bboxes = ops.xywh2xyxy(boxes)
            return torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)
    
    def get_learnable_params(self, train_params):
        return [{"params": self.model.parameters()}]


class YOLOv10(Model, PyTorchModelHubMixin, model_card_template=card_template_text):

    def __init__(self, model="yolov10x.pt", task=None, verbose=False, 
                 names=None):
        super().__init__(model=model, task=task, verbose=verbose)
        if names is not None:
            setattr(self.model, 'names', names)

    def push_to_hub(self, repo_name, **kwargs):
        config = kwargs.get('config', {})
        config['names'] = self.names
        config['model'] = self.model.yaml['yaml_file']
        config['task'] = self.task
        kwargs['config'] = config
        super().push_to_hub(repo_name, **kwargs)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": YOLOv10DetectionModel,
                "trainer": YOLOv10DetectionTrainer,
                "validator": YOLOv10DetectionValidator,
                "predictor": YOLOv10DetectionPredictor,
            },
        }