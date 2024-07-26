from copy import deepcopy
import torch

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import YOLOv10DetectionModel as YOLOv10DetectionModelUltra
from ultralytics.models.yolov10.val import YOLOv10DetectionValidator
from ultralytics.models.yolov10.predict import YOLOv10DetectionPredictor
from ultralytics.models.yolov10.train import YOLOv10DetectionTrainer
from ultralytics.utils import ops

from ultralytics.nn.modules import (
    OBB,
    Detect,
    Pose,
    Segment,
    v10Detect
)
from ultralytics.utils.torch_utils import initialize_weights, scale_img
from ultralytics.nn.tasks import parse_model, yaml_model_load, BaseModel
from ultralytics.utils import LOGGER

from huggingface_hub import PyTorchModelHubMixin
from ultralytics.models.yolov10.card import card_template_text

from sarfusion.utils.lossv10 import v10DetectLoss
from sarfusion.utils.structures import ModelOutput


class YOLOv10DetectionModel(BaseModel):
    def init_criterion(self):
        return v10DetectLoss(self)
    
    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        """Initialize the YOLOv8 detection model with the given config and parameters."""
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Pose, OBB)) else self.forward(x)
            if isinstance(m, v10Detect):
                forward = lambda x: self.forward(x).features["one2many"]
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")
            
    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)  # forward
            if isinstance(yi, dict):
                yi = yi["one2one"]  # yolov10 outputs
            if isinstance(yi, (list, tuple)):
                yi = yi[0]
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def forward(self, images):
        if self.training:
            features = super().forward(x=images)
            return ModelOutput(features=features)
        else:
            result = super().forward(x=images)
            if isinstance(result, dict):
                features = result["one2one"]
            else:
                features = result
            if isinstance(features, (list, tuple)):
                features = features[0]
            preds = features.transpose(-1, -2)
            boxes, scores, labels = ops.v10postprocess(preds, self.args["max_det"], len(self.names))
            bboxes = ops.xywh2xyxy(boxes)
            preds = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)
            return ModelOutput(logits=preds, features=result)
    
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