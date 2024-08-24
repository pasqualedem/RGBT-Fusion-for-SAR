from copy import copy
from ultralytics.models.yolov10.train import (
    YOLOv10DetectionTrainer,
    YOLOv10DetectionValidator,
)
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.cfg import cfg2dict, IterableSimpleNamespace
from sarfusion.data.wisard import WiSARDYOLODataset
from sarfusion.utils.general import colorstr
from ultralytics.utils.torch_utils import de_parallel, strip_optimizer
from sarfusion.utils.plots import plot_images

def build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32):
    """Build YOLO Dataset."""
    return WiSARDYOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
        augment_vis_ir=cfg.augment_vis_ir,
    )


WISARD_DEFAULT_CFG = IterableSimpleNamespace(
    **{
        **cfg2dict(DEFAULT_CFG),
        "augment_vis_ir": False,
    }
)


class WisardTrainer(YOLOv10DetectionTrainer):
    def __init__(self, cfg=WISARD_DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        im_file = [
            elem[0] if isinstance(elem, list) else elem for elem in batch["im_file"]
        ]
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=im_file,
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=gs,
        )

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_om", "cls_om", "dfl_om", "box_oo", "cls_oo", "dfl_oo",
        args = cfg2dict(copy(self.args))
        args.pop("augment_vis_ir")
        args = IterableSimpleNamespace(**args)
        return YOLOv10DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=args, _callbacks=self.callbacks
        )
        
    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        batch_size = self.batch_size if self.args.task == "obb" else self.batch_size * 2
        test_loader = self.get_dataloader(self.data['test'], batch_size=batch_size, mode="val", rank=-1)
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.validator.dataloader = test_loader
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.metrics = {k.replace("metrics/", "test/"): v for k,v in self.metrics.items()}
                    self.run_callbacks("on_fit_epoch_end")
