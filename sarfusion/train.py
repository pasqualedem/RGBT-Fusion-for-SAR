import os
from ultralytics import YOLOv10
from ultralytics.models.yolov10.train import YOLOv10DetectionTrainer, YOLOv10DetectionValidator
from ultralytics.utils.plotting import plot_images
from sarfusion.data.wisard import TEST_FOLDERS, TRAIN_FOLDERS, VAL_FOLDERS, WiSARDYOLODataset, build_wisard_items, get_wisard_folders
from sarfusion.utils.general import colorstr
from sarfusion.utils.torch_utils import de_parallel


class YOLOv10WiSARD(YOLOv10):
    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        task_map = super().task_map
        task_map['detect']['trainer'] = WisardTrainer
        # task_map['detect']['validator'] = WisardValidator
        return task_map


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
    )


class WisardTrainer(YOLOv10DetectionTrainer):
    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        im_file = [elem[0] if isinstance(elem, list) else elem for elem in batch["im_file"]]
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
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
    

def generate_wisard_filelist(root, folders, filename):
    items = build_wisard_items(root, folders)
    images = []
    for item in items:
        if isinstance(item[1][0], str):
            images.append(item[1][0])
        else:
            images.append(item[1][0][0] + "," + item[1][1][0])
    with open(os.path.join(root, filename), "w") as f:
        for item in images:
            f.write(f"{item}\n")
    return images


def yolo_train():
    root = "dataset/WiSARD"
    folders = "vis"
    
    train_folders = [folder for folder in get_wisard_folders(folders) if  folder in TRAIN_FOLDERS]
    generate_wisard_filelist(root, train_folders, "train.txt")
    val_folders = [folder for folder in get_wisard_folders(folders) if  folder in VAL_FOLDERS]
    generate_wisard_filelist(root, val_folders, "val.txt")
    test_folders = [folder for folder in get_wisard_folders(folders) if  folder in TEST_FOLDERS]
    generate_wisard_filelist(root, test_folders, "test.txt")

    model = YOLOv10.from_pretrained('jameslahm/yolov10n')
    args = dict(
        data="wisard.yaml",
        epochs=500,
        batch=4,
        imgsz=640,
        mosaic=False,
        plots=True,
        workers=0,
    )

    model.train(trainer=WisardTrainer, **args)