from enum import Enum
import gc
import contextlib
from copy import deepcopy
import inspect
import os

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import get_scheduler as get_transformers_scheduler

from sarfusion.data.wisard import build_wisard_items
from sarfusion.utils.structures import DataDict
from sarfusion.utils.structures import WrapperModelOutput
from sarfusion.tracker.abstract_tracker import AbstractLogger
from sarfusion.utils.logger import get_logger

logger = get_logger(__name__)


def parse_params(params_dict):
    train_params = params_dict.get("train", {})
    dataset_params = params_dict.get("dataset", {})
    model_params = params_dict.get("model", {})
    dataloader_params = params_dict.get("dataloader", {})

    return train_params, dataset_params, dataloader_params, model_params


def cast_model(model: torch.nn.Module, precision=torch.float32):
    if precision == torch.float32:
        return model
    model = model.type(precision)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.float()
    return model


class SchedulerStepMoment(Enum):
    BATCH = "batch"
    EPOCH = "epoch"


def get_scheduler(optimizer, num_training_steps, scheduler_params):
    scheduler_params = deepcopy(scheduler_params)
    scheduler_type = scheduler_params.pop("type")
    if scheduler_type is None:
        logger.warning("No scheduler type specified, using None")
        return None, None
    step_moment = scheduler_params.pop("step_moment", None)
    if step_moment is None:
        raise ValueError("step_moment must be specified, choose (batch, epoch)")
    step_moment = SchedulerStepMoment(step_moment)
    num_warmup_steps = scheduler_params.pop("num_warmup_steps", None)
    return (
        get_transformers_scheduler(
            scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            scheduler_specific_kwargs=scheduler_params,
            num_training_steps=num_training_steps,
        ),
        step_moment,
    )


def get_experiment_tracker(accelerator: Accelerator, params: dict) -> AbstractLogger:
    from sarfusion.tracker.wandb_tracker import (
        wandb_experiment as platform_logger,
    )
    return platform_logger(accelerator, params)


def check_nan(model, input_dict, output, loss, step, train_params):
    if not train_params.get("check_nan", False):
        return
    if step % train_params["check_nan"] != 0:
        return
    if torch.isnan(loss) or loss.detach() in [torch.inf, -torch.inf]:
        if (
            train_params["check_nan"] == 1
        ):  # Makes sense only if we are checking every step
            state_dict = {
                "model": model.state_dict(),
                "input_dict": input_dict,
                "loss": loss,
                "step": step,
                "output": output,
            }
            torch.save(state_dict, "nan.pt")
        raise ValueError("NaNs in loss")


def handle_oom(model, input_dict, optimizer, epoch, step):
    logger.warning(f"OOM at step {step}")
    logger.warning(torch.cuda.memory_summary())
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "input_dict": input_dict,
        },
        f"oom_epoch_{epoch}_step_{step}.pt",
    )
    optimizer.zero_grad()
    del input_dict
    gc.collect()
    torch.cuda.empty_cache()


@contextlib.contextmanager
def nosync_accumulation(accumulate=False, accelerator=None, model=None):
    if accumulate:
        with accelerator.no_sync(model):
            yield
    else:
        with contextlib.nullcontext():
            yield


class WrapperModule(torch.nn.Module):
    def __init__(self, model, loss=None) -> None:
        super().__init__()
        self.model = model
        self.loss = loss

    def forward(self, input_dict: DataDict):
        model_args = inspect.signature(self.model.forward).parameters
        model_dict = {k: v for k, v in input_dict.items() if k in model_args}
        result_dict = self.model(**model_dict)
        
        loss = None
        if input_dict.target is not None and self.loss is not None and self.training:
            loss = self.loss(result_dict, input_dict.target)
        return WrapperModelOutput(loss=loss, **result_dict)

    def get_learnable_params(self, train_params):
        model_params = list(self.model.get_learnable_params(train_params))
        if isinstance(self.loss, torch.nn.Module):
            loss_params = list(self.loss.parameters())
        else:
            loss_params = []
        if len(loss_params) > 0:
            loss_params = [{"params": loss_params}]
        return model_params + loss_params
        

def unwrap_model(model):
    if isinstance(model, WrapperModule):
        return model.model
    # Parallel
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
        return unwrap_model(model)
    return model


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
