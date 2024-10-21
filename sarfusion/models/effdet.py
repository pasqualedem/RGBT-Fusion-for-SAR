from typing import Dict
from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.bench import _post_process, _batch_detection
from effdet.efficientdet import HeadNet
from effdet.config.model_config import efficientdet_model_param_dict
import torch

from sarfusion.utils.structures import LossOutput, WrapperModelOutput, DataDict


def yolo_to_effdet_target(yolo_target: torch.Tensor, batch_size: int):
    """_summary_

    Args:
        yolo_target (torch.Tensor): tensor of type [batch_size, num_boxes, 6], where 6 is [batch_idx, class, x, y, w, h]

    Returns:
        dict: target dictionary for effdet, containing bbox: [batch_size, MAX_NUM_INSTANCES, 4], cls: [batch_size, MAX_NUM_INSTANCES]
    """
    MAX_NUM_INSTANCES = 100  # You may need to adjust this value

    # Initialize the output tensors
    bbox = torch.zeros(batch_size, MAX_NUM_INSTANCES, 4, device=yolo_target.device)
    cls = torch.zeros(batch_size, MAX_NUM_INSTANCES, dtype=torch.long, device=yolo_target.device)

    # Process each item in the batch
    for i in range(batch_size):
        # Get the boxes for the current batch item
        batch_boxes = yolo_target[yolo_target[:, 0] == i]
        
        num_boxes = min(batch_boxes.size(0), MAX_NUM_INSTANCES)
        
        if num_boxes > 0:
            # Extract class and box information
            cls[i, :num_boxes] = batch_boxes[:num_boxes, 1]
            
            # Convert YOLO format (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)
            x_center, y_center, width, height = batch_boxes[:num_boxes, 2:].t()
            bbox[i, :num_boxes, 0] = x_center - width / 2  # x_min
            bbox[i, :num_boxes, 1] = y_center - height / 2  # y_min
            bbox[i, :num_boxes, 2] = x_center + width / 2  # x_max
            bbox[i, :num_boxes, 3] = y_center + height / 2  # y_max
    return {
        'bbox': bbox,
        'cls': cls,
    }
    

def yolo_to_effdet_img_size_scale(dims: list):
    img_size = torch.stack([elem[0] for elem in dims])
    img_scale = torch.tensor([elem[1][0] for elem in dims], device=img_size.device)
    return img_size, img_scale


class EfficientDetAdapter(DetBenchTrain):
    def forward(self, data_dict: DataDict):
        x = data_dict.images
        target = yolo_to_effdet_target(data_dict.target, batch_size=x.size(0))
        img_size, img_scale = yolo_to_effdet_img_size_scale(data_dict.dims)
        class_out, box_out = self.model(x)
        if self.anchor_labeler is None:
            # target should contain pre-computed anchor labels if labeler not present in bench
            assert 'label_num_positives' in target
            cls_targets = [target[f'label_cls_{l}'] for l in range(self.num_levels)]
            box_targets = [target[f'label_bbox_{l}'] for l in range(self.num_levels)]
            num_positives = target['label_num_positives']
        else:
            cls_targets, box_targets, num_positives = self.anchor_labeler.batch_label_anchors(
                target['bbox'],
                target['cls'],
            )

        loss, class_loss, box_loss = self.loss_fn(
            class_out,
            box_out,
            cls_targets,
            box_targets,
            num_positives,
        )
        loss_dict = LossOutput(value=loss, components={'class_loss': class_loss, 'box_loss': box_loss})
        output = WrapperModelOutput(loss=loss_dict)
        if not self.training:
            # if eval mode, output detections for evaluation
            class_out_pp, box_out_pp, indices, classes = _post_process(
                class_out,
                box_out,
                num_levels=self.num_levels,
                num_classes=self.num_classes,
                max_detection_points=self.max_detection_points,
            )
            output.logits = _batch_detection(
                x.shape[0],
                class_out_pp,
                box_out_pp,
                self.anchors.boxes,
                indices,
                classes,
                #img_scale,
                #img_size,
                max_det_per_image=self.max_det_per_image,
                soft_nms=self.soft_nms,
            )
            # Class -= 1 to match the COCO format
            output.logits[:, :, 5] -= 1
        
        return output


def create_model(num_classes=1, image_size=512, backbone="tf_efficientnetv2_l"):
    efficientdet_model_param_dict['tf_efficientnetv2_l'] = dict(
        name='tf_efficientnetv2_l',
        backbone_name='tf_efficientnetv2_l',
        backbone_args=dict(drop_path_rate=0.2),
        num_classes=num_classes,
        url='', )
    
    config = get_efficientdet_config(backbone)
    config.update({'num_classes': num_classes})
    config.update({'image_size': (image_size, image_size)})
    
    print(config)

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    return EfficientDetAdapter(net)