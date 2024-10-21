
import torch.nn as nn
from transformers import DetrImageProcessor, DetrForObjectDetection



class Detr(nn.Module):
    def __init__(self, threshold=0.9):
        super(Detr, self).__init__()
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.threshold = threshold
        
    def forward(self, images, dims):
        outputs = self.model(images)
        if not self.training:
            return self.processor.post_process_object_detection(outputs, threshold=self.threshold, target_sizes=dims)
        return outputs