import torch
import torchvision.transforms.functional as F

from torchvision.transforms import Resize


class ResizePadKeepRatio(Resize):
    def __init__(self, size, interpolation=F.InterpolationMode.BILINEAR):
        super(ResizePadKeepRatio, self).__init__(size, interpolation)
        self.size = size

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            _, w, h = img.size()
        else:
            w, h = img.size
        if w > h:
            new_w = self.size
            new_h = int(h * new_w / w)
        else:
            new_h = self.size
            new_w = int(w * new_h / h)
        resized = F.resize(img, (new_w, new_h), self.interpolation)
        if new_h == self.size:
            pad = self.size - new_w
            return F.pad(resized, (0, 0, 0, pad))
        else:
            pad = self.size - new_h
            return F.pad(resized, (0, 0, pad, 0))