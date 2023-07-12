import torch
import torch.nn as nn

from torchvision import transforms
from .models import resnet18

class ArcFaceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18().cuda()
        self.size = 128
        self.tf = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(self.size),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        

    def forward(self, x, target):
        return self.cosine_distance(self.model(self.tf(x)), self.model(self.tf(target)))

    def cosine_distance(self, x1, x2):
        return 1 - nn.functional.cosine_similarity(x1, x2)[0]
