import torch.nn as nn
from torchvision import models

def get_mobilenet_model(num_classes):
    """
    Load and modify a pretrained MobileNetV2 model.
    """
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model