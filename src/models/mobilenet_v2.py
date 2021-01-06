import torch.nn as nn
import torchvision.models as models


class MobileNetV2(models.MobileNetV2):
    def __init__(self, num_classes, pretrained=True):
        self.pretrained = pretrained
        super(MobileNetV2, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(self.last_channel, 2048),
            nn.Linear(2048, num_classes)
        )
        self.name = 'mobilenet_v2'
