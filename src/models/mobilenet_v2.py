import torch.nn as nn
import torchvision.models as models


class MobileNetV2(models.MobileNetV2):
    def __init__(self, num_classes):
        # Mobilenet by default will use the num_classes to create a classifier with number of neurons in the output
        # of the last fully connected layer are the num_classes. Instead we want to be able to retrieve a feature vector
        # of 2048 on demand instead of the final num_classes vector.
        super(MobileNetV2, self).__init__(num_classes=2048)
        self.last_fc = nn.Linear(2048, num_classes)
        self.name = 'mobilenet_v2'

    def forward(self, x, skip_last_fc=False):
        x = super().forward(x)

        if not skip_last_fc:
            x = self.last_fc(x)

        return x
