import torch.nn as nn
import torchvision.models as models
from torch.hub import load_state_dict_from_url
from torchvision.models import mobilenet


class MobileNetV2(models.MobileNetV2):
    def __init__(self, num_classes, pretrained=True):
        self.name = 'mobilenet_v2'
        # Mobilenet by default will use the num_classes to create a classifier with number of neurons in the output
        # of the last fully connected layer are the num_classes. Instead we want to be able to retrieve a feature vector
        # of 2048 on demand instead of the final num_classes vector.
        super(MobileNetV2, self).__init__()

        if pretrained:
            state_dict = load_state_dict_from_url(mobilenet.model_urls['mobilenet_v2'], progress=False)
            self.load_state_dict(state_dict)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(self.last_channel, 2048),
        )

        self.last_fc = nn.Linear(2048, num_classes)

    def forward(self, x, skip_last_fc=False):
        x = super().forward(x)

        if not skip_last_fc:
            x = self.last_fc(x)

        return x
