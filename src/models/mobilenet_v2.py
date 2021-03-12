import torch.nn as nn
import torchvision.models as models
from torch.hub import load_state_dict_from_url
from torchvision.models import mobilenet


class MobileNetV2(models.MobileNetV2):
    def __init__(self, out_size, pretrained=True, norm_out=False, dropout_rate=0.8):
        self.name = 'mobilenet_v2'
        # Mobilenet by default will use the num_classes to create a classifier with number of neurons in the output
        # of the last fully connected layer are the num_classes. Instead we want to be able to retrieve a feature vector
        # on demand instead of the final num_classes vector.
        super(MobileNetV2, self).__init__()

        if pretrained:
            state_dict = load_state_dict_from_url(mobilenet.model_urls['mobilenet_v2'], progress=False)
            self.load_state_dict(state_dict)

        # Replace the classifier from the parent class
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(self.last_channel, 2048),
        )

        # Add the last fully connected separately so that we can skip it if needed
        self.last_fc = nn.Linear(2048, out_size)

        #  Don't skip last layer by default
        self.skip_last_fc = False
        self.norm_out = norm_out

    def forward(self, x):
        x = super().forward(x)
        if not self.skip_last_fc:
            x = self.last_fc(x)

        if self.norm_out:
            norm = x.norm(p=2, dim=1, keepdim=True)
            x = x.div(norm)

        return x
