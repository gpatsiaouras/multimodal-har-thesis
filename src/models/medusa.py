import torch.nn as nn
import torch

from models import CNN1D, MobileNetV2, MLP


class Medusa(nn.Module):
    def __init__(self, mlp_kwargs, n1_kwargs, n2_kwargs, n3_kwargs=None):
        super(Medusa, self).__init__()
        self.name = 'medusa'
        self.n1 = CNN1D(**n1_kwargs)
        self.n2 = MobileNetV2(**n2_kwargs)
        self.n3 = None
        if n3_kwargs:
            self.n3 = CNN1D(**n3_kwargs)
        self.mlp = MLP(**mlp_kwargs)

    def forward(self, x):
        if self.n3:
            (x1, x2, x3) = x
        else:
            (x1, x2) = x
        x1 = self.n1(x1)
        x2 = self.n2(x2)
        if self.n3:
            x3 = self.n3(x3)
            x = torch.cat((x1, x2, x3), dim=1)
        else:
            x = torch.cat((x1, x2), dim=1)
        # L2 Normalize the concatenated vector
        norm = x.norm(p=2, dim=1, keepdim=True)
        x = x.div(norm)

        x = self.mlp(x)

        return x
