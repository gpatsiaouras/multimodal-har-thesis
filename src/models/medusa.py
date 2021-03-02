import torch.nn as nn
import torch

from models import CNN1D, MobileNetV2, MLP


class Medusa(nn.Module):
    def __init__(self, out_size):
        super(Medusa, self).__init__()
        self.name = 'medusa'
        self.n1 = CNN1D(642, out_size=out_size, norm_out=True)
        self.n2 = MobileNetV2(out_size, norm_out=True)
        self.n3 = CNN1D(1107, out_size=out_size, norm_out=True)
        self.mlp = MLP(out_size * 3, 2048, out_size)

    def forward(self, x):
        (x1, x2, x3) = x
        x1 = self.n1(x1)
        x2 = self.n2(x2)
        x3 = self.n3(x3)

        # Concatenate vectors
        x_all = torch.cat((x1, x2, x3), dim=1)

        # Pass through mlp
        return self.mlp(x_all)
