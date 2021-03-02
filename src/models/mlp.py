import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, norm_out=False):
        super(MLP, self).__init__()
        self.norm_out = norm_out
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.relu(x)

        if self.norm_out:
            norm = x.norm(p=2, dim=1, keepdim=True)
            x = x.div(norm)

        return x
