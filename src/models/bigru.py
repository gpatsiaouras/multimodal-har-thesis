import torch.nn as nn


class BiGRU(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, num_layers, num_classes, device):
        super(BiGRU, self).__init__()
        self.name = 'gru'
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.device = device
        self.num_directions = 2  # bidirectional=True
        self.bigru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                            bidirectional=self.num_directions == 2)
        self.dropout = nn.Dropout(p=0.8, inplace=True)
        self.fc1 = nn.Linear(hidden_size * self.num_directions, 2048)
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x, h):
        out, h = self.bigru(x, h)
        # Retrieve only the last state => results to (batch_size, hidden_size)
        out = out[:, -1, :]
        # Forward to fully connected layers
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out, h

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers * self.num_directions, self.batch_size, self.hidden_size).zero_() \
            .to(self.device)

        return hidden
