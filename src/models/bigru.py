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
                            bidirectional=self.num_directions == 2, dropout=0.8)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x, h):
        out, h = self.bigru(x, h)
        # Retrieve only the last state => results to (batch_size, hidden_size)
        out = out[:, -1, :]
        # Forward to fully connected layer
        out = self.fc(out)

        return out, h

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers * self.num_directions, self.batch_size, self.hidden_size).zero_() \
            .to(self.device)

        return hidden
