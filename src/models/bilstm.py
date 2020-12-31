import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, num_layers, num_classes, device):
        super(BiLSTM, self).__init__()
        self.name = 'lstm'
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.device = device
        self.num_directions = 2  # bidirectional=True
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            bidirectional=self.num_directions == 2, dropout=0.8)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x, h, c):
        out, (h, c) = self.lstm(x, (h, c))
        # Retrieve only the last state => results to (batch_size, hidden_size)
        out = out[:, -1, :]
        # Forward to fully connected layer
        out = self.fc(out)

        return out, h, c

    def init_hidden(self):
        weight = next(self.parameters()).data
        h0 = weight.new(self.num_layers * self.num_directions, self.batch_size, self.hidden_size).zero_() \
            .to(self.device)
        c0 = weight.new(self.num_layers * self.num_directions, self.batch_size, self.hidden_size).zero_() \
            .to(self.device)

        return h0, c0
