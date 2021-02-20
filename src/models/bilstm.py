import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size):
        super(BiLSTM, self).__init__()
        self.name = 'lstm'
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_directions = 2
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=self.num_directions == 2)
        self.dropout = nn.Dropout(p=0.8, inplace=True)
        self.fc1 = nn.Linear(hidden_size * self.num_directions, 2048)
        self.fc2 = nn.Linear(2048, out_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Retrieve only the last state => results to (batch_size, hidden_size)
        out = out[:, -1]
        # Forward to fully connected layers
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)

        if self.norm_out:
            norm = out.norm(p=2, dim=1, keepdim=True)
            out = out.div(norm)

        return out
