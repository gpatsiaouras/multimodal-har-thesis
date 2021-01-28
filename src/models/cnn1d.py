import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self,
                 len_seq,
                 out_size,
                 in_channels=None,
                 out_channels=None,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 pool_padding=0,
                 pool_size=2,
                 dropout_rate=0.8,
                 fc_size=2048):
        """
        Initiate layers of a 5 convolutions layer network
        """
        super(CNN1D, self).__init__()
        if in_channels is None:
            in_channels = [1, 32, 64, 128, 256]
        if out_channels is None:
            out_channels = [32, 64, 128, 256, 512]

        self.name = 'CNN1D'
        self.num_layers = len(out_channels)
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=None)
        self.dropout = nn.Dropout(p=dropout_rate, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv1d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.batchNorm1 = nn.BatchNorm1d(out_channels[0])
        self.conv2 = nn.Conv1d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.batchNorm2 = nn.BatchNorm1d(out_channels[1])
        self.conv3 = nn.Conv1d(in_channels=in_channels[2], out_channels=out_channels[2], kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.batchNorm3 = nn.BatchNorm1d(out_channels[2])
        self.conv4 = nn.Conv1d(in_channels=in_channels[3], out_channels=out_channels[3], kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.batchNorm4 = nn.BatchNorm1d(out_channels[3])
        self.conv5 = nn.Conv1d(in_channels=in_channels[4], out_channels=out_channels[4], kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.batchNorm5 = nn.BatchNorm1d(out_channels[4])

        conv_out_size = len_seq
        for _ in range(self.num_layers):
            conv_out_size = int((conv_out_size + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
            conv_out_size = int((conv_out_size + 2 * pool_padding - (pool_size - 1) - 1) / pool_size + 1)

        self.fc1 = nn.Linear(int(out_channels[self.num_layers - 1] * conv_out_size), fc_size)
        self.fc2 = nn.Linear(fc_size, fc_size)
        self.fc3 = nn.Linear(fc_size, out_size)

    def forward(self, x, skip_last_fc=False):
        # Add a dimension in the middle because we only have one channel of data
        x = x.unsqueeze(1)

        # Forward
        x = self.relu(self.batchNorm1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.batchNorm2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.batchNorm3(self.conv3(x)))
        x = self.pool(x)
        x = self.relu(self.batchNorm4(self.conv4(x)))
        x = self.pool(x)
        x = self.relu(self.batchNorm5(self.conv5(x)))
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if not skip_last_fc:
            x = self.dropout(x)
            x = self.fc3(x)

        return x
