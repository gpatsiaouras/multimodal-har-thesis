import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self):
        """
        Initiate layers of a 5 convolutions layer network
        """
        super(CNN1D, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=None)
        self.dropout = nn.Dropout(p=0.8, inplace=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.batchNorm1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.batchNorm2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.batchNorm3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.batchNorm4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
        self.batchNorm5 = nn.BatchNorm1d(512)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(512*8, 2048)
        self.fc2 = nn.Linear(2048, 27)
        self.softmax = nn.Softmax()

    def forward(self, x):
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
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x
