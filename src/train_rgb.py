import time

import torch
import torch.nn as nn
import torchvision.models as models
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor

from datasets import UtdMhadDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_classes = 27
batch_size = 32
learning_rate = 0.0001
num_epochs = 50

# Load Data
train_dataset = UtdMhadDataset(modality='sdfdi', train=True, transform=Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor()
]))
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Model
model = models.mobilenet_v2(num_classes=num_classes)
model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Statistics
start_time = time.time()
train_accuracies = []
test_accuracies = []
losses = []

# Train Network
for epoch in range(num_epochs):
    running_loss = .0
    num_correct = 0
    num_samples = 0
    epoch_start_time = time.time()
    for batch_idx, (data, labels) in enumerate(train_loader):
        # Turn inference mode off (in case)
        model.train()
        # Get data to cuda if possible
        data = data.to(device=device)
        labels = labels.to(device=device)

        # forward
        scores = model(data.float())
        _, input_indices = torch.max(labels, dim=1)
        loss = criterion(scores, input_indices)

        # train accuracy
        pred_max_val, pred_max_id = scores.max(1)
        num_correct += int((labels.argmax(1) == pred_max_id).sum())
        num_samples += len(pred_max_id)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

        running_loss += loss

    # print statistics
    # Loss
    train_loss = running_loss / len(train_loader)
    losses.append(train_loss)
    # Train accuracy
    train_acc = float(num_correct) / float(num_samples) * 100
    train_accuracies.append(train_acc)
    # Timing
    total_epoch_time = time.time() - epoch_start_time
    total_time = time.time() - start_time
    print('=== Epoch %d ===' % (epoch + 1))
    print('loss: %.3f' % train_loss)
    print('accuracy: %f' % train_acc)
    print('epoch duration: %s' % time.strftime('%H:%M:%S', time.gmtime(total_epoch_time)))
    print('total duration until now: %s' % time.strftime('%H:%M:%S', time.gmtime(total_time)))