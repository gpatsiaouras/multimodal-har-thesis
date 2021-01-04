import os
import sys

import torch
import torch.nn as nn
import torchvision.models as models
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, RandomResizedCrop

from datasets import UtdMhadDataset
from tools import load_yaml, train
from visualizers import print_table

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Seed number generator
torch.manual_seed(0)

# Load parameters from yaml file.
if len(sys.argv) == 2:
    yaml_file = sys.argv[1]
else:
    yaml_file = os.path.join(os.path.dirname(__file__), 'parameters/rgb/default.yaml')
param_config = load_yaml(yaml_file)

# Assign hyper parameters
num_classes = param_config.get('dataset').get('num_classes')
learning_rate = param_config.get('hyper_parameters').get('learning_rate')
batch_size = param_config.get('hyper_parameters').get('batch_size')
num_epochs = param_config.get('hyper_parameters').get('num_epochs')

# Print parameters
print_table({
    'num_classes': num_classes,
    'learning_rate': learning_rate,
    'batch_size': batch_size,
    'num_epochs': num_epochs,
})

# Load Data
train_dataset = UtdMhadDataset(modality='sdfdi', train=True, transform=Compose([
    RandomResizedCrop(480),
    Resize(224),
    ToTensor()
]))
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = UtdMhadDataset(modality='sdfdi', train=False, transform=Compose([
    Resize(224),
    ToTensor()
]))
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Model
model = models.mobilenet_v2(pretrained=True)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=False),
    nn.Linear(model.last_channel, 2048),
    nn.Linear(2048, num_classes)
)
model.name = 'mobilenet_v2'
model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train(model, criterion, optimizer, train_loader, test_loader, num_epochs, batch_size, device)
