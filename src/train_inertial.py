import os
import sys

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from datasets import UtdMhadDataset
from models import CNN1D
from tools import load_yaml, train
from transforms import Sampler, Compose, Flatten, FilterDimensions, Jittering
from visualizers import print_table

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Seed number generator
torch.manual_seed(0)

# Load parameters from yaml file.
if len(sys.argv) == 2:
    yaml_file = sys.argv[1]
else:
    yaml_file = os.path.join(os.path.dirname(__file__), 'parameters/inertial/default.yaml')
param_config = load_yaml(yaml_file)

# Assign hyper parameters
num_classes = param_config.get('dataset').get('num_classes')
learning_rate = param_config.get('hyper_parameters').get('learning_rate')
batch_size = param_config.get('hyper_parameters').get('batch_size')
num_epochs = param_config.get('hyper_parameters').get('num_epochs')
jitter_factor = param_config.get('hyper_parameters').get('jitter_factor')

# Print parameters
print_table({
    'num_classes': num_classes,
    'learning_rate': learning_rate,
    'batch_size': batch_size,
    'num_epochs': num_epochs,
    'jitter_factor': jitter_factor
})

# Load Data
train_dataset = UtdMhadDataset(modality='inertial', train=True, transform=Compose([
    Sampler(107),
    FilterDimensions([0, 1, 2]),
    Jittering(jitter_factor),
    Flatten(),
]))
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = UtdMhadDataset(modality='inertial', train=False, transform=Compose([
    Sampler(107),
    FilterDimensions([0, 1, 2]),
    Jittering(jitter_factor),
    Flatten(),
]))
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = CNN1D().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0)

train(model, criterion, optimizer, train_loader, test_loader, num_epochs, batch_size, device)
