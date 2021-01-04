import os
import sys

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from datasets import UtdMhadDataset
from models import BiGRU, BiLSTM
from tools import train, load_yaml
from transforms import Resize, Compose, ToSequence, FilterJoints, Normalize, RandomEulerRotation
from visualizers import print_table

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Seed number generator
torch.manual_seed(0)

# Load parameters from yaml file.
if len(sys.argv) == 2:
    yaml_file = sys.argv[1]
else:
    yaml_file = os.path.join(os.path.dirname(__file__), 'parameters/skeleton/default.yaml')
param_config = load_yaml(yaml_file)

# Assign training hyper parameters
learning_rate = param_config.get('training').get('learning_rate')
batch_size = param_config.get('training').get('batch_size')
num_epochs = param_config.get('training').get('num_epochs')

# Data related
num_classes = param_config.get('dataset').get('num_classes')
selected_joints = param_config.get('dataset').get('selected_joints')
num_frames = param_config.get('dataset').get('num_frames')

# Model params
model_type = param_config.get('model').get('model_type')
hidden_size = param_config.get('model').get('hidden_size')
num_layers = param_config.get('model').get('num_layers')
input_size = len(selected_joints)  # number of joints as input size
sequence_length = num_frames * 3  # features(frames) x 3 dimensions xyz per frame

# Print parameters
print_table({
    'num_classes': num_classes,
    'learning_rate': learning_rate,
    'batch_size': batch_size,
    'num_epochs': num_epochs,
    'num_frames': num_frames,
    'model_type': model_type,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
    'input_size': input_size,
    'sequence_length': sequence_length,
})

# Load Data
train_dataset = UtdMhadDataset(modality='skeleton', train=True, transform=Compose([
    Normalize((0, 2)),
    Resize(num_frames),
    FilterJoints(selected_joints),
    RandomEulerRotation(-5, 5, 5),
    ToSequence(sequence_length, input_size)
]))
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataset = UtdMhadDataset(modality='skeleton', train=False, transform=Compose([
    Normalize((0, 2)),
    Resize(num_frames),
    FilterJoints(selected_joints),
    ToSequence(sequence_length, input_size)
]))
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Model
if model_type is 'lstm':
    model = BiLSTM(batch_size, input_size, hidden_size, num_layers, num_classes, device).to(device)
else:
    model = BiGRU(batch_size, input_size, hidden_size, num_layers, num_classes, device).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

train(model, criterion, optimizer, train_loader, test_loader, num_epochs, batch_size, device)
