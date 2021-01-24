import argparse
import importlib

import torch
from torch.utils.data import DataLoader

import datasets
import models
from configurators import AVAILABLE_MODALITIES
from datasets import get_transforms_from_config
from tools import load_yaml, train
from visualizers import print_table


parser = argparse.ArgumentParser()
parser.add_argument('--modality', choices=AVAILABLE_MODALITIES, default='inertial')
parser.add_argument('--gpu', type=int, default=0, help='Only applicable when cuda gpu is available')
parser.add_argument('--param_file', type=str, default='parameters/utd_mhad/default.yaml')
args = parser.parse_args()

# Select device
cuda_device = 'cuda:%d' % args.gpu
device = torch.device(cuda_device if torch.cuda.is_available() else 'cpu')

# Seed number generator
torch.manual_seed(0)

# Load parameters from yaml file.
param_config = load_yaml(args.param_file)

# Assign parameters
modality = args.modality
SelectedDataset = getattr(datasets, param_config.get('dataset').get('class_name'))
transforms, test_transforms = get_transforms_from_config(param_config.get('modalities').get(modality).get('transforms'))
learning_rate = param_config.get('modalities').get(modality).get('learning_rate')
batch_size = param_config.get('modalities').get(modality).get('batch_size')
num_epochs = param_config.get('modalities').get(modality).get('num_epochs')
shuffle = param_config.get('dataset').get('shuffle')
model_class_name = param_config.get('modalities').get(modality).get('model').get('class_name')
criterion = param_config.get('modalities').get(modality).get('criterion').get('class_name')
criterion_from = param_config.get('modalities').get(modality).get('criterion').get('from_module')
optimizer = param_config.get('modalities').get(modality).get('optimizer').get('class_name')
optimizer_from = param_config.get('modalities').get(modality).get('optimizer').get('from_module')

# Load Data
train_dataset = SelectedDataset(modality=modality, train=True, transform=transforms)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
test_dataset = SelectedDataset(modality=modality, train=False, transform=test_transforms)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)

# Initiate the model
model = getattr(models, model_class_name)(*param_config.get('modalities').get(modality).get('model').get('args'))
model = model.to(device)

# Loss and optimizer
criterion = getattr(importlib.import_module(criterion_from), criterion)()
optimizer = getattr(importlib.import_module(optimizer_from), optimizer)(model.parameters(), learning_rate)

# Print parameters
print_table({
    'param_file': args.param_file,
    'dataset': SelectedDataset.__name__,
    'modality': modality,
    'model': model.name,
    'learning_rate': learning_rate,
    'batch_size': batch_size,
    'num_epochs': num_epochs,
})

train(model, criterion, optimizer, train_loader, test_loader, num_epochs, batch_size, device)
