import argparse

import torch
from torch.utils.data import DataLoader

import datasets
import models
from configurators import AVAILABLE_MODALITIES
from datasets import get_transforms_from_config
from tools import load_yaml, plot_confusion_matrix, get_confusion_matrix, get_accuracy

parser = argparse.ArgumentParser()
parser.add_argument('--modality', choices=AVAILABLE_MODALITIES, default='inertial')
parser.add_argument('--gpu', type=int, default=0, help='Only applicable when cuda gpu is available')
parser.add_argument('--param_file', type=str, default='parameters/utd_mhad/default.yaml')
parser.add_argument('--saved_state', type=str, required=True)
args = parser.parse_args()

# Select device
cuda_device = 'cuda:%d' % args.gpu
device = torch.device(cuda_device if torch.cuda.is_available() else 'cpu')

# Load parameters from yaml file.
param_config = load_yaml(args.param_file)

# Assign parameters
modality = args.modality
SelectedDataset = getattr(datasets, param_config.get('dataset').get('class_name'))
_, transforms = get_transforms_from_config(param_config.get('modalities').get(modality).get('transforms'))
batch_size = param_config.get('modalities').get(modality).get('batch_size')
shuffle = param_config.get('dataset').get('shuffle')
model_class_name = param_config.get('modalities').get(modality).get('model').get('class_name')

# Load Data
test_dataset = SelectedDataset(modality=modality, train=False, transform=transforms)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)

# Initiate the model
model = getattr(models, model_class_name)(*param_config.get('modalities').get(modality).get('model').get('args'))
model = model.to(device)
model.load_state_dict(torch.load(args.saved_state))

print('Test Accuracy: %f' % get_accuracy(test_loader, model, device))
plot_confusion_matrix(
    cm=get_confusion_matrix(test_loader, model, device),
    title='Confusion Matrix - Percentage %',
    normalize=True,
    save=False,
    classes=test_dataset.get_class_names()
)
