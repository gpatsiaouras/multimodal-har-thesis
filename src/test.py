import argparse

import torch
from torch.utils.data import DataLoader

import datasets
import models
from datasets import AVAILABLE_MODALITIES
from datasets import get_transforms_from_config
from tools import load_yaml, get_confusion_matrix, get_accuracy, get_predictions_with_knn
from visualizers import plot_confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--modality', choices=AVAILABLE_MODALITIES, default='inertial')
parser.add_argument('--gpu', type=int, default=0, help='Only applicable when cuda gpu is available')
parser.add_argument('--param_file', type=str, default='parameters/utd_mhad/default.yaml')
parser.add_argument('--saved_state', type=str, required=True)
parser.add_argument('--knn', action='store_true')
parser.add_argument('--n_neighbors', type=int, default=2)
args = parser.parse_args()

# Select device
cuda_device = 'cuda:%d' % args.gpu
device = torch.device(cuda_device if torch.cuda.is_available() else 'cpu')

# Load parameters from yaml file.
param_config = load_yaml(args.param_file)

# Assign parameters
modality = args.modality
SelectedDataset = getattr(datasets, param_config.get('dataset').get('class_name'))
_, test_transforms = get_transforms_from_config(param_config.get('modalities').get(modality).get('transforms'))
batch_size = param_config.get('modalities').get(modality).get('batch_size')
shuffle = param_config.get('dataset').get('shuffle')
model_class_name = param_config.get('modalities').get(modality).get('model').get('class_name')
train_dataset_kwargs = param_config.get('dataset').get('train_kwargs')
test_dataset_kwargs = param_config.get('dataset').get('test_kwargs')

# Load Data
test_dataset = SelectedDataset(modality=modality, transform=test_transforms, **test_dataset_kwargs)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)

# Initiate the model
model = getattr(models, model_class_name)(*param_config.get('modalities').get(modality).get('model').get('args'))
model = model.to(device)
model.load_state_dict(torch.load(args.saved_state))

if args.knn:
    # Use test transforms for train_dataset too, we don't want random stuff happening.
    train_dataset = SelectedDataset(modality=modality, transform=test_transforms, **train_dataset_kwargs)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
    cm, test_accuracy = get_predictions_with_knn(
        n_neighbors=args.n_neighbors,
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        device=device
    )
else:
    cm = get_confusion_matrix(test_loader, model, device)
    test_accuracy = get_accuracy(test_loader, model, device)

print('Test Accuracy: %f' % test_accuracy)
plot_confusion_matrix(
    cm=cm,
    title='Confusion Matrix - Percentage % - Test Loader',
    normalize=True,
    save=False,
    show_figure=True,
    classes=test_dataset.get_class_names()
)
