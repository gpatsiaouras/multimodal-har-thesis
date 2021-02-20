import argparse
import importlib
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import datasets
import models
from datasets import get_transforms_from_config, AVAILABLE_MODALITIES
from tools import load_yaml, train, get_confusion_matrix, get_accuracy
from visualizers import print_table, plot_confusion_matrix

torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--modality', choices=AVAILABLE_MODALITIES, default='inertial')
parser.add_argument('--gpu', type=int, default=0, help='Only applicable when cuda gpu is available')
parser.add_argument('--param_file', type=str, default='parameters/utd_mhad/default.yaml')
parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--experiment', type=str, default=None)
parser.add_argument('--bs', type=int, default=None)
parser.add_argument('--lr', type=float, default=None)
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
modality_config = param_config.get('modalities').get(modality)
SelectedDataset = getattr(datasets, param_config.get('dataset').get('class_name'))
transforms, test_transforms = get_transforms_from_config(modality_config.get('transforms'))
learning_rate = modality_config.get('learning_rate') if args.lr is None else args.lr
batch_size = modality_config.get('batch_size') if args.bs is None else args.bs
num_epochs = modality_config.get('num_epochs') if args.epochs is None else args.epochs
shuffle = param_config.get('dataset').get('shuffle')
model_class_name = modality_config.get('model').get('class_name')
criterion = modality_config.get('criterion').get('class_name')
criterion_from = modality_config.get('criterion').get('from_module')
optimizer = modality_config.get('optimizer').get('class_name')
optimizer_from = modality_config.get('optimizer').get('from_module')
train_dataset_kwargs = param_config.get('dataset').get('train_kwargs')
validation_dataset_kwargs = param_config.get('dataset').get('validation_kwargs')
test_dataset_kwargs = param_config.get('dataset').get('test_kwargs')

# Load Data
train_dataset = SelectedDataset(modality=modality, transform=transforms, **train_dataset_kwargs)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
validation_dataset = SelectedDataset(modality=modality, transform=test_transforms, **validation_dataset_kwargs)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=shuffle)
test_dataset = SelectedDataset(modality=modality, transform=test_transforms, **test_dataset_kwargs)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)

# Initiate the model
model = getattr(models, model_class_name)(
    *modality_config.get('model').get('args'),
    **modality_config.get('model').get('kwargs')
)
model = model.to(device)

# Loss and optimizer
criterion = getattr(importlib.import_module(criterion_from), criterion)()
optimizer = getattr(importlib.import_module(optimizer_from), optimizer)(model.parameters(), learning_rate)

# Initiate Tensorboard writer with the given experiment name or generate an automatic one
experiment = '%s_%s_%s_%s' % (
    SelectedDataset.__name__,
    modality,
    args.param_file.split('/')[-1],
    time.strftime("%Y%m%d_%H%M", time.localtime())
) if args.experiment is None else args.experiment
writer_name = '../logs/%s' % experiment
writer = SummaryWriter(writer_name)

# Start training
train_accs, validation_accs, train_losses, validation_losses, last_step = train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=validation_loader,
    num_epochs=num_epochs,
    device=device,
    experiment=experiment,
    writer=writer
)

cm_image_train = plot_confusion_matrix(
    cm=get_confusion_matrix(train_loader, model, device),
    title='Confusion Matrix - Training',
    normalize=False,
    save=False,
    classes=train_dataset.get_class_names(),
    show_figure=False
)
cm_image_validation = plot_confusion_matrix(
    cm=get_confusion_matrix(validation_loader, model, device),
    title='Confusion Matrix - Validation',
    normalize=False,
    save=False,
    classes=validation_dataset.get_class_names(),
    show_figure=False
)
cm_image_test = plot_confusion_matrix(
    cm=get_confusion_matrix(test_loader, model, device),
    title='Confusion Matrix - Test',
    normalize=False,
    save=False,
    classes=test_dataset.get_class_names(),
    show_figure=False
)

# Add confusion matrices for each dataset, mark it for the last step which is num_epochs - 1
writer.add_images('ConfusionMatrix/Train', cm_image_train, dataformats='CHW', global_step=last_step)
writer.add_images('ConfusionMatrix/Validation', cm_image_validation, dataformats='CHW', global_step=last_step)
writer.add_images('ConfusionMatrix/Test', cm_image_test, dataformats='CHW', global_step=last_step)

print('Best validation accuracy %f' % max(validation_accs))
print('Test accuracy %f' % get_accuracy(test_loader, model, device))

# Print parameters
print_table({
    'param_file': args.param_file,
    'experiment': experiment,
    'tensorboard_folder': writer_name,
    'dataset': SelectedDataset.__name__,
    'criterion': type(criterion).__name__,
    'optimizer': type(optimizer).__name__,
    'modality': modality,
    'model': model.name,
    'learning_rate': learning_rate,
    'batch_size': batch_size,
    'num_epochs': num_epochs,
})
