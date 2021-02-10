import argparse
import importlib

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import datasets
import models
from datasets import BalancedSampler, AVAILABLE_MODALITIES, get_transforms_from_config
from tools import train_triplet_loss, get_predictions_with_knn, load_yaml
from visualizers import plot_confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--modality', choices=AVAILABLE_MODALITIES, default='inertial')
parser.add_argument('--gpu', type=int, default=0, help='Only applicable when cuda gpu is available')
parser.add_argument('--param_file', type=str, default='parameters/utd_mhad/triplet_loss.yaml')
parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--experiment', type=str, default=None)
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--margin', type=float, default=None)
args = parser.parse_args()

# Select device
cuda_device = 'cuda:%d' % args.gpu
device = torch.device(cuda_device if torch.cuda.is_available() else 'cpu')

# Load parameters from yaml file.
param_config = load_yaml(args.param_file)

# Basic parameters
modality = args.modality
modality_config = param_config.get('modalities').get(modality)

# Hyper params
num_neighbors = modality_config.get('num_neighbors')
learning_rate = modality_config.get('learning_rate') if args.lr is None else args.lr
batch_size = modality_config.get('batch_size')
num_epochs = modality_config.get('num_epochs') if args.epochs is None else args.epochs
shuffle = param_config.get('dataset').get('shuffle')

# Criterion and optimizer
model_class_name = modality_config.get('model').get('class_name')
criterion = modality_config.get('criterion').get('class_name')
criterion_from = modality_config.get('criterion').get('from_module')
criterion_args = modality_config.get('criterion').get('kwargs')
if args.margin:
    criterion_args['margin'] = args.margin
optimizer = modality_config.get('optimizer').get('class_name')
optimizer_from = modality_config.get('optimizer').get('from_module')

# Dataset config
SelectedDataset = getattr(datasets, param_config.get('dataset').get('class_name'))
transforms, test_transforms = get_transforms_from_config(param_config.get('modalities').get(modality).get('transforms'))
train_dataset_kwargs = param_config.get('dataset').get('train_kwargs')
validation_dataset_kwargs = param_config.get('dataset').get('validation_kwargs')
test_dataset_kwargs = param_config.get('dataset').get('test_kwargs')

# Load Data
train_dataset = SelectedDataset(modality=modality, transform=transforms, **train_dataset_kwargs)
num_actions = len(train_dataset.actions)
train_loader = DataLoader(dataset=train_dataset, batch_sampler=BalancedSampler(
    dataset=train_dataset,
    n_classes=num_actions,
    n_samples=modality_config['num_samples'],
    sampler=torch.utils.data.sampler.Sampler(train_dataset),
    batch_size=batch_size,
    drop_last=False
))
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
criterion = getattr(importlib.import_module(criterion_from), criterion)(**criterion_args)
optimizer = getattr(importlib.import_module(optimizer_from), optimizer)(model.parameters(), learning_rate)

# Tensorboard writer initialization
experiment = 'MarLrExp_%s_%s_TL_A%s_M%s_LR%s_%s_%sep' % (
    model.name, modality, str(num_actions), str(criterion_args['margin']), str(learning_rate),
    'semi_hard' if criterion_args['semi_hard'] else 'hard', num_epochs) if args.experiment is None else args.experiment
print('Experiment:  %s' % experiment)
writer = SummaryWriter('../logs/' + experiment)

min_val_loss, max_val_acc, last_step = train_triplet_loss(model=model,
                                                          criterion=criterion,
                                                          optimizer=optimizer,
                                                          class_names=train_dataset.get_class_names(),
                                                          train_loader=train_loader,
                                                          val_loader=validation_loader,
                                                          num_epochs=num_epochs,
                                                          device=device,
                                                          experiment=experiment,
                                                          writer=writer,
                                                          n_neighbors=num_neighbors
                                                          )

cm, test_accuracy = get_predictions_with_knn(
    n_neighbors=num_neighbors,
    train_loader=train_loader,
    test_loader=test_loader,
    model=model,
    device=device
)

print('Test accuracy: %f' % test_accuracy)
cm_image = plot_confusion_matrix(
    cm=cm,
    title='Confusion Matrix- Test Loader',
    normalize=False,
    save=False,
    show_figure=False,
    classes=test_dataset.get_class_names()
)
writer.add_images('ConfusionMatrix/Test', cm_image, dataformats='CHW', global_step=last_step)
writer.add_hparams({'learning_rate': learning_rate, 'margin': criterion_args['margin']},
                   {'val_accuracy': max_val_acc, 'val_loss': min_val_loss})
writer.flush()
writer.close()
