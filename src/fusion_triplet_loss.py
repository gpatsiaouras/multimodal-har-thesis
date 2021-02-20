import argparse

import torch
from torch.utils.data import DataLoader
import datasets
import models
from datasets import get_transforms_from_config
from tools import load_yaml, get_predictions
from sklearn.neighbors import KNeighborsClassifier

torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='Only applicable when cuda gpu is available')
parser.add_argument('--param_file', type=str, default='parameters/utd_mhad/triplet_loss.yaml')
parser.add_argument('--inertial_state', type=str, default=None, required=True)
parser.add_argument('--sdfdi_state', type=str, default=None, required=True)
parser.add_argument('--skeleton_state', type=str, default=None)
args = parser.parse_args()

# Select device
cuda_device = 'cuda:%d' % args.gpu
device = torch.device(cuda_device if torch.cuda.is_available() else 'cpu')

# Load parameters from yaml file.
param_config = load_yaml(args.param_file)
shuffle = False
SelectedDataset = getattr(datasets, param_config.get('dataset').get('class_name'))
train_dataset_kwargs = param_config.get('dataset').get('train_kwargs')
test_dataset_kwargs = param_config.get('dataset').get('test_kwargs')

# modalities
modalities = []
if args.inertial_state is not None:
    modalities.append('inertial')
if args.sdfdi_state is not None:
    modalities.append('sdfdi')
if args.skeleton_state is not None:
    modalities.append('skeleton')

# Synchronized lists
train_datasets = []
train_loaders = []
test_datasets = []
test_loaders = []
models_list = []
train_concat_scores = torch.tensor([], device=device)
train_concat_labels = None
test_concat_scores = torch.tensor([], device=device)
test_concat_labels = None

# Populate
for modality in modalities:
    if param_config.get('modalities').get(modality) is None:
        break

    batch_size = param_config.get('modalities').get(modality).get('batch_size')
    train_transforms, test_transforms = get_transforms_from_config(
        param_config.get('modalities').get(modality).get('transforms'))
    train_datasets.append(SelectedDataset(modality=modality, transform=train_transforms, **train_dataset_kwargs))
    train_loaders.append(DataLoader(dataset=train_datasets[-1], batch_size=batch_size, shuffle=shuffle))
    test_datasets.append(SelectedDataset(modality=modality, transform=test_transforms, **test_dataset_kwargs))
    test_loaders.append(DataLoader(dataset=test_datasets[-1], batch_size=batch_size, shuffle=shuffle))
    model = getattr(models, param_config.get('modalities').get(modality).get('model').get('class_name'))(
        *param_config.get('modalities').get(modality).get('model').get('args'),
        **param_config.get('modalities').get(modality).get('model').get('kwargs')
    )
    model.load_state_dict(torch.load(getattr(args, modality + '_state')))
    model = model.to(device)
    models_list.append(model)

    print('Getting train vectors from ' + modality)
    train_scores, train_labels = get_predictions(train_loaders[-1], model, device)
    train_concat_scores = torch.cat((train_concat_scores, train_scores), dim=1)
    if train_concat_labels is None:
        train_concat_labels = train_labels
    else:
        assert int((train_concat_labels - train_labels).sum()) == 0

    print('Getting test vectors from ' + modality)
    test_scores, test_labels = get_predictions(train_loaders[-1], model, device)
    test_concat_scores = torch.cat((test_concat_scores, test_scores), dim=1)
    if test_concat_labels is None:
        test_concat_labels = test_labels
    else:
        assert int((test_concat_labels - test_labels).sum()) == 0

if device.type == 'cuda':
    train_concat_scores = train_concat_scores.cpu()
    train_concat_labels = train_concat_labels.cpu()
    test_concat_scores = test_concat_scores.cpu()
    test_concat_labels = test_concat_labels.cpu()

classifier = KNeighborsClassifier(n_neighbors=4)
classifier.fit(train_concat_scores, train_concat_labels.argmax(1))
test_predictions = classifier.predict(test_concat_scores)

test_accuracy = int((test_concat_labels.argmax(1) == torch.Tensor(test_predictions)).sum()) / test_concat_labels.shape[0]
print('Test accuracy: %f' % test_accuracy)
