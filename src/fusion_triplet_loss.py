import argparse
import os
import random

import numpy
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import RMSprop
from torch.utils.data import DataLoader
import datasets
import models
from datasets import get_transforms_from_config
import torch.nn.functional as functional
from models import MLP
from tools import load_yaml, get_predictions, get_num_correct_predictions
from sklearn.neighbors import KNeighborsClassifier

# Seed for reproducibility
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=2, help='Only applicable when cuda gpu is available')
parser.add_argument('--out_size', type=int, default=None, help='Override out_size if needed')
parser.add_argument('--use_knn', action='store_true', default=False, help='Use knn as classifier, else use MLP')
parser.add_argument('--param_file', type=str, default='parameters/utd_mhad/triplet_loss.yaml')
parser.add_argument('--new_vectors', action='store_true', default=False,
                    help='Retrieve new vectors, don\'t use the saved ones')
parser.add_argument('--mlp_epochs', type=int, default=100, help='Number of epochs to train the mlp')
parser.add_argument('--inertial_state', type=str, default=None)
parser.add_argument('--sdfdi_state', type=str, default=None)
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

if len(modalities) < 2:
    raise Exception('You have to choose at least two modalities')

# Synchronized lists
train_concat_scores = torch.tensor([], device=device)
train_concat_labels = None
test_concat_scores = torch.tensor([], device=device)
test_concat_labels = None

# Get concatenated vectors
if not os.path.exists('train_concat_scores.pt') or args.new_vectors:
    for modality in modalities:
        if param_config.get('modalities').get(modality) is None:
            break

        batch_size = param_config.get('modalities').get(modality).get('batch_size')
        train_transforms, test_transforms = get_transforms_from_config(
            param_config.get('modalities').get(modality).get('transforms'))
        train_dataset = SelectedDataset(modality=modality, transform=train_transforms, **train_dataset_kwargs)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_dataset = SelectedDataset(modality=modality, transform=test_transforms, **test_dataset_kwargs)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)
        model_kwargs = param_config.get('modalities').get(modality).get('model').get('kwargs')
        if args.out_size is not None:
            model_kwargs['out_size'] = args.out_size
        model = getattr(models, param_config.get('modalities').get(modality).get('model').get('class_name'))(
            *param_config.get('modalities').get(modality).get('model').get('args'),
            **model_kwargs
        )
        model.load_state_dict(torch.load(getattr(args, modality + '_state')))
        model = model.to(device)

        print('Getting train vectors from ' + modality)
        train_scores, train_labels = get_predictions(train_loader, model, device)
        train_concat_scores = torch.cat((train_concat_scores, train_scores), dim=1)
        if train_concat_labels is None:
            train_concat_labels = train_labels
        else:
            assert int((train_concat_labels - train_labels).sum()) == 0

        print('Getting test vectors from ' + modality)
        test_scores, test_labels = get_predictions(test_loader, model, device)
        test_concat_scores = torch.cat((test_concat_scores, test_scores), dim=1)
        if test_concat_labels is None:
            test_concat_labels = test_labels
        else:
            assert int((test_concat_labels - test_labels).sum()) == 0

    # L2 Normalize the concatenated vectors
    train_concat_scores = train_concat_scores.div(train_concat_scores.norm(p=2, dim=1, keepdim=True))
    train_concat_labels = train_concat_labels.div(train_concat_labels.norm(p=2, dim=1, keepdim=True))
    test_concat_scores = test_concat_scores.div(test_concat_scores.norm(p=2, dim=1, keepdim=True))
    test_concat_labels = test_concat_labels.div(test_concat_labels.norm(p=2, dim=1, keepdim=True))

    # Save concatenated vectors temporarily to avoid getting scores everytime
    print('Saving vectors to save time for next time')
    torch.save(train_concat_scores, '/tmp/train_concat_scores.pt')
    torch.save(train_concat_labels, '/tmp/train_concat_labels.pt')
    torch.save(test_concat_scores, '/tmp/test_concat_scores.pt')
    torch.save(test_concat_labels, '/tmp/test_concat_labels.pt')
else:
    print('Vectors exist. Loading...')
    train_concat_scores = torch.load('/tmp/train_concat_scores.pt')
    train_concat_labels = torch.load('/tmp/train_concat_labels.pt')
    test_concat_scores = torch.load('/tmp/test_concat_scores.pt')
    test_concat_labels = torch.load('/tmp/test_concat_labels.pt')

if args.use_knn:
    if device.type == 'cuda':
        train_concat_scores = train_concat_scores.cpu()
        train_concat_labels = train_concat_labels.cpu()
        test_concat_scores = test_concat_scores.cpu()
        test_concat_labels = test_concat_labels.cpu()

    classifier = KNeighborsClassifier(n_neighbors=21)
    classifier.fit(train_concat_scores, train_concat_labels.argmax(1))
    test_predictions = classifier.predict(test_concat_scores)

    test_accuracy = int((test_concat_labels.argmax(1) == torch.Tensor(test_predictions)).sum()) / \
                    test_concat_labels.shape[0]
else:
    mlp = MLP(input_size=train_concat_scores.shape[1], hidden_size=2048, out_size=27, norm_out=False)
    mlp = mlp.to(device)
    criterion = CrossEntropyLoss()
    optimizer = RMSprop(mlp.parameters(), lr=0.001)

    mlp.train()
    for epoch in range(args.mlp_epochs):
        scores = mlp(train_concat_scores)
        loss = criterion(scores, train_concat_labels.argmax(1))
        accu = get_num_correct_predictions(scores, train_concat_labels) / train_concat_labels.shape[0]

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch %d/%d: loss %.2f accu %.2f' % (epoch, args.mlp_epochs, loss.item(), accu))

    mlp.eval()
    test_scores = mlp(test_concat_scores)
    test_scores = functional.softmax(test_scores, 1)

    test_accuracy = get_num_correct_predictions(test_scores, test_concat_labels) / test_concat_scores.shape[0]

print('Test accuracy: %f' % test_accuracy)
