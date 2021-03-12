import argparse
import importlib
import json
import sys
import time

import torch
import random
import numpy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import datasets
from datasets import get_transforms_from_config, BalancedSampler, ConcatDataset
from models.medusa import Medusa
from tools import load_yaml, get_predictions_with_knn, plot_confusion_matrix, train_triplet_loss

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=9229, stdoutToServer=True, stderrToServer=True)


def get_train_val_test_datasets(dataset, modality, param_config):
    # Datasets configuration
    train_dataset_kwargs = param_config.get('dataset').get('train_kwargs')
    val_dataset_kwargs = param_config.get('dataset').get('validation_kwargs')
    test_dataset_kwargs = param_config.get('dataset').get('test_kwargs')
    transforms, test_transforms = get_transforms_from_config(
        param_config.get('modalities').get(modality).get('transforms'))

    # Load datasets and Loaders
    train_dataset = dataset(modality=modality, transform=transforms, **train_dataset_kwargs)
    val_dataset = dataset(modality=modality, transform=test_transforms, **val_dataset_kwargs)
    test_dataset = dataset(modality=modality, transform=test_transforms, **test_dataset_kwargs)

    return train_dataset, val_dataset, test_dataset


def train_and_test(args: argparse.Namespace):
    param_config = load_yaml(args.param_file)

    # Select device
    cuda_device = 'cuda:%d' % args.gpu
    device = torch.device(cuda_device if torch.cuda.is_available() else 'cpu')

    # Generic arguments
    num_epochs = param_config.get('general').get('num_epochs') if args.epochs is None else args.epochs
    num_neighbors = param_config.get('general').get('num_neighbors')

    # Load the selected dataset
    selected_dataset = getattr(datasets, param_config.get('dataset').get('class_name'))

    # Initiate datasets and loaders for each modality
    train_inertial, val_inertial, test_inertial = get_train_val_test_datasets(selected_dataset, 'inertial',
                                                                              param_config)
    train_sdfdi, val_sdfdi, test_sdfdi = get_train_val_test_datasets(selected_dataset, 'sdfdi', param_config)
    train_skeleton, val_skeleton, test_skeleton = get_train_val_test_datasets(selected_dataset, 'skeleton',
                                                                              param_config)

    # Prepare concat datasets and loaders
    train_dataset = ConcatDataset(train_inertial, train_sdfdi, train_skeleton)
    val_dataset = ConcatDataset(val_inertial, val_sdfdi, val_skeleton)
    test_dataset = ConcatDataset(test_inertial, test_sdfdi, test_skeleton)
    num_actions = len(train_dataset.datasets[0].actions)
    batch_size = param_config.get('general').get('batch_size')
    shuffle = param_config.get('general').get('shuffle')
    train_loader = DataLoader(dataset=train_dataset, batch_sampler=BalancedSampler(
        labels=train_dataset.labels,
        n_classes=num_actions,
        n_samples=param_config.get('general').get('num_samples')
    ))
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)
    class_names = train_dataset.get_class_names()

    # Load medusa network
    n1_kwargs = param_config.get('modalities').get('inertial').get('model').get('kwargs')
    n2_kwargs = param_config.get('modalities').get('sdfdi').get('model').get('kwargs')
    n3_kwargs = param_config.get('modalities').get('skeleton').get('model').get('kwargs')
    mlp_kwargs = param_config.get('general').get('mlp_kwargs')

    model = Medusa(mlp_kwargs, n1_kwargs, n2_kwargs, n3_kwargs)
    if args.test:
        model.load_state_dict(torch.load(args.saved_state))
    model = model.to(device)

    # Criterion, optimizer
    criterion = param_config.get('general').get('criterion').get('class_name')
    criterion_from = param_config.get('general').get('criterion').get('from_module')
    criterion_kwargs = param_config.get('general').get('criterion').get('kwargs')
    optimizer = param_config.get('general').get('optimizer').get('class_name')
    optimizer_from = param_config.get('general').get('optimizer').get('from_module')
    optimizer_kwargs = param_config.get('general').get('optimizer').get('kwargs')
    if args.margin:
        criterion_kwargs['margin'] = args.margin
    if args.semi_hard is not None:
        criterion_kwargs['semi_hard'] = args.semi_hard
    if args.lr:
        optimizer_kwargs['lr'] = args.lr
    criterion = getattr(importlib.import_module(criterion_from), criterion)(**criterion_kwargs)
    optimizer = getattr(importlib.import_module(optimizer_from), optimizer)(model.parameters(), **optimizer_kwargs)

    if not args.test:
        if args.experiment is None:
            datetime = time.strftime("%Y%m%d_%H%M", time.localtime())
            experiment = '%s_medusa' % datetime
        else:
            experiment = args.experiment
        writer = SummaryWriter('../logs/medusa/' + experiment)

        train_losses, val_losses, val_accuracies, train_accuracies = train_triplet_loss(model, criterion, optimizer,
                                                                                        class_names,
                                                                                        train_loader, val_loader,
                                                                                        num_epochs, device,
                                                                                        experiment, num_neighbors,
                                                                                        writer, verbose=True,
                                                                                        skip_accuracy=args.skip_accuracy)

    cm, test_acc, test_scores, test_labels = get_predictions_with_knn(
        n_neighbors=num_neighbors,
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        device=device
    )

    cm_image = plot_confusion_matrix(
        cm=cm,
        title='Confusion Matrix- Test Loader',
        normalize=False,
        save=False,
        show_figure=False,
        classes=test_dataset.get_class_names()
    )
    if not args.test:
        writer.add_images('ConfusionMatrix/Test', cm_image, dataformats='CHW', global_step=num_epochs - 1)
        writer.add_embedding(test_scores, metadata=[class_names[idx] for idx in test_labels.int().tolist()],
                             tag="test (%f%%)" % test_acc)
        writer.add_text('config', json.dumps(param_config, indent=2))
        writer.add_text('args', json.dumps(args.__dict__, indent=2))
        writer.flush()
        writer.close()
    print('Test acc: %.5f' % test_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=2, help='Only applicable when cuda gpu is available')
    parser.add_argument('--out_size', type=int, default=None, help='Override out_size if needed')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--margin', type=float, default=None)
    parser.add_argument('--dr', type=float, default=None, help='Dropout rate')
    parser.add_argument('--param_file', type=str, default='parameters/utd_mhad/medusa.yaml')
    parser.add_argument('--experiment', type=str, default=None, help='use auto for an autogenerated experiment name')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the mlp')
    parser.add_argument('--test', action='store_true', help='Use this argument to test instead of train')
    parser.add_argument('--saved_state', type=str, default=None, help='Specify saved model when using --test')
    parser.add_argument('--semi_hard', dest='semi_hard', action='store_true')
    parser.add_argument('--hard', dest='semi_hard', action='store_false')
    parser.add_argument('--skip_accuracy', action='store_true', default=False)
    parser.set_defaults(semi_hard=None, verbose=True)
    args = parser.parse_args()

    train_and_test(args)
