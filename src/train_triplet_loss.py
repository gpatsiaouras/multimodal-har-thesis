import argparse
import importlib
import sys
import random
import torch
import json
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import datasets
import models
import numpy
from datasets import BalancedSampler, AVAILABLE_MODALITIES, get_transforms_from_config
from tools import train_triplet_loss, get_predictions_with_knn, load_yaml
from visualizers import plot_confusion_matrix

# Seed for reproducibility
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_and_test(args: argparse.Namespace):
    if args.test and args.saved_state is None:
        print('You have to use --saved_state when using --test, to specify the weights of the model')
        sys.exit(0)

    # Select device
    cuda_device = 'cuda:%d' % args.gpu
    device = torch.device(cuda_device if torch.cuda.is_available() else 'cpu')

    # Load parameters from yaml file.
    param_config = load_yaml(args.param_file)

    # Basic parameters
    modality = args.modality
    modality_config = param_config.get('modalities').get(modality)

    # Hyper params
    num_neighbors = modality_config.get('num_neighbors') if args.num_neighbors is None else args.num_neighbors
    batch_size = modality_config.get('batch_size')
    num_epochs = modality_config.get('num_epochs') if args.epochs is None else args.epochs
    shuffle = param_config.get('dataset').get('shuffle')

    # Criterion, optimizer and scheduler
    model_class_name = modality_config.get('model').get('class_name')
    criterion = modality_config.get('criterion').get('class_name')
    criterion_from = modality_config.get('criterion').get('from_module')
    criterion_kwargs = modality_config.get('criterion').get('kwargs')
    if args.margin:
        criterion_kwargs['margin'] = args.margin
    if args.semi_hard is not None:
        criterion_kwargs['semi_hard'] = args.semi_hard
    optimizer = modality_config.get('optimizer').get('class_name')
    optimizer_from = modality_config.get('optimizer').get('from_module')
    optimizer_kwargs = modality_config.get('optimizer').get('kwargs')
    if args.lr:
        optimizer_kwargs['lr'] = args.lr
    scheduler_class_name = modality_config.get('scheduler').get('class_name')
    scheduler_from = modality_config.get('scheduler').get('from_module')
    scheduler_kwargs = modality_config.get('scheduler').get('kwargs')

    # Dataset config
    selected_dataset = getattr(datasets, param_config.get('dataset').get('class_name'))
    transforms, test_transforms = get_transforms_from_config(
        param_config.get('modalities').get(modality).get('transforms'))
    train_dataset_kwargs = param_config.get('dataset').get('train_kwargs')
    validation_dataset_kwargs = param_config.get('dataset').get('validation_kwargs')
    test_dataset_kwargs = param_config.get('dataset').get('test_kwargs')

    # Load Data
    train_dataset = selected_dataset(modality=modality, transform=transforms, **train_dataset_kwargs)
    num_actions = len(train_dataset.actions)
    train_loader = DataLoader(dataset=train_dataset, batch_sampler=BalancedSampler(
        labels=train_dataset.labels,
        n_classes=num_actions,
        n_samples=modality_config['num_samples']
    ))
    validation_dataset = selected_dataset(modality=modality, transform=test_transforms, **validation_dataset_kwargs)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataset = selected_dataset(modality=modality, transform=test_transforms, **test_dataset_kwargs)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)
    class_names = train_dataset.get_class_names()

    # Initiate the model
    model_kwargs = modality_config.get('model').get('kwargs')
    if args.out_size is not None:
        model_kwargs['out_size'] = args.out_size
    if args.dr is not None and modality != 'sdfdi':
        model_kwargs['dropout_rate'] = args.dr
    model = getattr(models, model_class_name)(
        *modality_config.get('model').get('args'),
        **model_kwargs
    )
    if args.test:
        model.load_state_dict(torch.load(args.saved_state))
    model = model.to(device)

    # Loss, optimizer and scheduler
    criterion = getattr(importlib.import_module(criterion_from), criterion)(**criterion_kwargs)
    optimizer = getattr(importlib.import_module(optimizer_from), optimizer)(model.parameters(), **optimizer_kwargs)
    scheduler = None
    if not args.no_scheduler:
        scheduler = getattr(importlib.import_module(scheduler_from), scheduler_class_name)(optimizer,
                                                                                           **scheduler_kwargs)

    # Training procedure:
    # 1. Instantiate tensorboard writer
    # 2. Run training with triplet loss
    max_val_acc = -1
    max_train_acc = -1
    min_train_loss = -1
    min_val_loss = -1
    if not args.test:
        if args.experiment is None:
            print('Specify an experiment name by using --experiment argument')
            sys.exit(0)
        elif args.experiment == 'auto':
            experiment = '%s_%s_TL_A%s_M%s_LR%s_%s_%sep' % (
                model.name, modality, str(num_actions), str(criterion_kwargs['margin']), str(optimizer_kwargs['lr']),
                'semi_hard' if criterion_kwargs['semi_hard'] else 'hard', num_epochs)
        else:
            experiment = args.experiment
        if args.verbose:
            print('Experiment:  %s' % experiment)
        writer = SummaryWriter('../logs/' + experiment)

        train_losses, val_losses, val_accs, train_accs = train_triplet_loss(model=model,
                                                                            criterion=criterion,
                                                                            optimizer=optimizer,
                                                                            scheduler=scheduler,
                                                                            class_names=class_names,
                                                                            train_loader=train_loader,
                                                                            val_loader=validation_loader,
                                                                            num_epochs=num_epochs,
                                                                            device=device,
                                                                            experiment=experiment,
                                                                            writer=writer,
                                                                            n_neighbors=num_neighbors,
                                                                            verbose=args.verbose
                                                                            )
        max_val_acc = max(val_accs) if len(val_accs) > 0 else max_val_acc
        max_train_acc = max(train_accs) if len(train_accs) > 0 else max_train_acc
        min_train_loss = max(train_losses) if len(train_losses) > 0 else min_train_loss
        min_val_loss = max(val_losses) if len(val_losses) > 0 else min_val_loss

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
        writer.add_hparams({
            'learning_rate': optimizer_kwargs['lr'],
            'margin': criterion_kwargs['margin'],
            'semi_hard': criterion_kwargs['semi_hard'],
            'out_size': model_kwargs['out_size']
        }, {
            'hparam/val_acc': max_val_acc,
            'hparam/test_acc': test_acc,
            'hparam/train_acc': max_train_acc
        }, run_name='hparams')
        writer.add_images('ConfusionMatrix/Test', cm_image, dataformats='CHW', global_step=num_epochs - 1)
        writer.add_embedding(test_scores, metadata=[class_names[idx] for idx in test_labels.int().tolist()],
                             tag="test (%f%%)" % test_acc)
        writer.add_text('config', json.dumps(param_config, indent=2))
        writer.add_text('args', json.dumps(args.__dict__, indent=2))
        writer.flush()
        writer.close()

        return {
            'lr': optimizer_kwargs['lr'],
            'margin': criterion_kwargs['margin'],
            'semi_hard': criterion_kwargs['semi_hard'],
            'out_size': model_kwargs['out_size'],
            'test_acc': test_acc,
            'max_train_acc': max_train_acc,
            'max_val_acc': max_val_acc,
            'min_train_loss': min_train_loss,
            'min_val_loss': min_val_loss
        }

    return {'test_acc': test_acc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', choices=AVAILABLE_MODALITIES, default='inertial')
    parser.add_argument('--gpu', type=int, default=2, help='Only applicable when cuda gpu is available')
    parser.add_argument('--param_file', type=str, default='parameters/utd_mhad/triplet_loss.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--num_neighbors', type=int, default=None, help='Number of neighbors for the KNN')
    parser.add_argument('--experiment', type=str, default=None, help='use auto for an autogenerated experiment name')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--margin', type=float, default=None)
    parser.add_argument('--out_size', type=int, default=None)
    parser.add_argument('--dr', type=float, default=None, help='Dropout rate')
    parser.add_argument('--semi_hard', dest='semi_hard', action='store_true')
    parser.add_argument('--hard', dest='semi_hard', action='store_false')
    parser.add_argument('--test', action='store_true', help='Use this argument to test instead of train')
    parser.add_argument('--saved_state', type=str, default=None, help='Specify saved model when using --test')
    parser.add_argument('--no_scheduler', action='store_true', help='Don\'t use a scheduler')
    parser.set_defaults(no_scheduler=False, semi_hard=None, verbose=True)
    args = parser.parse_args()

    training_testing_results = train_and_test(args)

    print('Test accuracy: %f' % training_testing_results['test_acc'])
