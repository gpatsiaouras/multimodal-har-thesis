import argparse
import importlib
import random
import sys
import time
import json

import numpy
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import datasets
import models
from datasets import get_transforms_from_config, AVAILABLE_MODALITIES
from tools import load_yaml, train, get_confusion_matrix, get_accuracy, save_model
from visualizers import print_table, plot_confusion_matrix

# Seed for reproducibility
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(args):
    if args.test and args.saved_state is None:
        print('You have to use --saved_state when using --test, to specify the weights of the model')
        sys.exit(0)

    # Select device
    cuda_device = 'cuda:%d' % args.gpu
    device = torch.device(cuda_device if torch.cuda.is_available() else 'cpu')

    # Load parameters from yaml file.
    param_config = load_yaml(args.param_file)

    # Assign parameters
    modality = args.modality
    modality_config = param_config.get('modalities').get(modality)
    selected_dataset = getattr(datasets, param_config.get('dataset').get('class_name'))
    transforms, test_transforms = get_transforms_from_config(modality_config.get('transforms'))
    batch_size = modality_config.get('batch_size') if args.bs is None else args.bs
    num_epochs = modality_config.get('num_epochs') if args.epochs is None else args.epochs
    shuffle = param_config.get('dataset').get('shuffle')
    model_class_name = modality_config.get('model').get('class_name')
    criterion = modality_config.get('criterion').get('class_name')
    criterion_from = modality_config.get('criterion').get('from_module')
    optimizer = modality_config.get('optimizer').get('class_name')
    optimizer_from = modality_config.get('optimizer').get('from_module')
    optimizer_kwargs = modality_config.get('optimizer').get('kwargs')
    if args.lr:
        optimizer_kwargs['lr'] = args.lr
    train_dataset_kwargs = param_config.get('dataset').get('train_kwargs')
    validation_dataset_kwargs = param_config.get('dataset').get('validation_kwargs')
    test_dataset_kwargs = param_config.get('dataset').get('test_kwargs')

    # Load Data
    train_dataset = selected_dataset(modality=modality, transform=transforms, **train_dataset_kwargs)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
    validation_dataset = selected_dataset(modality=modality, transform=test_transforms, **validation_dataset_kwargs)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataset = selected_dataset(modality=modality, transform=test_transforms, **test_dataset_kwargs)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)

    # Initiate the model
    model_kwargs = modality_config.get('model').get('kwargs')
    if args.dr is not None:
        model_kwargs['dropout_rate'] = args.dr
    model = getattr(models, model_class_name)(
        *modality_config.get('model').get('args'),
        **modality_config.get('model').get('kwargs')
    )
    if args.test:
        model.load_state_dict(torch.load(args.saved_state))
    model = model.to(device)

    # Loss and optimizer
    criterion = getattr(importlib.import_module(criterion_from), criterion)()
    optimizer = getattr(importlib.import_module(optimizer_from), optimizer)(model.parameters(), **optimizer_kwargs)

    # Training procedure
    max_val_acc = -1
    max_train_acc = -1
    min_train_loss = -1
    min_val_loss = -1

    if not args.test:
        # Initiate Tensorboard writer with the given experiment name or generate an automatic one
        experiment = '%s_%s_%s_%s' % (
            selected_dataset.__name__,
            modality,
            args.param_file.split('/')[-1],
            time.strftime("%Y%m%d_%H%M", time.localtime())
        ) if args.experiment is None else args.experiment
        writer_name = '../logs/%s' % experiment
        writer = SummaryWriter(writer_name)

        # Print parameters
        print_table({
            'param_file': args.param_file,
            'experiment': experiment,
            'tensorboard_folder': writer_name,
            'dataset': selected_dataset.__name__,
            'criterion': type(criterion).__name__,
            'optimizer': type(optimizer).__name__,
            'modality': modality,
            'model': model.name,
            'learning_rate': optimizer_kwargs['lr'],
            'batch_size': batch_size,
            'num_epochs': num_epochs,
        })

        # Start training
        train_accs, val_accs, train_losses, val_losses = train(
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

        # Save last state of model
        save_model(model, '%s_last_state.pt' % experiment)

        max_val_acc = max(val_accs) if len(val_accs) > 0 else max_val_acc
        max_train_acc = max(train_accs) if len(train_accs) > 0 else max_train_acc
        min_train_loss = max(train_losses) if len(train_losses) > 0 else min_train_loss
        min_val_loss = max(val_losses) if len(val_losses) > 0 else min_val_loss

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
        writer.add_images('ConfusionMatrix/Train', cm_image_train, dataformats='CHW', global_step=num_epochs - 1)
        writer.add_images('ConfusionMatrix/Validation', cm_image_validation, dataformats='CHW',
                          global_step=num_epochs - 1)
        writer.add_images('ConfusionMatrix/Test', cm_image_test, dataformats='CHW', global_step=num_epochs - 1)
        print('Best validation accuracy: %f' % max(val_accs))

        writer.add_text('config', json.dumps(param_config, indent=2))
        writer.add_text('args', json.dumps(args.__dict__, indent=2))
        writer.flush()
        writer.close()

    test_accuracy = get_accuracy(test_loader, model, device)
    print('Test accuracy (not based on val): %f' % test_accuracy)

    return {
        'test_acc': test_accuracy,
        'max_train_acc': max_train_acc,
        'max_val_acc': max_val_acc,
        'min_train_loss': min_train_loss,
        'min_val_loss': min_val_loss
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', choices=AVAILABLE_MODALITIES, default='inertial')
    parser.add_argument('--test', action='store_true', help='Use this argument to test instead of train')
    parser.add_argument('--saved_state', type=str, default=None, help='Specify saved model when using --test')
    parser.add_argument('--gpu', type=int, default=2, help='Only applicable when cuda gpu is available')
    parser.add_argument('--param_file', type=str, default='parameters/utd_mhad/default.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--bs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--dr', type=float, default=None, help='Dropout rate')
    args = parser.parse_args()
    main(args)
