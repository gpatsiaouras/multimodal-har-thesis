import argparse
import random

import numpy
import torch
from torch.utils.data import DataLoader

import datasets
import models
from datasets import get_transforms_from_config
from models import ELM
from tools import get_predictions, load_yaml, get_fused_scores, get_fused_labels

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
    # Select device
    cuda_device = 'cuda:%d' % args.gpu
    device = torch.device(cuda_device if torch.cuda.is_available() else 'cpu')

    # Load parameters from yaml file.
    param_config = load_yaml(args.param_file)
    shuffle = False
    selected_dataset = getattr(datasets, param_config.get('dataset').get('class_name'))
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
    train_all_scores = None
    train_all_labels = None
    test_all_scores = None
    test_all_labels = None

    if len(modalities) < 2:
        raise RuntimeError('Cannot fuse with less than two modalities')

    for modality in modalities:
        if param_config.get('modalities').get(modality) is None:
            break

        batch_size = 16
        train_transforms, test_transforms = get_transforms_from_config(
            param_config.get('modalities').get(modality).get('transforms'))
        train_dataset = selected_dataset(modality=modality, transform=train_transforms, **train_dataset_kwargs)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_dataset = selected_dataset(modality=modality, transform=test_transforms, **test_dataset_kwargs)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)
        model = getattr(models, param_config.get('modalities').get(modality).get('model').get('class_name'))(
            *param_config.get('modalities').get(modality).get('model').get('args'),
            **param_config.get('modalities').get(modality).get('model').get('kwargs')
        )
        model.load_state_dict(torch.load(getattr(args, modality + '_state')))
        model.skip_last_fc = True
        model = model.to(device)

        print('Getting train vectors from ' + modality)
        train_scores, train_labels = get_predictions(train_loader, model, device)
        train_all_scores = get_fused_scores(train_all_scores, train_scores, args.rule)
        train_all_labels = get_fused_labels(train_all_labels, train_labels)
        print('Getting test vectors from ' + modality)
        test_scores, test_labels = get_predictions(test_loader, model, device)
        test_all_scores = get_fused_scores(test_all_scores, test_scores, args.rule)
        test_all_labels = get_fused_labels(test_all_labels, test_labels)

    # Elm initialization
    elm = ELM(input_size=train_all_scores.shape[1], num_classes=train_all_labels.shape[1], hidden_size=args.hidden_size,
              device=device)

    # Fit the ELM in training data. For labels use any of the three, they are all the same since shuffle is off.
    print('Training elm network...')
    elm.fit(train_all_scores, train_all_labels)

    # Get accuracy on test data
    accuracy = elm.evaluate(test_all_scores, test_all_labels)

    print('ELM Accuracy: %f' % accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=2, help='Only applicable when cuda gpu is available')
    parser.add_argument('--param_file', type=str, default='parameters/utd_mhad/default.yaml')
    parser.add_argument('--hidden_size', type=int, default=8192)
    parser.add_argument('--inertial_state', type=str, default=None)
    parser.add_argument('--sdfdi_state', type=str, default=None)
    parser.add_argument('--skeleton_state', type=str, default=None)
    parser.add_argument('--rule', choices=['concat', 'avg'], default='avg')
    args = parser.parse_args()
    main(args)
