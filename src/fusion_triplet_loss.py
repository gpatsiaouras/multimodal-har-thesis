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
from models import MLP
from tools import load_yaml, get_predictions, get_fused_scores, get_fused_labels, \
    train_simple, get_accuracy_simple
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

MLP_STATE_FILE = '/tmp/mlp_state.pt'
TRAIN_SCORES_FILE = '/tmp/train_concat_scores.pt'
TRAIN_LABELS_FILE = '/tmp/train_concat_labels.pt'
TEST_SCORES_FILE = '/tmp/test_concat_scores.pt'
TEST_LABELS_FILE = '/tmp/test_concat_labels.pt'


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

    if len(modalities) < 2:
        raise Exception('Cannot fuse with less than two modalities')

    # Synchronized lists
    train_concat_scores = None
    train_concat_labels = None
    test_concat_scores = None
    test_concat_labels = None

    # Get concatenated vectors
    if not os.path.exists('/tmp/train_concat_scores.pt') or args.new_vectors:
        for modality in modalities:
            if param_config.get('modalities').get(modality) is None:
                break

            batch_size = param_config.get('modalities').get(modality).get('batch_size')
            train_transforms, test_transforms = get_transforms_from_config(
                param_config.get('modalities').get(modality).get('transforms'))
            train_dataset = selected_dataset(modality=modality, transform=train_transforms, **train_dataset_kwargs)
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
            test_dataset = selected_dataset(modality=modality, transform=test_transforms, **test_dataset_kwargs)
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
            train_concat_scores = get_fused_scores(train_concat_scores, train_scores, args.rule)
            train_concat_labels = get_fused_labels(train_concat_labels, train_labels)

            print('Getting test vectors from ' + modality)
            test_scores, test_labels = get_predictions(test_loader, model, device)
            test_concat_scores = get_fused_scores(test_concat_scores, test_scores, args.rule)
            test_concat_labels = get_fused_labels(test_concat_labels, test_labels)

        # L2 Normalize the concatenated vectors
        train_concat_scores = train_concat_scores.div(train_concat_scores.norm(p=2, dim=1, keepdim=True))
        test_concat_scores = test_concat_scores.div(test_concat_scores.norm(p=2, dim=1, keepdim=True))

        # Save concatenated vectors temporarily to avoid getting scores everytime
        print('Saving vectors to save time for next time')
        torch.save(train_concat_scores, TRAIN_SCORES_FILE)
        torch.save(train_concat_labels, TRAIN_LABELS_FILE)
        torch.save(test_concat_scores, TEST_SCORES_FILE)
        torch.save(test_concat_labels, TEST_LABELS_FILE)
    else:
        print('Vectors exist. Loading...')
        train_concat_scores = torch.load(TRAIN_SCORES_FILE)
        train_concat_labels = torch.load(TRAIN_LABELS_FILE)
        test_concat_scores = torch.load(TEST_SCORES_FILE)
        test_concat_labels = torch.load(TEST_LABELS_FILE)

    if args.use_knn:
        if device.type == 'cuda':
            train_concat_scores = train_concat_scores.cpu()
            train_concat_labels = train_concat_labels.cpu()
            test_concat_scores = test_concat_scores.cpu()
            test_concat_labels = test_concat_labels.cpu()

        classifier = KNeighborsClassifier(n_neighbors=args.n_neighbors)
        classifier.fit(train_concat_scores, train_concat_labels.argmax(1))
        test_predictions = classifier.predict(test_concat_scores)

        test_accuracy = int((test_concat_labels.argmax(1) == torch.Tensor(test_predictions)).sum()) / \
                        test_concat_labels.shape[0]
    else:
        mlp = MLP(input_size=train_concat_scores.shape[1],
                  hidden_size=args.mlp_hidden_size,
                  out_size=train_concat_labels.shape[1],
                  dropout_rate=args.mlp_dr,
                  norm_out=False)
        mlp = mlp.to(device)
        criterion = CrossEntropyLoss()
        optimizer = RMSprop(mlp.parameters(), lr=args.mlp_lr)
        if not os.path.exists(MLP_STATE_FILE) or args.new_mlp:
            train_simple(mlp, criterion, optimizer, args.mlp_epochs, train_concat_scores, train_concat_labels)
            torch.save(mlp.state_dict(), MLP_STATE_FILE)
        else:
            print('MLP is already trained. Loading...')
            mlp.load_state_dict(torch.load(MLP_STATE_FILE))
        test_accuracy = get_accuracy_simple(mlp, test_concat_scores, test_concat_labels)

    print('Test accuracy: %f' % test_accuracy)
    return test_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=2, help='Only applicable when cuda gpu is available')
    parser.add_argument('--out_size', type=int, default=None, help='Override out_size if needed')
    parser.add_argument('--use_knn', action='store_true', default=False, help='Use knn as classifier, else use MLP')
    parser.add_argument('--n_neighbors', type=int, default=21)
    parser.add_argument('--param_file', type=str, default='parameters/utd_mhad/triplet_loss.yaml')
    parser.add_argument('--new_vectors', action='store_true', default=False,
                        help='Retrieve new vectors, don\'t use the saved ones')
    parser.add_argument('--new_mlp', action='store_true', default=False,
                        help='Don\'t use saved state. Train mlp again.')
    parser.add_argument('--mlp_epochs', type=int, default=100, help='Number of epochs to train the mlp')
    parser.add_argument('--mlp_hidden_size', type=int, default=512)
    parser.add_argument('--mlp_lr', type=float, default=0.001)
    parser.add_argument('--mlp_dr', type=float, default=0.8)
    parser.add_argument('--inertial_state', type=str, default=None)
    parser.add_argument('--sdfdi_state', type=str, default=None)
    parser.add_argument('--skeleton_state', type=str, default=None)
    parser.add_argument('--rule', choices=['concat', 'avg'], default='concat')
    args = parser.parse_args()
    main(args)
