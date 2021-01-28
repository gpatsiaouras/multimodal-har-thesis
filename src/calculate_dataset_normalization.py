import argparse

import numpy as np
from PIL.JpegImagePlugin import JpegImageFile
from torch.utils.data import Dataset

from datasets import AVAILABLE_MODALITIES, AVAILABLE_MODALITIES_DIM, AVAILABLE_DATASETS
from datasets import UtdMhadDataset, MmactDataset


def calculate_means_stds(dataset: Dataset, dim=None):
    """
    Calculates the mean and std for each sample based on the given dimensions and then averages them
    :param dataset: Dataset
    :param dim: Dimensions of sample
    :return: mean, std
    """
    means = []
    stds = []
    for (data, labels) in dataset:
        if isinstance(data, JpegImageFile):
            data = np.array(data)
            means.append(data.mean())
            stds.append(data.std())
        else:
            means.append(data.mean(dim))
            stds.append(data.std(dim))

    mean = np.array(means).mean(0 if dim is not None else None)
    std = np.array(stds).mean(0 if dim is not None else None)

    return mean, std


def get_dataset_class(dataset_name):
    if dataset_name == 'utd_mhad':
        return UtdMhadDataset
    elif dataset_name == 'mmact':
        return MmactDataset
    else:
        raise Exception('Unsupported dataset: %s' % dataset_name)


# Set argument parser
parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--dataset', choices=AVAILABLE_DATASETS, default='utd_mhad')
parser.add_argument('--modality', choices=AVAILABLE_MODALITIES, required=True)
args = parser.parse_args()

SelectedDataset = get_dataset_class(args.dataset)
selectedDim = AVAILABLE_MODALITIES_DIM[AVAILABLE_MODALITIES.index(args.modality)]

train_dataset = SelectedDataset(modality=args.modality)
mean, std = calculate_means_stds(train_dataset, dim=selectedDim)
print('Mean: %s' % mean)
print('Std: %s' % std)
