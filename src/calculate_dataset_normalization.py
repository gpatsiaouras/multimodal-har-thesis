import numpy as np
from PIL.JpegImagePlugin import JpegImageFile

from datasets import UtdMhadDataset


def calculate_means_stds(dataset: UtdMhadDataset, dim=None):
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


# Inertial calculation
train_dataset = UtdMhadDataset(modality='inertial', train=True)
inertial_mean, inertial_std = calculate_means_stds(train_dataset, dim=0)
print('Inertial mean std per axis (gyro x,y,z accel x,y,z)')
print(inertial_mean)
print(inertial_std)

# SDFDI calculation
train_dataset = UtdMhadDataset(modality='sdfdi', train=True)
sdfdi_mean, sdfdi_std = calculate_means_stds(train_dataset)
print('SDFDI (RGB) mean std per channel')
print(sdfdi_mean)
print(sdfdi_std)

# Skeleton calculation
train_dataset = UtdMhadDataset(modality='skeleton', train=True)
skeleton_mean, skeleton_std = calculate_means_stds(train_dataset, dim=(0, 2))
print('Skeleton mean std per axis of every joint')
print(skeleton_mean)
print(skeleton_std)
