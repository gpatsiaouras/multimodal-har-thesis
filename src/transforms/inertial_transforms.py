from scipy import signal
import torch
import numpy as np


class CropToSize:
    """
    Crops sample to a speficic given size. Trashes remaining rows in the end of the vector
    bigger than the specified size.
    """

    def __init__(self, size=107):
        """
        Initiates transform with a number for cropping size
        :param size: int
        """
        self.size = size

    def __call__(self, x):
        """
        Crops the input in the first dimension in the given size
        :param x: ndarray
        :return: ndarray
        """
        return x[:self.size]


class Jittering:
    """
    Data augmentation through signal jittering, adding white Gaussian noise
    """
    def __init__(self, jitter_factor=500):
        """
        Initialize with jitter factor default 500
        :param jitter_factor:
        """
        self.jitter_factor = jitter_factor

    def __call__(self, x):
        """
        Jitter signal by adding white Gaussian noise according to the algorithm
        :param x: ndarray
        :return: ndarray
        """
        jittered_x = np.zeros(x.shape)
        # Seed random
        np.random.seed(0)
        for i in range(3):
            data = x[:, i]
            data_unique = np.unique(np.sort(data))
            data_diff = np.diff(data_unique)
            smallest_diff = np.min(data_diff)
            scale_factor = 0.2 * self.jitter_factor * smallest_diff
            jittered_x[:, i] = data + scale_factor * np.random.randn(x.shape[0])
        return jittered_x


class Sampler:
    """
    Resamples a signal from any size of timesteps to the given size
    """
    def __init__(self, size):
        """
        Initiate sampler with the size to resample to
        :param size: int
        """
        self.size = size

    def __call__(self, x):
        """
        Uses scipy signal resample function to downsample/upsample the signal to the given size
        :param x: ndarray
        :return: ndarray
        """
        return signal.resample(x, self.size)


class FilterDimensions:
    """
    Returns specific dimensions from the input data
    """
    def __init__(self, dims):
        self.dims = dims

    def __call__(self, x):
        """
        Returns specific dimensions from the input data
        :param x: ndarray
        :return: ndarray
        """
        return x[:, self.dims]


class Flatten:
    """
    Flattens a multi dimensional signal
    """
    def __call__(self, x):
        return x.flatten()


class ToTensor:
    """
    Convert data to tensor
    """
    def __call__(self, x):
        return torch.tensor(x.values)
