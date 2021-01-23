from scipy import signal
import torch
import numpy as np
import random


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
        if isinstance(self.jitter_factor, list):
            random_idx = random.randint(0, len(self.jitter_factor) - 1)
            jitter_factor = self.jitter_factor[random_idx]
        else:
            jitter_factor = self.jitter_factor

        if jitter_factor == 0:
            return x
        else:
            jittered_x = np.zeros(x.shape)
            # Seed random
            np.random.seed(0)
            for i in range(x.shape[1]):
                data = x[:, i]
                data_unique = np.unique(np.sort(data))
                data_diff = np.diff(data_unique)
                smallest_diff = np.min(data_diff)
                scale_factor = 0.2 * jitter_factor * smallest_diff
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
