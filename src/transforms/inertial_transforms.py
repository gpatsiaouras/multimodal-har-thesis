from scipy import signal
import torch
import numpy as np
import random


class CropToSize:
    """
    Crops sample to a specific given size. Trashes remaining rows in the end of the vector
    bigger than the specified size.
    """

    def __init__(self, size, randomStart=False):
        """
        Initiates transform with a number for cropping size
        When request size is smaller than the x size, if randomStart
        is true, it will not start from the beginning up to the requested size
        but from a random point
        :param size: Frames to crop to
        :param randomStart: True or false
        """
        self.size = size
        self.randomStart = randomStart

    def __call__(self, x):
        """
        Crops the input in the first dimension in the given size
        :param x: ndarray
        :return: ndarray
        """
        diff = self.size - x.shape[0]
        if diff > 0:
            # if the requested size is larger than the x size. Start in the middle
            starting_point = diff // 2
            # Initiate a zero array
            cropped = np.zeros((self.size, x.shape[1]))
            # Replace the middle of the array with x
            cropped[starting_point:starting_point + x.shape[0]] = x
        elif diff < 0:
            # if the requested size is smaller than the x size
            if self.randomStart:
                # Calculate a random starting point
                starting_point = np.random.randint(0, np.abs(diff))
                # Start from there and retrieve requested samples.
                cropped = x[starting_point:starting_point+self.size]
            else:
                # Start from the beginning and retrieve self.size samples.
                cropped = x[:self.size]
        else:
            cropped = x

        return cropped


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
