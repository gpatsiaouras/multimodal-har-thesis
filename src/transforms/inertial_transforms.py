from scipy import signal
import torch


class Compose:
    """
    Composes multiple transforms and executes them in the order that they were inputted
    """

    def __init__(self, transforms):
        """
        Initiates a compose object, with a list of transforms
        :param transforms: list
        """
        self.transforms = transforms

    def __call__(self, x):
        """
        When the transform function is called it iterates through all of the transform functions and
        updates the x given.
        :param x: ndarray
        :return: ndarray
        """
        for fn in self.transforms:
            x = fn(x)
        return x


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
    def __init__(self, jitter_factor=500):
        self.jitter_factor = jitter_factor

    # TODO implement the jittering
    def __call__(self, x):
        return x


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
