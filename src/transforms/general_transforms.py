import numpy as np


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


class Normalize:
    """
    Normalizes based on mean and std. Used by skeleton and inertial modalities
    """

    def __init__(self, modality, mean, std):
        self.modality = modality
        self.mean = mean
        self.std = std

    def __call__(self, x):
        if self.modality == 'skeleton':
            # Reshape x from (joints, axis, frames) to (frames, joints, axis)
            x = np.moveaxis(x, [0, 1, 2], [1, 2, 0])
            x = (x - self.mean) / self.std
            return x
        elif self.modality == 'inertial':
            return (x - self.mean) / self.std
