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


class CropToSize:
    """
    Crops sample to a specific given size. Trashes remaining rows in the end of the vector
    bigger than the specified size.
    """

    def __init__(self, size, random_start=False):
        """
        Initiates transform with a number for cropping size
        When request size is smaller than the x size, if randomStart
        is true, it will not start from the beginning up to the requested size
        but from a random point
        :param size: Frames to crop to
        :param random_start: True or false
        """
        self.size = size
        self.random_start = random_start

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
            if len(x.shape) == 3:
                cropped = np.zeros((self.size, x.shape[1], x.shape[2]))
            else:
                cropped = np.zeros((self.size, x.shape[1]))
            # Replace the middle of the array with x`
            cropped[starting_point:starting_point + x.shape[0]] = x
        elif diff < 0:
            # if the requested size is smaller than the x size
            if self.random_start:
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