import random

import numpy as np
from scipy import signal
from scipy.spatial.transform import Rotation as R

random.seed(0)


class Resize:
    def __init__(self, size=125):
        """
        Initiates transform with a target number for resize
        :param size: int
        """
        self.size = size

    def __call__(self, x):
        """
        Output the requested frame size. If frames of sample are smaller it adds padding,
        if frames are bigger it resamples to the requested size, otherwise returns sample.
        :param x: ndarray
        :return: ndarray
        """
        diff = self.size - x.shape[0]
        if diff > 0:
            starting_point = diff // 2
            result = np.zeros((self.size, x.shape[1], x.shape[2],))
            result[:, :, starting_point:starting_point + x.shape[0]] = x
        elif diff < 0:
            result = signal.resample(x, self.size)
        else:
            result = x
        return result


class FilterJoints:
    """
    Returns specific joints (indices) from the skeleton data
    Default joints: head, left elbow, left hand, right elbow,
    right hand, left knee, left foot, right knee and right foot
    """

    def __init__(self, joints=None):
        if joints is None:
            joints = [0, 5, 7, 9, 11, 13, 15, 17, 19]
        self.joints = joints

    def __call__(self, x):
        """
        Returns x filtered according to the selected joints
        :param x: ndarray
        :return: ndarray
        """
        return x[:, self.joints]


class ToSequence:
    """
    Transforms the data by reshaping to (sequence_length, input_size)
    """

    def __init__(self, sequence_length, input_size):
        self.sequence_length = sequence_length
        self.input_size = input_size

    def __call__(self, x):
        """
        Reshapes to Flattens the sample
        :param x: ndarray
        :return: ndarray
        """
        x = x.reshape(self.sequence_length, self.input_size)
        return x


class RandomEulerRotation:
    """
    Data augmentation transform, applies a random rotation of -5, 0 or 5 degrees (by default) in the x,y axis of
    every joint in the skeleton.
    """

    def __init__(self, start=-5, end=5, step=5):
        self.start = start
        self.end = end
        self.step = step

    def __call__(self, x):
        rotate_to = random.randrange(self.start, self.end + 1, self.step)
        rotation = R.from_euler('xy', (rotate_to, rotate_to), degrees=True)
        for frame_idx in range(x.shape[0]):
            x[frame_idx, :, :] = rotation.apply(x[frame_idx, :, :])
        return x
