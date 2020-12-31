import numpy as np
from scipy import signal


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
        diff = self.size - x.shape[2]
        if diff > 0:
            starting_point = diff // 2
            result = np.zeros((x.shape[0], x.shape[1], self.size))
            result[:, :, starting_point:starting_point+x.shape[2]] = x
        elif diff < 0:
            result = signal.resample(x, self.size, axis=2)
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
        return x[self.joints]


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
        x = x.flatten()
        x = x.reshape(self.sequence_length, self.input_size)
        return x


class Normalize:
    """
    Normalizes a sample to 0, 1 values.
    """

    def __init__(self, axis):
        self.axis = axis

    def __call__(self, x):
        x_min = x.min(axis=self.axis, keepdims=True)
        x_max = x.max(axis=self.axis, keepdims=True)

        return (x - x_min) / (x_max - x_min)


class SwapJoints:
    """
    Converts from (joints) x (axis) x (frames) to (axis) × (joints) × (frames).
    When used, to normalize use "Normalize((1, 2))" else "Normalize((0, 2))"
    """

    def __call__(self, x):
        return x.swapaxes(0, 1)
