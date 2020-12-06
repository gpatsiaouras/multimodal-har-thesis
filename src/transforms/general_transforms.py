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
