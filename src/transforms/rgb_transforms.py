import cv2 as cv

from tools import generate_sdfdi


class SDFDI:
    def __call__(self, x):
        return generate_sdfdi(x)


class Normalize:
    def __init__(self, min=0, max=1):
        self.min = min
        self.max = max

    def __call__(self, x):
        return cv.normalize(x, None, self.min, self.max, cv.NORM_MINMAX)
