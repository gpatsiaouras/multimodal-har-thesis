import csv
import importlib
import os

import cv2
import numpy as np
from PIL import Image

import transforms
from tools import generate_sdfdi


def read_video(filename):
    """
    Receives a filename of a video, opes the file and saves all concurrent frames in a list as ndarray
    :param filename: name of the video file
    :return: list of frames
    """
    video = cv2.VideoCapture(filename)
    frames = []
    ret = True
    while ret:
        ret, frame = video.read()
        if ret:
            frames.append(frame)

    return frames


def create_jpg_image(jpg_filename, video_filename):
    """
    Receives a filename of an image and a video, reads the video
    generates the SDFDI image and saves it with the given image filename.
    :param jpg_filename: The filename to save the image to
    :param video_filename: The filename to read the video from
    """
    # Create the directory of the SDFDI modality if not exists
    if not os.path.isdir(os.path.dirname(jpg_filename)):
        os.makedirs(os.path.dirname(jpg_filename))
    frames = read_video(video_filename)
    # Generate SDFDI and save the image as jpg
    sdfdi = generate_sdfdi(frames)
    cv2.imwrite(jpg_filename, sdfdi)


def read_image(filename):
    """
    Reads a filename of an image as PIL pytorch format image. Returns the image
    :param filename: Path to image
    :return: PIL Image
    """
    return Image.open(filename)


def read_csv(filename):
    """
    Reads a csv file into a numpy array
    :param filename:
    :return:
    """
    all = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            all.append(row[1:])

    return np.array(all, dtype=float)


def get_transforms_from_config(transform_definitions):
    train_transforms = []
    test_transforms = []
    for trans_config in transform_definitions:
        from_module = importlib.import_module(trans_config['from_module'])
        trans_class = getattr(from_module, trans_config['class_name'])
        transform = trans_class(*trans_config['args'])
        train_transforms.append(transform)
        if trans_config['in_test']:
            test_transforms.append(transform)

    return transforms.Compose(train_transforms), transforms.Compose(test_transforms)