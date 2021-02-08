import csv
import importlib
import os

import cv2
import numpy as np
from PIL import Image
from scipy import signal

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
    Reads a csv file into a numpy array, skips first column because it's datetime
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
    """
    Returns instances of transforms dynamically based on the definitions (read from a parameter file)
    :param transform_definitions:
    :return: train_transforms, test_transforms
    """
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


def read_combine(filenames):
    """
    Reads multiple files into one data representation by appending columns.
    1. Reads all the files and saves them as numpy array
    2. Keeps track of the minimum amount of steps
    3. Resizes all data to be the minimum amount of steps to align the data
    4. Stackes all the modalities into one representation
    :param filenames: list of filenames
    :return:
    """
    combined = []
    min_steps = float('inf')
    # Read the data from each of the files and save the minimum number of steps existing
    for filename in filenames:
        data = read_csv(filename)
        if data.shape[0] < min_steps:
            min_steps = data.shape[0]
        combined.append(data)

    # Iterate the different data again and reshape to the smallest one if needed to align
    for data_idx in range(len(combined)):
        if combined[data_idx].shape[0] > min_steps:
            combined[data_idx] = signal.resample(combined[data_idx], min_steps)

    stacked_data = np.hstack(combined)
    return stacked_data
