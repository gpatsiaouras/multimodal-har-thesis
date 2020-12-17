import os.path

import cv2
import numpy as np
import scipy.io
import torch
from PIL import Image
from torch.utils.data import Dataset

from configurators.utd_mhad import UtdMhadDatasetConfig
from tools import generate_sdfdi


def _read_video(filename):
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


def _create_jpg_image(jpg_filename, video_filename):
    """
    Receives a filename of an image and a video, reads the video
    generates the SDFDI image and saves it with the given image filename.
    :param jpg_filename: The filename to save the image to
    :param video_filename: The filename to read the video from
    """
    # Create the directory of the SDFDI modality if not exists
    if not os.path.isdir(os.path.dirname(jpg_filename)):
        os.mkdir(os.path.dirname(jpg_filename))
    frames = _read_video(video_filename)
    # Generate SDFDI and save the image as jpg
    sdfdi = generate_sdfdi(frames)
    cv2.imwrite(jpg_filename, sdfdi)


class UtdMhadDataset(Dataset):
    def __init__(self, modality, train=True, transform=None):
        """
        Initializes the utd mhad dataset for a specific modality.
        If train is true, it returns train set, otherwise it returns test set
        Transforms are applied to the data when __get__ is called.
        :param modality: string
        :param train: boolean
        :param transform: Transform
        """
        self.filenames = []
        self.dataset_config = UtdMhadDatasetConfig()
        self.modality = self.dataset_config.modalities[modality]
        self.subjects = [1, 3, 5, 7] if train else [2, 4, 6, 8]
        self.labels = []
        self.read_files()
        self.transform = transform

    def read_files(self):
        """
        Creates a list of the filenames for every action, subject and repetition
        :return:
        """
        for actionKey, actionValue in self.dataset_config.actions.items():
            for subject in self.subjects:
                for repetition in self.dataset_config.repetitions:
                    filename = self.dataset_config.get_filename(
                        action=self.dataset_config.actions[actionKey],
                        modality=self.modality,
                        subject=subject,
                        repetition=repetition)
                    # Only include if the file exists. Some action/subject do not have 4 repetitions.
                    # For the modality of SDFDI in the case that the file doesn't exist
                    # it means that it has to be created by a generation process. Although if the original video file,
                    # doesn't exist then it's the same as above that the repetition probably doesn't exist
                    # and we ignore it
                    if os.path.isfile(filename):
                        self.filenames.append(filename)
                        self.labels.append(actionValue['file_id'])
                    elif self.modality['folder_name'] == 'SDFDI':
                        video_filename = self.dataset_config.get_filename(
                            action=self.dataset_config.actions[actionKey],
                            modality=self.dataset_config.modalities['rgb'],
                            subject=subject,
                            repetition=repetition)
                        if os.path.isfile(video_filename):
                            print('Item %s doesn\'t exist. Creating...' % filename)
                            _create_jpg_image(filename, video_filename)
                            self.filenames.append(filename)
                            self.labels.append(actionValue['file_id'])

    def __len__(self):
        """
        Allows the use of len(dataset), returns the number of samples
        :return: int
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.modality['file_ext'] == 'mat':
            data = self._read_inertial(idx)
        elif self.modality['file_ext'] == 'avi':
            data = _read_video(self.filenames[idx])
        elif self.modality['file_ext'] == 'jpg':
            data = self._read_image(idx)
        else:
            raise Exception('Unsupported extension: %s' % self.modality['file_ext'])

        actions = np.zeros(len(self.dataset_config.actions))
        actions[self.labels[idx] - 1] = 1

        if self.transform:
            data = self.transform(data)

        return data, actions

    def _read_inertial(self, idx):
        return scipy.io.loadmat(self.filenames[idx])[self.modality['data_key']]

    def _read_image(self, idx):
        return Image.open(self.filenames[idx])
