import os.path

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

from configurators import UtdMhadDatasetConfig
from .functions import read_image, read_video, create_jpg_image


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
                            create_jpg_image(filename, video_filename)
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
            data = read_video(self.filenames[idx])
        elif self.modality['file_ext'] == 'jpg':
            data = read_image(self.filenames[idx])
        else:
            raise Exception('Unsupported extension: %s' % self.modality['file_ext'])

        actions = np.zeros(len(self.dataset_config.actions))
        actions[self.labels[idx] - 1] = 1

        if self.transform:
            data = self.transform(data)

        return data, actions

    def _read_inertial(self, idx):
        """
        Reads data the data from a mat file and returns an numpy array
        :param idx: index of sample
        :return: numpy array
        """
        return scipy.io.loadmat(self.filenames[idx])[self.modality['data_key']]

    def get_class_names(self):
        """
        Returns a list of action names (classes)
        Used in confusion matrix printing
        :return:
        """
        return [action for action in self.dataset_config.actions]
