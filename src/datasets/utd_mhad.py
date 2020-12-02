import cv2
import scipy.io
import torch
from torch.utils.data import Dataset
import numpy as np
import os.path

from configurators.utd_mhad import UtdMhadDatasetConfig


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
                    # Only include if the file exists. Some action/subject do not have 4 repetitions
                    if os.path.isfile(filename):
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
            data = scipy.io.loadmat(self.filenames[idx])[self.modality['data_key']]
        elif self.modality['file_ext'] == 'avi':
            data = cv2.VideoCapture(self.filenames[idx])
        else:
            raise Exception('Unsupported extention: %s' % self.modality['file_ext'])

        actions = np.zeros(27)
        actions[self.labels[idx] - 1] = 1

        if self.transform:
            data = self.transform(data)

        return data, actions
