import scipy.io
import torch
from torch.utils.data import Dataset
import numpy as np

from configurators.utd_mhad import UtdMhadDatasetConfig


class UtdMhadInertialDataset(Dataset):
    """UTD MHAD Inertial Modality Dataset"""

    def __init__(self, transform=None):
        self.filenames = []
        self.labels = []
        self.read_files()
        self.transform = transform

    def read_files(self):
        """
        Creates a list of the filenames for every action, subject and repetition
        :return:
        """
        utd = UtdMhadDatasetConfig()
        for actionKey, actionValue in utd.actions.items():
            for subject in utd.subjects:
                for repetition in utd.repetitions:
                    filename = utd.get_filename(utd.actions[actionKey], utd.modalities['inertial'], subject, repetition)
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

        data = scipy.io.loadmat(self.filenames[idx])['d_iner']
        actions = np.zeros(27)
        actions[self.labels[idx] - 1] = 1

        if self.transform:
            data = self.transform(data)

        return data, actions
