import os

import numpy as np
import torch
from torch.utils.data import Dataset

from configurators import MmactDatasetConfig
from datasets.functions import read_video, create_jpg_image, read_csv, read_image


class MmactDataset(Dataset):
    def __init__(self, modality, train, transform=None, sessions=None, scenes=None):
        self.dataset_config = MmactDatasetConfig()
        self.subjects = [1, 3, 5, 7, 9] if train else [2, 4, 6, 8, 10]
        if scenes is None:
            scenes = self.dataset_config.scenes
        if sessions is None:
            sessions = self.dataset_config.sessions
        self.scenes = scenes
        self.sessions = sessions
        self.modality = self.dataset_config.modalities[modality]
        self.transform = transform
        self.filenames = []
        self.labels = []

        self.read_files()

    def read_files(self):
        """
        Saves filenames and labes in the model while also checking if the file exists and if the modality exists
        :return:
        """
        for action_idx in range(len(self.dataset_config.actions)):
            for subject in self.subjects:
                for scene in self.scenes:
                    for session in self.sessions:
                        filename = self.dataset_config.get_filename(
                            action=self.dataset_config.actions[action_idx],
                            modality=self.modality,
                            subject=subject,
                            scene=scene,
                            session=session,
                        )
                        if os.path.isfile(filename) and os.path.getsize(filename):
                            self.filenames.append(filename)
                            self.labels.append(action_idx)
                        elif self.modality['folder_name'] == 'sdfdi':
                            video_filename = self.dataset_config.get_filename(
                                action=self.dataset_config.actions[action_idx],
                                modality=self.dataset_config.modalities['video'],
                                subject=subject,
                                scene=scene,
                                session=session,
                            )
                            if os.path.isfile(video_filename):
                                print('Item %s doesn\'t exist. Creating...' % filename)
                                create_jpg_image(filename, video_filename)
                                self.filenames.append(filename)
                                self.labels.append(self.dataset_config.actions[action_idx])

    def __len__(self):
        """
        Allows the use of len(dataset), returns the number of samples
        :return: int
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.modality['file_ext'] == 'csv':
            data = read_csv(self.filenames[idx])
        elif self.modality['file_ext'] == 'mp4':
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

    def get_class_names(self):
        """
        Returns a list of the action (Classes) names
        Used in confusion matrix axis name printing
        :return: list of class names
        """
        return self.dataset_config.actions
