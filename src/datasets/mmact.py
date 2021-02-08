import os

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from datasets.functions import read_video, create_jpg_image, read_csv, read_image, read_combine

MMACT_YAML_CONFIG = 'mmact.yaml'


class MmactDataset(Dataset):
    def __init__(self, modality, transform=None, subjects=None, sessions=None, scenes=None):
        self.dataset_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'mmact')

        # Read dataset configuration yaml file and assign variables
        file = open(os.path.join(os.path.dirname(__file__), MMACT_YAML_CONFIG))
        config = yaml.load(file, Loader=yaml.FullLoader)
        self.actions = config.get('actions')
        self.modalities = config.get('modalities')
        self.modality = self.modalities[modality]
        self.subjects = range(1, config.get('subjects') + 1)
        self.scenes = range(1, config.get('scenes') + 1)
        self.sessions = range(1, config.get('sessions') + 1)
        self.cam = 1  # only use cam1
        file.close()

        # Assign user provided variables if given
        if subjects is not None:
            self.subjects = subjects
        if scenes is not None:
            self.scenes = scenes
        if sessions is not None:
            self.sessions = sessions

        self.transform = transform
        self.filenames = []
        self.labels = []

        self.read_files()

    def get_filename(self, action, modality, subject, scene, session):
        """
        Returns the name of the file for that chunk of data
        :param action: dict
        :param modality: dict
        :param subject: int
        :param scene: int
        :param session: int
        :return: string
        """
        return '{}/{}/subject{}/scene{}/session{}/{}.{}'.format(
            self.dataset_dir,
            modality['folder_name'],
            subject,
            scene,
            session,
            action,
            modality['file_ext']
        )

    def read_files(self):
        """
        Saves filenames and labes in the model while also checking if the file exists and if the modality exists
        :return:
        """
        for action_idx in range(len(self.actions)):
            for subject in self.subjects:
                for scene in self.scenes:
                    for session in self.sessions:
                        filename = self.get_filename(
                            action=self.actions[action_idx],
                            modality=self.modality,
                            subject=subject,
                            scene=scene,
                            session=session,
                        )
                        if os.path.isfile(filename) and os.path.getsize(filename):
                            self.filenames.append(filename)
                            self.labels.append(action_idx)
                        elif self.modality['folder_name'] == 'sdfdi':
                            video_filename = self.get_filename(
                                action=self.actions[action_idx],
                                modality=self.modalities['video'],
                                subject=subject,
                                scene=scene,
                                session=session,
                            )
                            if os.path.isfile(video_filename):
                                print('Item %s doesn\'t exist. Creating...' % filename)
                                create_jpg_image(filename, video_filename)
                                self.filenames.append(filename)
                                self.labels.append(action_idx)
                        elif self.modality['folder_name'] == 'inertial':
                            filenames = []
                            for submodality in self.modality['combine']:
                                filename = self.get_filename(
                                    action=self.actions[action_idx],
                                    modality=self.modalities[submodality],
                                    subject=subject,
                                    scene=scene,
                                    session=session,
                                )
                                if os.path.isfile(filename) and os.path.getsize(filename):
                                    filenames.append(filename)
                            # Only if we have all necessary files consider this a sample.
                            if len(filenames) == len(self.modality['combine']):
                                self.filenames.append(filenames)
                                self.labels.append(action_idx)

    def __len__(self):
        """
        Allows the use of len(dataset), returns the number of samples
        :return: int
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.modality['file_ext'] == 'csv' and isinstance(self.filenames[idx], list):
            data = read_combine(self.filenames[idx])
        elif self.modality['file_ext'] == 'csv':
            data = read_csv(self.filenames[idx])
        elif self.modality['file_ext'] == 'mp4':
            data = read_video(self.filenames[idx])
        elif self.modality['file_ext'] == 'jpg':
            data = read_image(self.filenames[idx])
        else:
            raise Exception('Unsupported extension: %s' % self.modality['file_ext'])

        actions = np.zeros(len(self.actions))
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
        return self.actions
