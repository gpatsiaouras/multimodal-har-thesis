import os.path

import numpy as np
import scipy.io
import torch
import yaml
from torch.utils.data import Dataset
from operator import itemgetter
from .functions import read_image, read_video, create_jpg_image

UTD_MHAD_YAML_CONFIG = 'utd_mhad.yaml'


class UtdMhadDataset(Dataset):
    def __init__(self, modality, actions=None, subjects=None, repetitions=None, transform=None):
        """
        Initializes the utd mhad dataset for a specific modality.
        Can specify actions subjects and repetitions.
        Transforms are applied to the data when __get__ is called.
        :param modality: string
        :param actions: list
        :param subjects: list
        :param repetitions: list
        :param transform: Transform
        """
        self.dataset_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'utd_mhad')

        # Assign variables from config
        file = open(os.path.join(os.path.dirname(__file__), UTD_MHAD_YAML_CONFIG))
        config = yaml.load(file, Loader=yaml.FullLoader)
        self.actions = config.get('actions')
        self.modalities = config.get('modalities')
        self.modality = self.modalities[modality]
        self.joint_names = config.get('joint_names')
        self.bones = config.get('bones')
        self.subjects = range(1, config.get('subjects') + 1)
        self.repetitions = range(1, config.get('repetitions') + 1)
        file.close()

        self.filenames = []
        if actions is not None:
            self.actions = itemgetter(*actions)(self.actions)
        if subjects is not None:
            self.subjects = subjects
        if repetitions is not None:
            self.repetitions = repetitions
        self.labels = []
        self.read_files()
        self.transform = transform

    def get_filename(self, action, modality, subject, repetition):
        """
        Returns the name of the file for that chunk of data
        :param action: int
        :param modality: dict
        :param subject: int
        :param repetition: int
        :return: string
        """
        return '{}/{}/a{}_s{}_t{}_{}.{}'.format(
            self.dataset_dir,
            modality['folder_name'],
            action,
            subject,
            repetition,
            modality['file_alias'],
            modality['file_ext']
        )

    def read_files(self):
        """
        Creates a list of the filenames for every action, subject and repetition
        :return:
        """
        for actionObj in self.actions:
            for subject in self.subjects:
                for repetition in self.repetitions:
                    filename = self.get_filename(
                        action=actionObj['file_id'],
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
                        self.labels.append(actionObj['file_id'])
                    elif self.modality['folder_name'] == 'SDFDI':
                        video_filename = self.get_filename(
                            action=actionObj['file_id'],
                            modality=self.modalities['rgb'],
                            subject=subject,
                            repetition=repetition)
                        if os.path.isfile(video_filename):
                            print('Item %s doesn\'t exist. Creating...' % filename)
                            create_jpg_image(filename, video_filename)
                            self.filenames.append(filename)
                            self.labels.append(actionObj['file_id'])

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

        actions = np.zeros(len(self.actions))
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
        data = scipy.io.loadmat(self.filenames[idx])[self.modality['data_key']]
        if self.modality['file_alias'] == 'skeleton':
            data = np.moveaxis(data, [0, 1, 2], [1, 2, 0])
        return data

    def get_class_names(self):
        """
        Returns a list of action names (classes)
        Used in confusion matrix printing
        :return: list
        """
        return [action['name'] for action in self.actions]
