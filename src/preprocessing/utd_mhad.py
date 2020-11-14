import scipy.io
import yaml
import os
import cv2

from src.visualizers.cv2player import avi_player
from src.visualizers.plotter import plot_inertial_accelerometer, plot_inertial_gyroscope

UTD_MHAD_YAML_CONFIG = 'utd_mhad.yaml'


class UTDDatasetAdapter:
    """
    Dataset adapter for UTD-MHAD
    """
    def __init__(self):
        self.dataset_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'utd_mhad')
        self.actions = {}
        self.modalities = {}
        self.numbers = {}
        self.get_dataset_info()

    def get_dataset_info(self):
        """
        Assigns lists of actions, modalities and numbers to the object, after reading the config file
        for this dataset.
        """
        with open(os.path.join(os.path.dirname(__file__), UTD_MHAD_YAML_CONFIG)) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            self.actions = config.get('actions')
            self.modalities = config.get('modalities')
            self.numbers = config.get('numbers')

    # TODO refactor this one to have consistent return type
    def get_chunk(self, action, modality, subject, retry):
        """
        Generates a filename, reads the appropriate dataset file and returns the parsed data.
        :param action: dict
        :param modality: dict
        :param subject: dict
        :param retry: dict
        :return: ndarray|cv2
        """
        filename = '{}/{}/a{}_s{}_t{}_{}.{}'.format(self.dataset_dir, modality['folder_name'], action['file_id'],
                                                    subject, retry,
                                                    modality['file_alias'], modality['file_ext'])
        if modality['file_ext'] == 'mat':
            return scipy.io.loadmat(filename)[modality['data_key']]
        elif modality['file_ext'] == 'avi':
            return cv2.VideoCapture(filename)


if __name__ == "__main__":
    # Load an example
    utd = UTDDatasetAdapter()
    for mod in [utd.modalities['rgb'], utd.modalities['inertial']]:
        data = utd.get_chunk(utd.actions['swipe_left'], mod, 1, 1)
        if isinstance(data, cv2.VideoCapture):
            avi_player(data)
        else:
            plot_inertial_accelerometer(data)
            plot_inertial_gyroscope(data)
            print(data)
