import yaml
import os

UTD_MHAD_YAML_CONFIG = 'utd_mhad.yaml'


class UtdMhadDatasetConfig:
    def __init__(self):
        self.dataset_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'utd_mhad')
        # Assign list of actions, modalities and repetitions after reading the config file.
        with open(os.path.join(os.path.dirname(__file__), UTD_MHAD_YAML_CONFIG)) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            self.actions = config.get('actions')
            self.modalities = config.get('modalities')
            self.subjects = range(1, config.get('subjects'))
            self.repetitions = range(1, config.get('repetitions'))

    def get_filename(self, action, modality, subject, repetition):
        """
        Returns the name of the file for that chunk of data
        :param action: dict
        :param modality: dict
        :param subject: int
        :param repetition: int
        :return: string
        """
        return '{}/{}/a{}_s{}_t{}_{}.{}'.format(
            self.dataset_dir,
            modality['folder_name'],
            action['file_id'],
            subject,
            repetition,
            modality['file_alias'],
            modality['file_ext']
        )
