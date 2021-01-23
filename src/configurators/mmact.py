import yaml
import os

MMACT_YAML_CONFIG = 'mmact.yaml'


class MmactDatasetConfig:
    def __init__(self):
        self.dataset_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'mmact')
        # Assign list of actions, modalities, scenes, sessions after reading the config file.
        with open(os.path.join(os.path.dirname(__file__), MMACT_YAML_CONFIG)) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            self.actions = config.get('actions')
            self.modalities = config.get('modalities')
            self.subjects = range(1, config.get('subjects') + 1)
            self.scenes = range(1, config.get('scenes') + 1)
            self.sessions = range(1, config.get('sessions') + 1)
            self.cam = 1  # only use cam1

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

    def get_class_names(self):
        """
        Returns all class names in a list
        :return: list
        """
        return self.actions
