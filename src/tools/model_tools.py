import os
import time

import torch


def save_model(model, filename):
    """
    Saves the model state_dict in a folder saved_model with a filename containing datatime info and the model name
    :param model: Model object
    :param filename: name of the file to be saved
    """
    # Default directory to save models is project_root/saved_models. If it doesn't exist create it
    save_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'saved_models')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # construct a name for the saved_model
    datetime = time.strftime("%Y%m%d_%H%M", time.localtime())
    filename = '_'.join([datetime, model.name, filename])

    # Save the model in the saved_models/ directory
    torch.save(model.state_dict(), os.path.join(save_dir, filename))
