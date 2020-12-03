import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import time


def plot_inertial_accelerometer(data):
    """
    Receives a chunk of inertial data (one file) and plots the accelerometer data
    :param data: inertial data
    :type data: ndarray
    """
    plot_inertial(data, title='Accelerometer', y_label='Acceleration (g)')


def plot_inertial_gyroscope(data, save):
    """
    Receives a chunk of inertial data (one file) and plots the gyroscope data
    :param data: inertial data
    :param save: Save figure to file
    :type data: ndarray
    :type data: boolean
    """
    plot_inertial(data, title='Gyroscope', y_label='deg/sec', save=save)


def plot_inertial_gyroscope_two(title, y_labels, data):
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(title)
    for i in range(len(data)):
        axs[i].plot(data[i])
        axs[i].set_title(y_labels[i])
    fig.show()


def plot_inertial(data, title, y_label, save=False):
    """
    Plots inertial data
    :param data: ndarray
    :param title: Plot title
    :param y_label: Y axis label
    :param save: Save figure to file
    """
    plt.style.use('seaborn-whitegrid')
    plt.plot(data[:, 0], label='X')
    plt.plot(data[:, 1], label='Y')
    plt.plot(data[:, 2], label='Z')
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel('Timesteps')
    plt.legend()
    if save:
        _save_plot('%s.png' % title.strip())
    plt.show()


def plot_accuracy(train_acc, test_acc=None, save=False):
    """
    Plots a list of accuracies over epochs
    :param train_acc: list
    :param test_acc: list
    :param save: Save figure to file
    """
    plt.style.use('seaborn-whitegrid')
    plt.plot(train_acc, label='Train')
    if test_acc is not None:
        plt.plot(test_acc, label='Test')
    plt.title('Accuracy')
    plt.ylabel('% of correct classification')
    plt.xlabel('Epochs')
    plt.legend()
    if save:
        _save_plot('%s.png' % 'accu')
    plt.show()


def plot_loss(data, save=False):
    """
    Plots a list of losses over epochs
    :param data: list
    :param save: Save figure to file
    """
    plt.style.use('seaborn-whitegrid')
    plt.plot(data, label='Loss')
    plt.title('Training loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    if save:
        _save_plot('%s.png' % 'loss')
    plt.show()


def plot_confusion_matrix(cm, classes, title='Confusion matrix', normalize=False, save=False,
                          cmap=plt.get_cmap('Blues')):
    if normalize:
        cm = cm / cm.sum(axis=1)[:, np.newaxis]
        cm = cm * 100  # Make it percentage from 0-100%

    plt.style.use('default')
    plt.figure(figsize=(11, 11))  # Set the size manually for the plot to be drawn correctly
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=270)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, int(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save:
        _save_plot('%s.png' % 'CM')
    plt.show()


def _save_plot(filename):
    """
    Saves a plot to the 'plots' directory in the project root folder
    :param filename: filename with extension
    :return:
    """
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'plots'))
    datetime = time.strftime("%Y%m%d_%H%M", time.localtime())
    filename = datetime + '_' + filename
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    plt.savefig(os.path.join(save_dir, filename))
