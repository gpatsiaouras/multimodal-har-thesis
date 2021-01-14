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
    fig, ax = plt.subplots()
    ax.plot(data[:, 0], label='X')
    ax.plot(data[:, 1], label='Y')
    ax.plot(data[:, 2], label='Z')
    ax.set(title=title, ylabel=y_label, xlabel='Timesteps')
    fig.legend()
    if save:
        _save_plot(fig, '%s.png' % title.strip())
    fig.show()


def plot_accuracy(train_acc, test_acc=None, save=False):
    """
    Plots a list of accuracies over epochs
    :param train_acc: list
    :param test_acc: list
    :param save: Save figure to file
    """
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots()
    ax.plot(train_acc, label='Train')
    if test_acc is not None:
        ax.plot(test_acc, label='Test')
    ax.set(title='Accuracy', ylabel='% of correct classification', xlabel='Epochs')
    ax.legend()
    if save:
        _save_plot(fig, '%s.png' % 'accu')
    fig.show()


def plot_loss(data, save=False):
    """
    Plots a list of losses over epochs
    :param data: list
    :param save: Save figure to file
    """
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots()
    ax.plot(data, label='Loss')
    ax.set(title='Training loss', ylabel='Loss', xlabel='Epochs')
    ax.legend()
    if save:
        _save_plot(fig, '%s.png' % 'loss')
    fig.show()


def plot_confusion_matrix(cm, classes, title='Confusion matrix', normalize=False, save=False,
                          cmap=plt.get_cmap('Blues')):
    if normalize:
        cm = cm / cm.sum(axis=1)[:, np.newaxis]
        cm = cm * 100  # Make it percentage from 0-100%

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(11, 11))  # Set the size manually for the plot to be drawn correctly
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set(title=title, ylabel='True label', xlabel='Predicted label')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, horizontalalignment='right')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, int(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    if save:
        _save_plot(fig, '%s.png' % 'CM')
    fig.show()


def _save_plot(fig, filename):
    """
    Saves a plot to the 'plots' directory in the project root folder
    :param fig: figure
    :param filename: filename with extension
    :return:
    """
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'plots'))
    datetime = time.strftime("%Y%m%d_%H%M", time.localtime())
    filename = datetime + '_' + filename
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    fig.savefig(os.path.join(save_dir, filename))
