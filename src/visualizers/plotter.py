import itertools
import os
import time

import matplotlib.pyplot as plt
import numpy as np


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


def plot_inertial_gyroscope_multiple(title, y_label, legends, data, save=False, show_figure=True):
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots()
    for i in range(len(data)):
        ax.plot(data[i], label=legends[i])
    ax.set(title=title, ylabel=y_label, xlabel='Timesteps')
    fig.legend()
    if save:
        _save_plot(fig, '%s.png' % title.strip())
    if show_figure:
        plt.show()


def plot_inertial(data, title, y_label, save=False, show_figure=True):
    """
    Plots inertial data
    :param data: ndarray
    :param title: Plot title
    :param y_label: Y axis label
    :param save: Save figure to file
    :param show_figure: Show figure
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
    if show_figure:
        plt.show()


def plot_accuracy(train_acc, validation_acc=None, save=False, show_figure=True):
    """
    Plots a list of accuracies over epochs
    :param train_acc: list
    :param validation_acc: list
    :param save: Save figure to file
    :param show_figure: fig.show() when true
    """
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots()
    ax.plot(train_acc, label='Train')
    if validation_acc is not None:
        ax.plot(validation_acc, label='Test')
    ax.set(title='Accuracy', ylabel='% of correct classification', xlabel='Epochs')
    ax.legend()
    if save:
        _save_plot(fig, '%s.png' % 'accu')
    if show_figure:
        fig.show()


def plot_loss(data, save=False, show_figure=True):
    """
    Plots a list of losses over epochs
    :param data: list
    :param save: Save figure to file
    :param show_figure: fig.show() when true
    """
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots()
    ax.plot(data, label='Loss')
    ax.set(title='Training loss', ylabel='Loss', xlabel='Epochs')
    ax.legend()
    if save:
        _save_plot(fig, '%s.png' % 'loss')
    if show_figure:
        fig.show()


def plot_confusion_matrix(cm, classes, title='Confusion matrix', normalize=False, save=False, show_figure=True):
    """
    Plots the confusion matrix using matplotlib, saves if selected and returns the rendered image to use with
    tensorboard
    :param cm: confusion matrix data
    :param classes: List of classes with names
    :param title: Title of the figure
    :param normalize: Normalizing the values pulls the numbers between 0 - 100%
    :param save: If it should save the figure as an image in the local filesystem
    :param show_figure: True to show the figure locally
    :return: image of the figure (for tensorboard)
    """
    if normalize:
        cm = cm / cm.sum(axis=1)[:, np.newaxis]
        cm = cm * 100  # Make it percentage from 0-100%

    plt.style.use('default')
    cmap = plt.get_cmap('Blues')
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

    if show_figure:
        fig.show()

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape((1100, 1100, 3)).transpose((2, 0, 1))
    plt.close('all')
    return image


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
    final_filename = os.path.join(save_dir, filename)
    fig.savefig(final_filename)
    print('Saved figure in ' + final_filename)
