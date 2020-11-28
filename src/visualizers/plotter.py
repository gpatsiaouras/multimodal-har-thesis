import matplotlib.pyplot as plt


def plot_inertial_accelerometer(data):
    """
    Receives a chunk of inertial data (one file) and plots the accelerometer data
    :param data: inertial data
    :type data: ndarray
    """
    _plot_inertial(data, title='Accelerometer', y_label='Acceleration (g)')


def plot_inertial_gyroscope(data):
    """
    Receives a chunk of inertial data (one file) and plots the gyroscope data
    :param data: inertial data
    :type data: ndarray
    """
    _plot_inertial(data, title='Gyroscope', y_label='deg/sec')


def plot_inertial_gyroscope_two(title, y_labels, data):
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(title)
    for i in range(len(data)):
        axs[i].plot(data[i])
        axs[i].set_title(y_labels[i])
    fig.show()


def _plot_inertial(data, title, y_label):
    """
    Plots inertial data
    :param data: ndarray
    :param title: Plot title
    :param y_label: Y axis label
    """
    plt.style.use('seaborn-whitegrid')
    plt.plot(data[:, 0], label='X')
    plt.plot(data[:, 1], label='Y')
    plt.plot(data[:, 2], label='Z')
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel('Timesteps')
    plt.legend()
    plt.show()


def plot_accuracy(train_acc, test_acc=None):
    """
    Plots a list of accuracies over epochs
    :param train_acc: list
    :param test_acc: list
    """
    plt.style.use('seaborn-whitegrid')
    plt.plot(train_acc, label='Train')
    if test_acc is not None:
        plt.plot(test_acc, label='Test')
    plt.title('Accuracy')
    plt.ylabel('% of correct classification')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


def plot_loss(data):
    """
    Plots a list of losses over epochs
    :param data: list
    """
    plt.style.use('seaborn-whitegrid')
    plt.plot(data, label='Loss')
    plt.title('Training loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
