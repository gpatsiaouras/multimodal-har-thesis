import matplotlib.pyplot as plt


def plot_inertial_accelerometer(data):
    """
    Receives a chunk of inertial data (one file) and plots the accelerometer data
    :param data: inertial data
    :type data: ndarray
    """
    plt.style.use('seaborn-whitegrid')
    plt.plot(data[:, 0], label='X')
    plt.plot(data[:, 1], label='Y')
    plt.plot(data[:, 2], label='Z')
    plt.title('Accelerometer')
    plt.ylabel("Acceleration (g)")
    plt.xlabel('Timesteps')
    plt.legend()
    plt.show()


def plot_inertial_gyroscope(data):
    """
    Receives a chunk of inertial data (one file) and plots the gyroscope data
    :param data: inertial data
    :type data: ndarray
    """
    plt.style.use('seaborn-whitegrid')
    plt.plot(data[:, 3], label='X')
    plt.plot(data[:, 4], label='Y')
    plt.plot(data[:, 5], label='Z')
    plt.title('Gyroscope')
    plt.ylabel('deg/sec')
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
