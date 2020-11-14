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
    plt.xlabel('Time(s)')
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
    plt.xlabel('Time(s)')
    plt.legend()
    plt.show()
