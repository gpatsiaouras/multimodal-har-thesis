from datasets import UtdMhadDataset
from transforms import Jittering, Compose, FilterDimensions, Sampler
from visualizers import plot_inertial


# Initiate the dataset with transforms of sampler and filter only gyroscope(x, y, z)
dataset = UtdMhadDataset(modality='inertial', train=True, transform=Compose([
    Sampler(107),
    FilterDimensions([0, 1, 2])
]))

# Retrieve one sample
(sample, _) = dataset[150]
jittering500 = Jittering(500)
jittering1000 = Jittering(1000)
jittering1500 = Jittering(1500)
jittering2000 = Jittering(2000)
jittering2500 = Jittering(2500)

plot_inertial(sample, title='Gyroscope - Jittering Original', y_label='deg/sec', save=True)
plot_inertial(jittering500(sample), title='Gyroscope - Jittering 500', y_label='deg/sec', save=True)
plot_inertial(jittering1000(sample), title='Gyroscope - Jittering 1000', y_label='deg/sec', save=True)
plot_inertial(jittering1500(sample), title='Gyroscope - Jittering 1500', y_label='deg/sec', save=True)
plot_inertial(jittering2000(sample), title='Gyroscope - Jittering 2000', y_label='deg/sec', save=True)
plot_inertial(jittering2500(sample), title='Gyroscope - Jittering 2500', y_label='deg/sec', save=True)
