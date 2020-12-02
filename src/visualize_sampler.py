from datasets import UtdMhadDataset
from transforms import Jittering, Compose, FilterDimensions, Sampler
from visualizers import plot_inertial


# Initiate the dataset with transforms of sampler and filter only gyroscope(x, y, z)
dataset = UtdMhadDataset(modality='inertial', train=True, transform=Compose([
    FilterDimensions([0, 1, 2])
]))

# Retrieve one sample
(sample, _) = dataset[150]
Sampler100 = Sampler(100)
Sampler150 = Sampler(150)
Sampler200 = Sampler(200)

plot_inertial(sample, title='Gyroscope - Sampler Original', y_label='deg/sec', save=True)
plot_inertial(Sampler100(sample), title='Gyroscope - Sampler 100', y_label='deg/sec', save=True)
plot_inertial(Sampler150(sample), title='Gyroscope - Sampler 150', y_label='deg/sec', save=True)
plot_inertial(Sampler200(sample), title='Gyroscope - Sampler 200', y_label='deg/sec', save=True)
