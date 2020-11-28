from datasets import UtdMhadInertialDataset
from transforms import Jittering, Compose, FilterDimensions, Sampler
from visualizers import plot_inertial_gyroscope_two

# Params
jitter_factor = 500

# Initiate the dataset with transforms of sampler and filter only gyroscope(x, y, z)
dataset = UtdMhadInertialDataset(train=True, transform=Compose([
    Sampler(107),
    FilterDimensions([0, 1, 2])
]))

# Retrieve one sample
(sample, _) = dataset[0]
jittering = Jittering(jitter_factor)

plot_inertial_gyroscope_two(
    title='Gyroscope',
    y_labels=['Original', 'Jitter Factor %s' % jittering.jitter_factor],
    data=[sample, jittering(sample)]
)
