import argparse
import numpy as np
from datasets import UtdMhadDataset
from transforms import Jittering, Compose, FilterDimensions, Sampler
from visualizers import plot_inertial, plot_inertial_gyroscope_multiple


def main(args):
    # Initiate the dataset with transforms of sampler and filter only gyroscope(x, y, z)
    dataset = UtdMhadDataset(modality='inertial', transform=Compose([
        Sampler(107),
        FilterDimensions([0, 1, 2])
    ]))

    # Retrieve one sample
    (sample, _) = dataset[args.dataset_idx]
    if args.jitter_factor:
        jittering = Jittering(args.jitter_factor)
        plot_data = jittering(sample)
        append_to_title = ' - Jittering %d' % args.jitter_factor
    else:
        plot_data = sample
        append_to_title = ' - Original'

    if args.compare:
        data = np.array([sample[:, 0], plot_data[:, 0]])
        legends = ['Original', 'Jittered %d' % args.jitter_factor]
        plot_inertial_gyroscope_multiple(title='Gyroscope', y_label='deg/sec', legends=legends, data=data,
                                         save=args.save,
                                         show_figure=args.show)
    else:
        plot_inertial(plot_data, title='Gyroscope' + append_to_title, y_label='deg/sec', save=args.save,
                      show_figure=args.show)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_idx', type=int, default=0, help='idx to plot from train dataset')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--jitter_factor', type=int, default=None)
    parser.add_argument('--no_show', dest='show', action='store_false')
    parser.add_argument('--compare', action='store_true')
    parser.set_defaults(save=False, show=True, compare=False)
    args = parser.parse_args()
    main(args)
