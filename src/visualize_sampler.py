import argparse

from datasets import UtdMhadDataset
from transforms import Compose, FilterDimensions, Sampler
from visualizers import plot_inertial


def main(args):
    # Initiate the dataset with transforms of sampler and filter only gyroscope(x, y, z)
    dataset = UtdMhadDataset(modality='inertial')

    # Retrieve one sample
    (sample, _) = dataset[args.dataset_idx]
    if args.sampler_size:
        append_to_title = ' - Sampler %d' % args.sampler_size
        sampler = Sampler(args.sampler_size)
        plot_data = sampler(sample)
    else:
        append_to_title = ''
        plot_data = sample

    plot_inertial(plot_data[:, :3], title='Gyroscope' + append_to_title, y_label='deg/sec', save=args.save,
                  show_figure=args.show)
    plot_inertial(plot_data[:, 3:], title='Accelerometer' + append_to_title, y_label='m/sec^2', save=args.save,
                  show_figure=args.show)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_idx', type=int, default=0, help='idx to plot from train dataset')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--sampler_size', type=int, default=None)
    parser.add_argument('--no_show', dest='show', action='store_false')
    parser.set_defaults(save=False, show=True)
    args = parser.parse_args()
    main(args)
