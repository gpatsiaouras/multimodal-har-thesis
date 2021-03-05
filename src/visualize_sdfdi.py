import argparse

from datasets import UtdMhadDataset, MmactDataset
from visualizers import frames_player


def main(args):
    if args.use_mmact:
        dataset_sdfdi = MmactDataset(modality='sdfdi')
        dataset_rgb = MmactDataset(modality='video')
    else:
        dataset_sdfdi = UtdMhadDataset(modality='sdfdi')
        dataset_rgb = UtdMhadDataset(modality='rgb')

    # Retrieve one sample, the sample is already converted to an sdfdi image so just display it
    (sample_sdfdi, _) = dataset_sdfdi[args.dataset_idx]
    (sample_rgb, _) = dataset_rgb[args.dataset_idx]

    # Play the video first to see the action being performed. And afterwards show the SDFDI equivalent
    sample_sdfdi.show()
    frames_player(sample_rgb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_mmact', action='store_true', default=False, help='Use mmact dataset instead of UTD-MHAD')
    parser.add_argument('--dataset_idx', type=int, default=0, help='idx to plot from train dataset')
    args = parser.parse_args()
    main(args)
