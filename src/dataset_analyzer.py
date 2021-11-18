import argparse
from datasets import AVAILABLE_MODALITIES, AVAILABLE_DATASETS
from datasets import UtdMhadDataset, MmactDataset


def get_dataset_class(dataset_name):
    if dataset_name == 'utd_mhad':
        return UtdMhadDataset
    elif dataset_name == 'mmact':
        return MmactDataset
    else:
        raise Exception('Unsupported dataset: %s' % dataset_name)


def get_stats(args):
    selected_dataset = get_dataset_class(args.dataset)
    train_dataset = selected_dataset(modality=args.modality, subjects=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    num_steps = []
    for (data, labels) in train_dataset:
        num_steps.append(data.shape[0])

    mean = sum(num_steps) / len(num_steps)
    print('Min steps: %d' % min(num_steps))
    print('Max steps: %d' % max(num_steps))
    print('Mean steps: %f' % mean)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('--dataset', choices=AVAILABLE_DATASETS, default='utd_mhad')
    parser.add_argument('--modality', choices=AVAILABLE_MODALITIES, required=True)
    args = parser.parse_args()

    get_stats(args)
