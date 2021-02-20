import argparse
import itertools
import json
import time
from argparse import Namespace

from datasets import AVAILABLE_MODALITIES
from train_triplet_loss import train_and_test
from visualizers import print_table


def get_permutations(dict_a, dict_b):
    """
    Receives a dict_a containing all configuration the model needs, and a dict_b containing the
    keys that need to be part of the cross validation. Then it generates all the configuration permutations
    fully by merging dict_a with every permutation of dict_b
    :param dict_a:
    :param dict_b:
    :return:
    """
    values = list(dict_b.values())
    keys = list(dict_b.keys())
    permuted_individual_values = list(itertools.product(*values))
    permutations = []
    for individual in permuted_individual_values:
        permutation = dict_a.copy()
        for idx in range(len(keys)):
            permutation[keys[idx]] = individual[idx]
        permutations.append(permutation)

    return permutations


def run():
    args = {
        'margin': None,
        'lr': None,
        'test': False,
        'gpu': parser_args.gpu,
        'param_file': parser_args.param_file,
        'modality': parser_args.modality,
        'num_neighbors': None,
        'epochs': None,
        'semi_hard': parser_args.semi_hard,
        'out_size': None,
        'verbose': False
    }

    args_to_check = {
        'lr': [1e-5, 1e-6, 1e-7, 1e-8],
        'margin': [0.1, 0.2, 0.3, 0.4, 0.5],
        'out_size': [32, 512]
    }
    print_table(args_to_check)

    permutations_dicts = get_permutations(args, args_to_check)

    results = []

    start_time = time.time()
    time_per_training = []
    current_training_counter = 0
    for single_args in permutations_dicts:
        training_start_time = time.time()
        single_args['experiment'] = 'exp_m%s_lr%s_os%s_sm%s' % (
            str(single_args['margin']),
            str(single_args['lr']),
            str(single_args['out_size']),
            str(single_args['semi_hard'])
        )
        single_args = Namespace(**single_args)

        # Start training
        max_train_acc, max_val_acc, test_acc = train_and_test(single_args)

        # Collect results
        results.append({
            'config': vars(single_args),
            'train_acc': max_train_acc,
            'val_acc': max_val_acc,
            'test_acc': test_acc,
        })

        current_training_counter += 1

        # Log times
        total_training_time = time.time() - training_start_time
        time_per_training.append(total_training_time)
        total_time = time.time() - start_time
        avg_time_per_training = sum(time_per_training) / len(time_per_training)
        remaining_time = (len(permutations_dicts) - current_training_counter) * avg_time_per_training
        # Print times
        print('\n=== Training permutation %d/%d ===' % (current_training_counter, len(permutations_dicts)))
        print('Training duration: %s' % time.strftime('%H:%M:%S', time.gmtime(total_training_time)))
        print('Elapsed / Remaining time: %s/%s' % (
            time.strftime('%H:%M:%S', time.gmtime(total_time)), time.strftime('%H:%M:%S', time.gmtime(remaining_time))))

    with open('experiment_results_sm%s.json' % str(parser_args.semi_hard), 'w') as outfile:
        json.dump({
            'results': results,
            'stats': {
                'start_time': time.ctime(start_time),
                'end_time': time.ctime(time.time()),
                'total_duration': time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
            }
        }, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', choices=AVAILABLE_MODALITIES, default='inertial')
    parser.add_argument('--gpu', type=int, default=0, help='Only applicable when cuda gpu is available')
    parser.add_argument('--param_file', type=str, default='parameters/utd_mhad/triplet_loss.yaml')
    parser.add_argument('--semi_hard', dest='semi_hard', action='store_true')
    parser.add_argument('--hard', dest='semi_hard', action='store_false')
    parser.set_defaults(semi_hard=True)
    parser_args = parser.parse_args()

    run()
