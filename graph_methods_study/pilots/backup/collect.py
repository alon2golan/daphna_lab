"""Verify the five paired seeds and summarize the backup-coverage pilot."""

import argparse
import csv
import itertools
import json
from pathlib import Path
from statistics import mean, stdev

from backup_coverage import BUDGETS, GEOMETRY_SHA256, SEEDS

CONDITIONS = ('clean', 'sym20', 'sym50', 'asym20', 'asym50', 'cifar100n')
ARMS = ('max', 'backup')


def checked_results(root):
    table = {}
    hashes = set()
    warm_batches = set()
    for seed in SEEDS:
        artifact = json.loads((root / f'seed_{seed}.json').read_text())
        if artifact['geometry_sha256'] != GEOMETRY_SHA256:
            raise ValueError('Geometry changed across seeds')
        hashes.add((artifact['source_sha256'], artifact['helper_sha256']))
        tracks = artifact['acquisitions']
        if any(len(tracks[arm]) != 2000 or len(set(tracks[arm])) != 2000 for arm in ARMS):
            raise ValueError('Acquisition counts differ')
        warm = tuple(tracks['max'][:100])
        if tuple(tracks['backup'][:100]) != warm:
            raise ValueError('Warm starts differ within a seed')
        warm_batches.add(warm)
        for row in artifact['records']:
            key = seed, row['budget'], row['condition'], row['arm']
            if key in table or row['seed'] != seed or row['test_count'] != 10000:
                raise ValueError(f'Invalid or repeated score {key}')
            table[key] = row
    if len(hashes) != 1 or len(warm_batches) != len(SEEDS):
        raise ValueError('Source identity differs or warm batches are duplicated')
    if set(table) != set(itertools.product(SEEDS, BUDGETS, CONDITIONS, ARMS)):
        raise ValueError('Missing or unexpected score grid')
    return table


def summary(table):
    results = []
    for condition, budget in itertools.product(CONDITIONS, BUDGETS):
        baseline = [table[seed, budget, condition, 'max'] for seed in SEEDS]
        backup = [table[seed, budget, condition, 'backup'] for seed in SEEDS]
        graph_deltas = [(b['test_correct'] - a['test_correct']) / 100 for a, b in zip(baseline, backup)]
        nn_deltas = [(b['nn_correct'] - a['nn_correct']) / 100 for a, b in zip(baseline, backup)]
        results.append(dict(condition=condition, budget=budget, seeds=len(SEEDS),
                            graph_baseline=mean(x['test_correct'] for x in baseline) / 100,
                            graph_backup=mean(x['test_correct'] for x in backup) / 100,
                            graph_delta_pp=mean(graph_deltas), graph_delta_sd=stdev(graph_deltas),
                            graph_wins=sum(d > 0 for d in graph_deltas), graph_seed_deltas=graph_deltas,
                            one_nn_delta_pp=mean(nn_deltas), one_nn_delta_sd=stdev(nn_deltas),
                            first_reach_delta_pp=mean(100 * (b['pool_closest_count'] - a['pool_closest_count']) / a['pool_count']
                                                      for a, b in zip(baseline, backup)),
                            second_reach_delta_pp=mean(100 * (b['pool_backup_count'] - a['pool_backup_count']) / a['pool_count']
                                                       for a, b in zip(baseline, backup)),
                            test_support_delta_pp=mean((b['test_supported'] - a['test_supported']) / 100
                                                       for a, b in zip(baseline, backup)),
                            wrong_labels_delta=mean(b['wrong_annotations'] - a['wrong_annotations']
                                                    for a, b in zip(baseline, backup))))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    table = checked_results(args.root)
    scores = summary(table)
    with (args.root / 'summary.csv').open('x', newline='') as stream:
        out = csv.DictWriter(stream, fieldnames=list(scores[0]))
        out.writeheader()
        out.writerows(scores)
    for row in scores:
        print(row['condition'], row['budget'],
              f"{row['graph_baseline']:.2f} -> {row['graph_backup']:.2f} ({row['graph_delta_pp']:+.2f} pp)",
              f"1NN {row['one_nn_delta_pp']:+.2f} pp", row['graph_seed_deltas'])


if __name__ == '__main__':
    main()
