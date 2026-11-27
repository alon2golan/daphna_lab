"""Verify the grouped backup grid and calculate matched accuracy differences."""

import argparse
import csv
import itertools
import json
from pathlib import Path
from statistics import mean

from backup_coverage import DEPTHS, GEOMETRY_SHA256
from grouped_backup import ARMS, BATCH, BUDGETS, CONDITIONS, SEEDS


def load_verified(root):
    cases = {}
    identities = []
    noise = {}
    warm_batches = set()
    for seed in SEEDS:
        result = json.loads((root / f'seed_{seed}.json').read_text())
        if result['seed'] != seed or result['geometry_sha256'] != GEOMETRY_SHA256:
            raise ValueError(f'Mismatched seed or geometry for {seed}')
        identities.append((result['revisions'], result['partitions']))
        acquisitions = result['acquisitions']
        if set(acquisitions) != set(ARMS):
            raise ValueError(f'Missing or unexpected trajectories for {seed}')
        warm = tuple(acquisitions['max'][:BATCH])
        warm_batches.add(warm)
        for order in acquisitions.values():
            if len(order) != max(BUDGETS) or len(set(order)) != len(order) or tuple(order[:BATCH]) != warm:
                raise ValueError(f'Mismatched trajectory for {seed}')
        for row in result['records']:
            key = (seed, row['budget'], row['condition'], row['depth'], row['arm'])
            if key in cases or row['seed'] != seed or row['test_count'] != 10000 or row['pool_count'] != 45000 - row['budget']:
                raise ValueError(f'Invalid or duplicate score {key}')
            cases[key] = row
        noise[seed] = result['noise']
        if set(noise[seed]) != set(CONDITIONS):
            raise ValueError(f'Missing corruption manifest for {seed}')
        if noise[seed]['cluster50']['fraction'] != noise[seed]['iid_rate_matched']['fraction']:
            raise ValueError(f'Noise rates do not match for {seed}')
    expected = set(itertools.product(SEEDS, BUDGETS, CONDITIONS, DEPTHS, ARMS))
    if (set(cases) != expected or not all(i == identities[0] for i in identities)
            or len(warm_batches) != len(SEEDS)):
        raise ValueError('Results do not share one protocol or omit a required comparison')
    return cases, noise


def summarize(cases):
    result = []
    for budget, condition, depth in itertools.product(BUDGETS, CONDITIONS, DEPTHS):
        by_arm = {arm: [cases[seed, budget, condition, depth, arm] for seed in SEEDS] for arm in ARMS}
        row = dict(budget=budget, condition=condition, depth=depth)
        for arm, values in by_arm.items():
            row[f'{arm}_accuracy'] = mean(v['correct'] for v in values) / 100
            row[f'{arm}_1nn'] = mean(v['nn_correct'] for v in values) / 100
            row[f'{arm}_error_rate'] = mean(100 * v['wrong_annotations'] / budget for v in values)
            row[f'{arm}_two_groups'] = mean(100 * v['pool_distinct_group_second'] / v['pool_count'] for v in values)
        for arm in ('max', 'backup', 'shuffled'):
            deltas = [(a['correct'] - b['correct']) / 100 for a, b in zip(by_arm['grouped'], by_arm[arm])]
            row[f'grouped_vs_{arm}'] = mean(deltas)
            row[f'grouped_vs_{arm}_seeds'] = deltas
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    cases, noise = load_verified(args.root)
    table = summarize(cases)
    with (args.root / 'summary.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for row in table:
        if row['depth'] == 6:
            print(row['budget'], row['condition'],
                  'max/backup/grouped/shuffled',
                  *(f"{row[f'{arm}_accuracy']:.2f}" for arm in ARMS),
                  f"grouped − backup {row['grouped_vs_backup']:+.2f} pp",
                  row['grouped_vs_backup_seeds'])
    for condition in ('cluster50', 'iid_rate_matched'):
        deltas = [
            mean((cases[seed, b, condition, 6, 'grouped']['correct']
                  - cases[seed, b, condition, 6, 'backup']['correct']) / 100 for b in BUDGETS)
            for seed in SEEDS
        ]
        print(condition, 'averaged across budgets', f'{mean(deltas):+.2f} pp', deltas)
        print(condition, 'error rate and within-selector covariance per seed', [
            (noise[s][condition]['fraction'], noise[s][condition]['within_selector_covariance']) for s in SEEDS
        ])


if __name__ == '__main__':
    main()
