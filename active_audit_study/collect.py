"""Verify that all audit policies used the planned requests, then emit paired deltas."""

import argparse
import csv
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np

from active_audit import (
    BUDGETS, CONDITIONS, HUMAN_CONDITIONS, SEEDS, SIGNED_BUDGETS, SIGNED_CONDITIONS, SIGNED_SEEDS,
)

POLICIES = ('no_audit', 'random_audit', 'loo_audit', 'single_footprint', 'diverse_footprint', 'fresh_clean')
METRICS = ('test_correct', 'test_1nn_correct', 'pool_correct')


def collect(root, conditions, seeds, budgets, policies):
    runs = {seed: json.loads((root / f'seed_{seed}.json').read_text()) for seed in seeds}
    for key in ('geometry_sha256', 'source_sha256'):
        assert len({run[key] for run in runs.values()}) == 1, key
    if (root / 'geometry.npz').exists():
        with (root / 'geometry.npz').open('rb') as source:
            assert hashlib.file_digest(source, 'sha256').hexdigest() == runs[seeds[0]]['geometry_sha256']
    assert len({tuple(run['acquisition_rows']) for run in runs.values()}) == len(seeds)
    indexed = {}
    for seed, run in runs.items():
        order = run['acquisition_rows']
        assert len(order) == len(set(order)) == 880
        for row in run['rows']:
            budget, condition, policy = row['budget'], row['condition'], row['policy']
            key = (seed, budget, condition, policy)
            assert key not in indexed and row['seed'] == seed
            extra = budget // 10
            requests = row['request_rows']
            assert row['test_count'] == 10000 and row['pool_count'] == 45000 - budget - extra
            assert len(set(requests)) == len(requests) == row['clean_requests']
            if policy == 'fresh_clean':
                assert requests == order[budget:budget + extra]
            elif policy == 'no_audit':
                assert not requests
            else:
                assert len(requests) == extra and set(requests) <= set(order[:budget])
            for correct, total in zip(METRICS, (row['test_count'], row['test_count'], row['pool_count'])):
                assert 0 <= row[correct] <= total
            indexed[key] = row
    assert set(indexed) == set(product(seeds, budgets, [name for name, _ in conditions], policies))
    if any(name == 'clean' for name, _ in conditions):
        for seed, budget in product(seeds, budgets):
            base = indexed[seed, budget, 'clean', 'no_audit']
            for policy in policies:
                if policy == 'fresh_clean':
                    continue
                row = indexed[seed, budget, 'clean', policy]
                assert row['corrected'] == 0 and all(row[m] == base[m] for m in METRICS)
    return indexed


def summary(indexed, conditions, seeds, budgets, policies):
    for budget, (condition, _), policy, (metric, total) in product(
        budgets, conditions, policies, zip(METRICS, ('test_count', 'test_count', 'pool_count')),
    ):
        rows = [indexed[seed, budget, condition, policy] for seed in seeds]
        accuracy = np.array([100 * row[metric] / row[total] for row in rows])
        output = dict(budget=budget, condition=condition, policy=policy, metric=metric,
                      mean_accuracy=float(accuracy.mean()), sd_accuracy=float(accuracy.std(ddof=1)),
                      mean_corrected=float(np.mean([row['corrected'] for row in rows])))
        for control in ('no_audit', 'random_audit', 'loo_audit', 'single_footprint', 'fresh_clean'):
            reference = np.array([100 * indexed[seed, budget, condition, control][metric] / indexed[seed, budget, condition, control][total]
                                  for seed in seeds])
            delta = accuracy - reference
            output[f'{control}_delta_pp'] = float(delta.mean())
            output[f'{control}_paired_sd_pp'] = float(delta.std(ddof=1))
            output[f'{control}_wins'] = int((delta > 0).sum())
        yield output


def write_csv(path, rows):
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--human-labels', action='store_true')
    modes.add_argument('--signed-study', action='store_true')
    args = parser.parse_args()
    conditions = HUMAN_CONDITIONS if args.human_labels else CONDITIONS
    study = ((SIGNED_CONDITIONS, SIGNED_SEEDS, SIGNED_BUDGETS, POLICIES + ('signed_peer',))
             if args.signed_study else (conditions, SEEDS, BUDGETS, POLICIES))
    rows = collect(args.directory, *study)
    write_csv(args.directory / 'metrics.csv', list(rows.values()))
    write_csv(args.directory / 'paired_summary.csv', list(summary(rows, *study)))
    print(f'Verified {len(rows)} matched policy evaluations')


if __name__ == '__main__':
    main()
