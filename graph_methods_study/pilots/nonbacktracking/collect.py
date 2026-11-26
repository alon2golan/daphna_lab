"""Verify the frozen predictor grid and summarize paired test deltas."""

import argparse
import csv
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np

from graph_kernel import BUDGETS, CONDITIONS, GEOMETRY_SHA256, KERNELS, SEEDS


def collect(root, audit_reference):
    expected = hashlib.sha256(Path(__file__).with_name('graph_kernel.py').read_bytes()).hexdigest()
    indexed = {}
    for seed in SEEDS:
        run = json.loads((root / f'seed_{seed}.json').read_text())
        prior = json.loads((audit_reference / f'seed_{seed}.json').read_text())
        assert run['geometry_sha256'] == prior['geometry_sha256'] == GEOMETRY_SHA256
        assert run['source_sha256'] == expected
        ordered = run['acquisition_rows']
        assert len(ordered) == len(set(ordered)) == max(BUDGETS)
        assert ordered[:len(prior['acquisition_rows'])] == prior['acquisition_rows']
        for row in run['rows']:
            key = (seed, row['budget'], row['condition'], row['kernel'])
            assert key not in indexed and row['seed'] == seed and row['test_count'] == 10000
            assert 0 <= row['test_correct'] <= row['test_count']
            indexed[key] = row
        for ref in prior['rows']:
            if ref['policy'] == 'no_audit':
                key = (seed, ref['budget'], ref['condition'], 'walk3')
                assert indexed[key]['test_correct'] == ref['test_correct']
                assert indexed[key]['realized_corruption'] == ref['initial_wrong']
    assert set(indexed) == set(product(SEEDS, BUDGETS, [name for name, _ in CONDITIONS], KERNELS))
    return indexed


def summary(indexed):
    for budget, (condition, _), kernel in product(BUDGETS, CONDITIONS, KERNELS):
        rows = [indexed[seed, budget, condition, kernel] for seed in SEEDS]
        refs = [indexed[seed, budget, condition, 'walk3'] for seed in SEEDS]
        accuracy = np.array([100 * row['test_correct'] / row['test_count'] for row in rows])
        base = np.array([100 * row['test_correct'] / row['test_count'] for row in refs])
        delta = accuracy - base
        yield dict(budget=budget, condition=condition, kernel=kernel, mean_accuracy=float(accuracy.mean()),
                   delta_vs_walk3_pp=float(delta.mean()), paired_sd_pp=float(delta.std(ddof=1)),
                   paired_wins=int((delta > 0).sum()),
                   mean_supported=float(np.mean([100 * row['test_supported'] / row['test_count'] for row in rows])),
                   mean_vote_mass=float(np.mean([row['total_vote_mass'] for row in rows])))


def csv_write(path, rows):
    with path.open('x', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('audit_reference', type=Path)
    args = parser.parse_args()
    rows = collect(args.results, args.audit_reference)
    csv_write(args.results / 'metrics.csv', list(rows.values()))
    csv_write(args.results / 'summary.csv', list(summary(rows)))
    print(f'Verified {len(rows)} held-out evaluations and exact original depth-three baselines')


if __name__ == '__main__':
    main()
