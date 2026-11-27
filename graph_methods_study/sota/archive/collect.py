"""Verify matched archive replay outputs and export exact test counts."""

import argparse
import csv
from itertools import product
from pathlib import Path

from replay import BUDGETS, CONDITIONS, GEOMETRY_SHA256, EVALUATOR_SHA256, HEADS, METHODS, digest, read


def collect(directory):
    source_sha = digest(Path(__file__).with_name('replay.py'))
    shared = None
    data = {}
    parity = []
    for i, condition in enumerate(CONDITIONS):
        run = read(directory / f'condition_{i}.json')
        assert run['condition'] == condition and run['seed'] == 0 and run['test_count'] == 10000
        assert run['source_sha256'] == source_sha and run['geometry_sha256'] == GEOMETRY_SHA256
        assert run['evaluator_sha256'] == EVALUATOR_SHA256
        acquisitions = run['acquisitions']
        assert set(acquisitions) == set(METHODS)
        assert all(len(rows) == len(set(rows)) == max(BUDGETS) for rows in acquisitions.values())
        if shared is None:
            shared = {name: acquisitions[name] for name in ('maxherding', 'backup_coverage')}
        assert all(acquisitions[name] == ids for name, ids in shared.items())
        assert all(rows[:100] == shared['maxherding'][:100] for rows in acquisitions.values())
        for cell in run['rows']:
            method, budget = cell['acquisition'], cell['budget']
            assert cell['condition'] == condition and set(cell['correct']) == set(HEADS)
            assert 0 <= cell['observed_wrong'] <= budget
            if 'archived_1nn_correct' in cell:
                delta = cell['correct']['1nn'] - cell['archived_1nn_correct']
                parity.append(delta)
            for head, correct in cell['correct'].items():
                key = condition, method, budget, head
                assert key not in data and 0 <= correct <= 10000
                data[key] = dict(condition=condition, acquisition=method, budget=budget, head=head,
                                 test_correct=correct, test_count=10000, accuracy_percent=correct / 100,
                                 observed_wrong=cell['observed_wrong'])
    assert set(data) == set(product(CONDITIONS, METHODS, BUDGETS, HEADS))
    assert len(parity) == len(CONDITIONS) * (len(METHODS) - 1) * len(BUDGETS)
    return data, parity


def write(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    data, parity = collect(args.directory)
    write(args.directory / 'metrics.csv', list(data.values()))
    comparisons = []
    for condition, budget, head in product(CONDITIONS, BUDGETS, HEADS):
        values = {method: data[condition, method, budget, head]['accuracy_percent'] for method in METHODS}
        row = dict(condition=condition, budget=budget, head=head, **values)
        row['backup_minus_maxherding_pp'] = round(values['backup_coverage'] - values['maxherding'], 2)
        row['backup_minus_best_reference_pp'] = round(values['backup_coverage'] - max(values[m] for m in METHODS[:-1]), 2)
        comparisons.append(row)
    write(args.directory / 'paired.csv', comparisons)
    print(f'Verified {len(data)} matched held-out test counts. Exact archival 1NN checks: '
          f'{parity.count(0)}/{len(parity)}. Maximum difference in correct predictions: {max(map(abs, parity))}.')


if __name__ == '__main__':
    main()
