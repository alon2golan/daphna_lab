"""Check each saved comparator and export the matched scorecard."""

import argparse
import csv
from itertools import product
import json
from pathlib import Path

from netta_baselines import BATCH, BUDGETS, CONDITIONS, METHODS, TEST_GRAPH_SHA256


def collect(root):
    summary = []
    reference = None
    warm = None
    for condition, method in product(CONDITIONS, METHODS):
        source = root / condition / f'{method}.json'
        result = json.loads(source.read_text())
        entries = result['rows']
        if len(entries) != max(BUDGETS) or len(set(entries)) != max(BUDGETS):
            raise ValueError(f'Invalid trajectory: {source}')
        if (len(result['labels']) != len(entries) or any(not 0 <= y < 100 for y in result['labels'])
                or result['provenance']['head_graph_sha256'] != TEST_GRAPH_SHA256):
            raise ValueError(f'Invalid observations or head: {source}')
        if (result['condition'], result['method']) != (condition, method):
            raise ValueError(f'Unexpected identity: {source}')
        if warm is None:
            warm = entries[:BATCH]
        if entries[:BATCH] != warm:
            raise ValueError(f'Unmatched warm start: {source}')
        if method == 'maxherding_reference':
            if reference is None:
                reference = entries
            if reference != entries:
                raise ValueError(f'MaxHerding geometry changed across conditions: {source}')
        scores = {(v['budget'], v['training_filter']): v for v in result['scores']}
        if len(scores) != len(result['scores']) or set(scores) != set(product(BUDGETS, ('all', 'aum'))):
            raise ValueError(f'Incomplete or duplicate score grid: {source}')
        for (budget, filt), item in sorted(scores.items()):
            labeled = entries[:budget]
            removed = result['masks'][str(budget)]['noisy_rows']
            if item['test_count'] != 10000 or not set(removed).issubset(labeled):
                raise ValueError(f'Wrong partition or test denominator: {source}')
            expected = budget if filt == 'all' else budget - len(removed)
            if item['retained'] != expected:
                raise ValueError(f'Membership and retained count disagree: {source}')
            correct = item['correct']
            if any(v < 0 or v > item['test_count'] for v in correct.values()):
                raise ValueError(f'Impossible test accuracy: {source}')
            summary.append(dict(condition=condition, method=method, budget=budget,
                                training_filter=filt, retained=item['retained'],
                                **{m: (correct[m] / 100 if m in correct else None) for m in ('raw3', 'raw6', '1nn')}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    rows = collect(args.root)
    with args.output.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Verified {len(rows)} scores on {len(CONDITIONS)} matched conditions.')


if __name__ == '__main__':
    main()
