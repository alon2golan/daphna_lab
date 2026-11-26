"""Confirm source compression and backup coverage with frozen CIFAR-100 inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

import active_audit as source

GEOMETRY_SHA256 = '486509610f2947676c2bf5dfc7d3e6950e79805580290f9ca0d54790058e682b'
HUMAN_SHA256 = 'fbe91e5598e2311cdd78a17f0123c76b4000446c044027f5584f096597fe5319'
RBF_SHA256 = '31f8c13336b52bd31cca6c21abf9dc4ecc39d13283243d4dc6581a8206fffb4f'
SYNTHETIC = tuple((f'{kind}{q}', q / 100) for kind in ('sym', 'asym') for q in (30, 40, 50))
WALK_DEPTHS = (3, 4, 6, 8)
POWER_DEPTHS = (3, 6)
SEEDS = tuple(range(25, 30)) + tuple(range(40, 50))


def walk_matrix(neighbors, columns):
    rows, degree = neighbors.shape
    return csr_matrix(
        (np.ones(rows * degree, dtype=np.float64), neighbors.ravel(),
         np.arange(rows + 1, dtype=np.int32) * degree),
        shape=(rows, columns),
    )


def effective_support(weights):
    square_sum = np.einsum('ij,ij->i', weights, weights)
    return np.divide(weights.sum(axis=1) ** 2, square_sum,
                     out=np.zeros(len(weights)), where=square_sum > 0)


def evaluate_order(train, test, labels, canonical, truth, budgets, order, rows,
                   walk, predict, acquisition):
    class_count = source.CLASSES
    anchors = rows[order]
    normalized = source.unit_rows(train[anchors])
    similarities = test @ normalized.T
    observed = {name: values[anchors] for name, values in labels.items()}
    eyes = np.eye(class_count, dtype=np.float64)
    targets = np.concatenate([eyes[y] for y in observed.values()], axis=1)
    shape = (len(truth), len(observed), class_count)
    records = {(b, c): {'acquisition': acquisition, 'budget': b, 'condition': c,
                        'observed_wrong': int((y[:b] != canonical[anchors[:b]]).sum()), 'correct': {}}
               for b in budgets for c, y in observed.items()}

    basis = np.zeros((len(rows), len(order)), dtype=np.float64)
    basis[order, np.arange(len(order))] = 1
    for depth in range(1, max(WALK_DEPTHS) + 1):
        if depth in WALK_DEPTHS:
            weights = predict @ basis
            for budget in budgets:
                window = weights[:, :budget]
                votes = {'raw': (window @ targets[:budget]).reshape(shape)}
                if depth in POWER_DEPTHS:
                    sqrt = np.sqrt(window)
                    ordinary_effective = effective_support(window)
                    sqrt_effective = effective_support(sqrt)
                    if np.any(sqrt_effective + 1e-10 < ordinary_effective):
                        raise ValueError('Square-root weights unexpectedly reduced effective support')
                    votes['sqrt'] = (sqrt @ targets[:budget]).reshape(shape)
                    votes['binary'] = ((window > 0) @ targets[:budget]).reshape(shape)
                for k, (condition, y) in enumerate(observed.items()):
                    result = records[budget, condition]
                    for power, evidence in votes.items():
                        prediction = evidence[:, k].argmax(axis=1)
                        result['correct'][f'{power}{depth}'] = int((prediction == truth).sum())
                    if depth in POWER_DEPTHS:
                        result[f'mean_effective_raw{depth}'] = float(ordinary_effective.mean())
                        result[f'mean_effective_sqrt{depth}'] = float(sqrt_effective.mean())
                    if depth == 3:
                        closest = similarities[:, :budget].argmax(axis=1)
                        result['correct']['1nn'] = int((y[closest] == truth).sum())
            print(f'{acquisition} completed {depth} hops', flush=True)
        if depth < max(WALK_DEPTHS):
            basis = walk @ basis
    return list(records.values())


def run(seed, geometry, human, exact_orders, output):
    if source.sha256(geometry) != GEOMETRY_SHA256 or source.sha256(human) != HUMAN_SHA256:
        raise ValueError('Frozen graph or human labels changed')
    with np.load(geometry, allow_pickle=False) as graph:
        rows, neighbors, heldout = (graph[k] for k in ('rows', 'neighbors', 'heldout_neighbors'))
        metadata = json.loads(graph['metadata'].item())
    if metadata['bank_sha256'] != source.BANK_HASHES or metadata['groups_sha256'] != source.sha256(source.GROUPS):
        raise ValueError('Graph bank provenance changed')
    canonical = np.load(source.BANK / 'train_targets.npy', allow_pickle=False)
    truth = np.load(source.BANK / 'test_targets.npy', allow_pickle=False)
    if len(rows) != 45000 or len(truth) != 10000:
        raise ValueError('Unexpected feature split')
    groups = list(json.loads(source.GROUPS.read_text()).values())
    successor = np.empty(source.CLASSES, dtype=np.int64)
    for group in groups:
        successor[group] = np.roll(group, -1)
    labels = {name: source.corrupt_labels(canonical, name, rate, seed, successor)
              for name, rate in SYNTHETIC}
    order_hash = None
    if seed >= 40:
        budgets = (400, 800, 1200, 1600)
        selected = {'seeded_coverage': source.coverage_order(neighbors, max(budgets), seed)}
    else:
        budgets = (400, 800, 1200, 1600, 2000)
        records = json.loads(exact_orders.read_text())
        if (records['geometry_sha256'] != RBF_SHA256 or records['common_prefix'] != 100
                or records['maxherding'][:100] != records['backup_coverage'][:100]
                or records['backup_weight'] != .5):
            raise ValueError('Exact orders do not match the frozen experiment')
        selected = {name: source.replay_order(neighbors, rows, max(budgets), seed, records[name])
                    for name in ('maxherding', 'backup_coverage')}
        order_hash = source.sha256(exact_orders)
    if seed >= 40 or seed == 25:
        labels['clean'] = canonical
        labels['cifar100n'] = np.load(human, allow_pickle=False)

    test = source.unit_rows(np.load(source.BANK / 'test_features.npy', allow_pickle=False))
    train = np.load(source.BANK / 'train_features.npy', mmap_mode='r', allow_pickle=False)
    walk, predict = walk_matrix(neighbors, len(rows)), walk_matrix(heldout, len(rows))
    results = []
    for acquisition, order in selected.items():
        results.extend(evaluate_order(train, test, labels, canonical, truth, budgets, order, rows,
                                      walk, predict, acquisition))
    artifact = {
        'seed': seed, 'test_count': len(truth), 'geometry_sha256': GEOMETRY_SHA256,
        'human_sha256': HUMAN_SHA256, 'exact_orders_sha256': order_hash,
        'source_sha256': source.sha256(Path(__file__)),
        'acquisition_rows': {kind: rows[order].tolist() for kind, order in selected.items()},
        'rows': results,
    }
    temporary = output.with_suffix('.tmp')
    with temporary.open('x') as stream:
        json.dump(artifact, stream)
    temporary.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seed', type=int, choices=SEEDS)
    parser.add_argument('geometry', type=Path)
    parser.add_argument('human', type=Path)
    parser.add_argument('exact_orders', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.seed, args.geometry, args.human, args.exact_orders, args.output)
