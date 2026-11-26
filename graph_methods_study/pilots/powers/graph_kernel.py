"""Compare ordinary and non-backtracking votes on the frozen CIFAR-100 BYOL graph."""

import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = (5, 6, 7, 8, 9)
BUDGETS = (100, 400, 800, 1200)
CONDITIONS = (('clean', 0.), ('sym20', .2), ('sym50', .5), ('asym20', .2), ('asym50', .5), ('cifar100n', .402))
KERNELS = ('walk1', 'walk2', 'walk3', 'walk4', 'nonbacktracking3', 'sqrt3', 'binary3')
GEOMETRY_SHA256 = '486509610f2947676c2bf5dfc7d3e6950e79805580290f9ca0d54790058e682b'


def sum_neighbors(counts, neighbors, output):
    for start in range(0, len(neighbors), 256):
        output[start:start + 256] = counts[neighbors[start:start + 256]].sum(axis=1, dtype=np.int16)


def heldout_walks(neighbors, heldout, anchors):
    """Exact endpoint counts through one heldout edge and up to three train edges."""
    counts = np.zeros((len(neighbors), len(anchors)), dtype=np.int16)
    counts[anchors, np.arange(len(anchors))] = 1
    next_step = np.empty_like(counts)
    walks = []
    for depth in range(1, 5):
        votes = np.empty((len(heldout), len(anchors)), dtype=np.int16)
        sum_neighbors(counts, heldout, votes)
        walks.append(votes)
        if depth < 4:
            sum_neighbors(counts, neighbors, next_step)
            counts, next_step = next_step, counts
    return walks


def nonbacktracking_three(neighbors, anchors, walks):
    reciprocal = (neighbors[neighbors] == np.arange(len(neighbors))[:, None, None]).any(axis=2)
    reciprocal_degree = reciprocal.sum(axis=1, dtype=np.int16)
    # The only possible reversal is test -> u -> v -> u; there is no edge to test.
    return walks[2] - walks[0] * reciprocal_degree[anchors]


def compressed_source_votes(walks):
    """Reduce the influence of many paths ending at one annotation."""
    weights = walks.astype(np.float32)
    return np.sqrt(weights), walks > 0


def replay(seed, geometry, output):
    import active_audit as prior

    if prior.sha256(geometry) != GEOMETRY_SHA256:
        raise ValueError('Frozen BYOL geometry checksum mismatch')
    for name in ('train_targets', 'test_targets'):
        if prior.sha256(prior.BANK / f'{name}.npy') != prior.BANK_HASHES[name]:
            raise ValueError(f'Targets checksum mismatch for {name}')
    with np.load(geometry, allow_pickle=False) as data:
        rows, neighbors, heldout = data['rows'], data['neighbors'], data['heldout_neighbors']
    if neighbors.shape != (45000, 13) or heldout.shape != (10000, 13):
        raise ValueError('Unexpected frozen graph shape')
    clean = np.load(prior.BANK / 'train_targets.npy', allow_pickle=False)
    truth = np.load(prior.BANK / 'test_targets.npy', allow_pickle=False)
    human = prior.human_annotations(clean)
    successor = np.empty(prior.CLASSES, dtype=np.int64)
    for group in json.loads(prior.GROUPS.read_text()).values():
        successor[group] = np.roll(group, -1)
    anchors = prior.coverage_order(neighbors, max(BUDGETS), seed)
    if len(np.unique(anchors)) != len(anchors):
        raise ValueError('Repeated acquired row')
    walks = heldout_walks(neighbors, heldout, anchors)
    kernels = walks + [nonbacktracking_three(neighbors, anchors, walks), *compressed_source_votes(walks[2])]
    if any((matrix < 0).any() for matrix in kernels):
        raise ValueError('Invalid negative paths')
    identity = np.eye(prior.CLASSES, dtype=np.float32)
    records = []
    for budget in BUDGETS:
        selected = rows[anchors[:budget]]
        predictions = [matrix[:, :budget].astype(np.float32) for matrix in kernels]
        for condition, rate in CONDITIONS:
            observed = human if condition == 'cifar100n' else prior.corrupt_labels(clean, condition, rate, seed, successor)
            labeled = observed[selected]
            for name, counts in zip(KERNELS, predictions):
                score = counts @ identity[labeled]
                if not np.isfinite(score).all():
                    raise ValueError('Nonfinite prediction')
                records.append(dict(seed=seed, condition=condition, budget=budget, kernel=name,
                                    test_correct=int((score.argmax(axis=1) == truth).sum()), test_count=len(truth),
                                    test_supported=int((score.max(axis=1) > 0).sum()),
                                    realized_corruption=int((labeled != clean[selected]).sum()),
                                    total_vote_mass=float(score.sum(dtype=np.float64))))
        print(f'{seed} {budget} scored', flush=True)
    if output.exists():
        raise FileExistsError(output)
    result = dict(rows=records, acquisition_rows=rows[anchors].tolist(), numpy=np.__version__,
                  geometry_sha256=GEOMETRY_SHA256, source_sha256=prior.sha256(Path(__file__)),
                  helper_sha256=prior.sha256(Path(prior.__file__)), human_sha256=prior.HUMAN_SHA256)
    temporary = output.with_suffix('.tmp')
    with temporary.open('x') as target:
        json.dump(result, target, sort_keys=True)
    temporary.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seed', choices=SEEDS, type=int)
    parser.add_argument('geometry', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    replay(args.seed, args.geometry, args.output)


if __name__ == '__main__':
    main()
