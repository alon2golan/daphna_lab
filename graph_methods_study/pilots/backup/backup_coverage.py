"""Compare one- and two-anchor sparse coverage on a frozen CIFAR-100 bank."""

import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = (17, 29, 41, 53, 67)
BATCH = 100
BUDGETS = (400, 800, 1200, 1600, 2000)
GEOMETRY_SHA256 = '486509610f2947676c2bf5dfc7d3e6950e79805580290f9ca0d54790058e682b'


def sparse_affinities(neighbors, features):
    """Exact RBF affinity on self, 13NN and neighbors of the five closest."""
    n = len(neighbors)
    candidates = np.column_stack((np.arange(n), neighbors, neighbors[neighbors[:, :5]].reshape(n, -1)))
    keys = np.unique(np.arange(n, dtype=np.int64)[:, None] * n + candidates)
    target, candidate = (keys // n).astype(np.int32), (keys % n).astype(np.int32)
    similarity = np.empty(len(keys), dtype=np.float32)
    for start in range(0, len(keys), 8192):
        block = slice(start, start + 8192)
        left, right = features[target[block]], features[candidate[block]]
        similarity[block] = np.exp(-np.sum((left - right)**2, axis=1))
    return target, candidate, similarity


def select(target, candidate, similarity, n, count, batch_size, seed, backup, warm=None):
    """Greedy top-one plus backup-weighted top-two facility coverage."""
    target_offset = np.r_[0, np.cumsum(np.bincount(target, minlength=n))]
    target_degree = np.diff(target_offset)
    by_candidate = np.argsort(candidate, kind='stable')
    candidate_offset = np.r_[0, np.cumsum(np.bincount(candidate, minlength=n))]
    tie = np.random.default_rng(seed).random(n) * 1e-10
    first = np.zeros(n, dtype=np.float64)
    second = np.zeros_like(first)
    selected = np.zeros(n, dtype=bool)
    order = []
    for begin in range(0, count, batch_size):
        # The first batch is the control trajectory in both arms.
        lam = backup if begin else 0.
        demand = ~selected
        contribution = ((1 - lam) * np.maximum(similarity - first[target], 0.)
                        + lam * np.maximum(similarity - second[target], 0.))
        gain = np.bincount(candidate, weights=contribution * demand[target], minlength=n)
        gain[selected] = -np.inf
        for offset in range(min(batch_size, count - begin)):
            picked = int(warm[offset]) if begin == 0 and warm is not None else int(np.argmax(gain + tie))
            order.append(picked)
            selected[picked] = True
            gain[picked] = -np.inf
            edges = by_candidate[candidate_offset[picked]:candidate_offset[picked + 1]]
            affected = target[edges]
            new = similarity[edges]
            previous_first, previous_second = first[affected].copy(), second[affected].copy()
            second[affected] = np.maximum(previous_second, np.minimum(previous_first, new))
            first[affected] = np.maximum(previous_first, new)
            lengths = target_degree[affected]
            edge_owner = np.repeat(np.arange(len(affected)), lengths)
            start = np.repeat(target_offset[affected] - (np.cumsum(lengths) - lengths), lengths)
            adjacent = np.arange(lengths.sum()) + start
            similarity_of_adjacent = similarity[adjacent]
            delta = ((1 - lam) * (np.maximum(similarity_of_adjacent - first[target[adjacent]], 0.)
                                  - np.maximum(similarity_of_adjacent - previous_first[edge_owner], 0.))
                     + lam * (np.maximum(similarity_of_adjacent - second[target[adjacent]], 0.)
                              - np.maximum(similarity_of_adjacent - previous_second[edge_owner], 0.)))
            np.add.at(gain, candidate[adjacent], delta * demand[target[adjacent]])
    return np.asarray(order)


def check_selector():
    n = 14
    rng = np.random.default_rng(171)
    matrix = rng.random((n, n)) * (rng.random((n, n)) > .4)
    np.fill_diagonal(matrix, 1.)
    target, candidate = np.nonzero(matrix)
    for backup in (0., .5):
        chosen = select(target, candidate, matrix[target, candidate], n, 10, 3, 17, backup)
        order = []
        tie = np.random.default_rng(17).random(n) * 1e-10
        for begin in (0, 3, 6, 9):
            demand = np.setdiff1d(np.arange(n), order)
            lam = backup if begin else 0.
            for _ in range(min(3, 10 - begin)):
                open_ = np.setdiff1d(np.arange(n), order)
                ranks = np.sort(np.column_stack((np.zeros((n, 2)), matrix[:, order])), axis=1)
                gain = ((1 - lam) * np.maximum(matrix[demand] - ranks[demand, -1, None], 0.)
                        + lam * np.maximum(matrix[demand] - ranks[demand, -2, None], 0.)).sum(axis=0)
                order.append(int(open_[np.argmax((gain + tie)[open_])]))
        np.testing.assert_array_equal(chosen, order)


def run(seed, geometry, output):
    import active_audit as reference

    if output.exists():
        raise FileExistsError(output)
    if reference.sha256(geometry) != GEOMETRY_SHA256:
        raise ValueError('Unexpected frozen geometry checksum')
    for name, expected in reference.BANK_HASHES.items():
        if reference.sha256(reference.BANK / f'{name}.npy') != expected:
            raise ValueError(f'BYOL checksum differs for {name}')
    with np.load(geometry, allow_pickle=False) as artifact:
        rows, neighbors, heldout = artifact['rows'], artifact['neighbors'], artifact['heldout_neighbors']
    if neighbors.shape != (45000, 13) or heldout.shape != (10000, 13):
        raise ValueError('Unexpected train or test geometry')
    features = reference.unit_rows(np.load(reference.BANK / 'train_features.npy', mmap_mode='r')[rows])
    test_features = reference.unit_rows(np.load(reference.BANK / 'test_features.npy', allow_pickle=False))
    target, candidate, similarity = sparse_affinities(neighbors, features)
    warm = reference.coverage_order(neighbors, BATCH, seed)
    arms = {name: select(target, candidate, similarity, len(rows), max(BUDGETS), BATCH, seed, setting, warm)
            for name, setting in (('max', 0.), ('backup', .5))}
    for order in arms.values():
        np.testing.assert_array_equal(order[:BATCH], warm)
    if any(len(np.unique(order)) != max(BUDGETS) for order in arms.values()):
        raise ValueError('Incorrect number of unique acquisitions')

    # Select samples before reading evaluation labels or generating corruption.
    clean = np.load(reference.BANK / 'train_targets.npy', allow_pickle=False)
    test_truth = np.load(reference.BANK / 'test_targets.npy', allow_pickle=False)
    successor = np.empty(reference.CLASSES, dtype=np.int64)
    groups = json.loads(reference.GROUPS.read_text())
    for group in groups.values():
        successor[group] = np.roll(group, -1)
    observed = {name: reference.corrupt_labels(clean, name, rate, seed, successor)
                for name, rate in reference.CONDITIONS}
    observed['cifar100n'] = reference.human_annotations(clean)
    eye = np.eye(reference.CLASSES, dtype=np.float32)
    results = []
    for arm, anchors in arms.items():
        _, test_votes = reference.path_counts(neighbors, heldout, anchors, depth=3)
        distances = test_features @ features[anchors].T
        for budget in BUDGETS:
            selected = rows[anchors[:budget]]
            votes = test_votes[:, :budget].astype(np.float32)
            nearest = distances[:, :budget].argmax(axis=1)
            selected_edges = np.isin(candidate, anchors[:budget])
            target_coverage = np.bincount(target[selected_edges], minlength=len(rows))
            unqueried = ~np.isin(np.arange(len(rows)), anchors[:budget])
            for condition, noise_labels in observed.items():
                annotations = noise_labels[selected]
                score = votes @ eye[annotations]
                if not np.isfinite(score).all():
                    raise ValueError('Nonfinite test votes')
                results.append(dict(seed=seed, budget=budget, condition=condition, arm=arm,
                                    test_correct=int((score.argmax(1) == test_truth).sum()), test_count=len(test_truth),
                                    nn_correct=int((annotations[nearest] == test_truth).sum()),
                                    test_supported=int((score.max(1) > 0).sum()),
                                    wrong_annotations=int((annotations != clean[selected]).sum()),
                                    pool_count=int(unqueried.sum()),
                                    pool_closest_count=int((target_coverage[unqueried] >= 1).sum()),
                                    pool_backup_count=int((target_coverage[unqueried] >= 2).sum())))
            print(f'{seed} {arm} {budget} verified', flush=True)
    value = dict(records=results, acquisitions={k: rows[v].tolist() for k, v in arms.items()},
                 source_sha256=reference.sha256(Path(__file__)), geometry_sha256=GEOMETRY_SHA256,
                 helper_sha256=reference.sha256(Path(reference.__file__)), numpy=np.__version__)
    tmp = output.with_suffix('.tmp')
    with tmp.open('x') as stream:
        json.dump(value, stream, sort_keys=True)
    tmp.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--geometry', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    check_selector()
    if args.seed is not None:
        run(args.seed, args.geometry, args.output)


if __name__ == '__main__':
    main()
