"""Test whether backup coverage needs representatives from separate feature groups."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import backup_coverage as baseline

SEEDS = (70, 71, 72)
BUDGETS = (800, 1600)
BATCH = 100
BACKUP = .5
SELECTOR_GROUPS = (512, 764)
STRESS_GROUPS = (256, 188)
SHUFFLE_SEED = 765
CONDITIONS = ('clean', 'iid50', 'cyclic50', 'cluster50', 'iid_rate_matched', 'cifar100n')
ARMS = ('max', 'backup', 'grouped', 'shuffled')


def edge_gain(similarity, first, second, shares_primary, weight):
    primary_gain = np.maximum(similarity - first, 0)
    backup_gain = np.maximum(similarity - second, 0)
    return primary_gain + weight * np.where(shares_primary, 0, backup_gain - primary_gain)


def grouped_select(target, candidate, similarity, groups, count, batch_size, seed, warm):
    """Greedy top-two coverage after taking a maximum within each source group."""
    n = len(groups)
    target_offset = np.r_[0, np.cumsum(np.bincount(target, minlength=n))]
    target_degree = np.diff(target_offset)
    by_candidate = np.argsort(candidate, kind='stable')
    candidate_offset = np.r_[0, np.cumsum(np.bincount(candidate, minlength=n))]
    tie = np.random.default_rng(seed).random(n) * 1e-10
    first = np.zeros(n, dtype=np.float64)
    second = np.zeros_like(first)
    primary_group = np.full(n, -1, dtype=np.int64)
    selected = np.zeros(n, dtype=bool)
    edge_groups = groups[candidate]
    order = []
    for begin in range(0, count, batch_size):
        weight = BACKUP if begin else 0.
        demand = ~selected
        contribution = edge_gain(similarity, first[target], second[target],
                                 edge_groups == primary_group[target], weight)
        gain = np.bincount(candidate, weights=contribution * demand[target], minlength=n)
        gain[selected] = -np.inf
        for offset in range(min(batch_size, count - begin)):
            picked = int(warm[offset]) if begin == 0 else int(np.argmax(gain + tie))
            order.append(picked)
            selected[picked] = True
            gain[picked] = -np.inf

            edges = by_candidate[candidate_offset[picked]:candidate_offset[picked + 1]]
            affected = target[edges]
            new = similarity[edges]
            old_first = first[affected].copy()
            old_second = second[affected].copy()
            old_group = primary_group[affected].copy()
            new_group = groups[picked]
            same_group = old_group == new_group
            second[affected] = np.where(same_group, old_second, np.maximum(old_second, np.minimum(old_first, new)))
            first[affected] = np.maximum(old_first, new)
            primary_group[affected] = np.where(new > old_first, new_group, old_group)

            lengths = target_degree[affected]
            owner = np.repeat(np.arange(len(affected)), lengths)
            start = np.repeat(target_offset[affected] - (np.cumsum(lengths) - lengths), lengths)
            adjacent = np.arange(lengths.sum()) + start
            dest = target[adjacent]
            adj_group = edge_groups[adjacent]
            old_gain = edge_gain(similarity[adjacent], old_first[owner], old_second[owner],
                                 adj_group == old_group[owner], weight)
            new_gain = edge_gain(similarity[adjacent], first[dest], second[dest],
                                 adj_group == primary_group[dest], weight)
            np.add.at(gain, candidate[adjacent], (new_gain - old_gain) * demand[dest])
    return np.asarray(order)


def landmark_groups(features, neighbors, count, seed, coverage_order):
    landmarks = coverage_order(neighbors, count, seed)
    centroids = np.ascontiguousarray(features[landmarks].T)
    groups = np.empty(len(features), dtype=np.int32)
    for start in range(0, len(features), 4096):
        part = slice(start, start + 4096)
        groups[part] = np.argmax(features[part] @ centroids, axis=1)
    if len(np.unique(groups)) != count:
        raise ValueError('Partition contains empty groups')
    return groups, landmarks


def paired_observations(truth, stress_groups, successor, seed):
    """Use common wrong-class draws and align the two IID mask comparisons."""
    rng = np.random.default_rng(seed)
    row_order = rng.permutation(len(truth))
    raw_wrong = rng.integers(len(successor) - 1, size=len(truth))
    wrong = raw_wrong + (raw_wrong >= truth)
    cluster_wrong = rng.random(int(stress_groups.max()) + 1) < .5
    cluster_mask = cluster_wrong[stress_groups]
    iid_mask = np.zeros(len(truth), dtype=bool)
    iid_mask[row_order[:len(truth) // 2]] = True
    matched_mask = np.zeros(len(truth), dtype=bool)
    matched_mask[row_order[:int(cluster_mask.sum())]] = True
    return {
        'clean': truth,
        'iid50': np.where(iid_mask, wrong, truth),
        'cyclic50': np.where(iid_mask, successor[truth], truth),
        'cluster50': np.where(cluster_mask, wrong, truth),
        'iid_rate_matched': np.where(matched_mask, wrong, truth),
    }


def array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values)).hexdigest()


def within_group_error_covariance(wrong, groups):
    total = np.bincount(groups).astype(float)
    errors = np.bincount(groups, weights=wrong, minlength=len(total))
    pairs = np.sum(total * (total - 1))
    both = np.sum(errors * (errors - 1)) / pairs
    one = np.sum(errors * (total - 1)) / pairs
    return float(both - one**2)


def run(seed, geometry, output):
    import active_audit as reference

    if output.exists():
        raise FileExistsError(output)
    if reference.sha256(geometry) != baseline.GEOMETRY_SHA256:
        raise ValueError('Frozen geometry checksum changed')
    for name, expected in reference.BANK_HASHES.items():
        if reference.sha256(reference.BANK / f'{name}.npy') != expected:
            raise ValueError(f'Frozen {name} checksum changed')
    with np.load(geometry, allow_pickle=False) as artifact:
        rows, neighbors, heldout = artifact['rows'], artifact['neighbors'], artifact['heldout_neighbors']
    if neighbors.shape != (45000, 13) or heldout.shape != (10000, 13):
        raise ValueError('Unexpected geometry')
    features = reference.unit_rows(np.load(reference.BANK / 'train_features.npy', mmap_mode='r')[rows])
    test_features = reference.unit_rows(np.load(reference.BANK / 'test_features.npy', allow_pickle=False))
    groups, landmarks = landmark_groups(features, neighbors, *SELECTOR_GROUPS, reference.coverage_order)
    stress_groups, stress_landmarks = landmark_groups(features, neighbors, *STRESS_GROUPS, reference.coverage_order)
    shuffled_groups = np.random.default_rng(SHUFFLE_SEED).permutation(groups)
    np.testing.assert_array_equal(np.bincount(groups), np.bincount(shuffled_groups))

    target, candidate, similarity = baseline.sparse_affinities(neighbors, features)
    warm = reference.coverage_order(neighbors, BATCH, seed)
    count = max(BUDGETS)
    arms = {
        'max': baseline.select(target, candidate, similarity, len(rows), count, BATCH, seed, 0., warm),
        'backup': baseline.select(target, candidate, similarity, len(rows), count, BATCH, seed, BACKUP, warm),
        'grouped': grouped_select(target, candidate, similarity, groups, count, BATCH, seed, warm),
        'shuffled': grouped_select(target, candidate, similarity, shuffled_groups, count, BATCH, seed, warm),
    }
    for order in arms.values():
        np.testing.assert_array_equal(order[:BATCH], warm)
        if len(np.unique(order)) != count:
            raise ValueError('Duplicate acquisitions')

    clean = np.load(reference.BANK / 'train_targets.npy', allow_pickle=False)
    truth = clean[rows]
    test_truth = np.load(reference.BANK / 'test_targets.npy', allow_pickle=False)
    successor = np.empty(reference.CLASSES, dtype=int)
    for label_group in json.loads(reference.GROUPS.read_text()).values():
        successor[label_group] = np.roll(label_group, -1)
    observed = paired_observations(truth, stress_groups, successor, seed)
    observed['cifar100n'] = reference.human_annotations(clean)[rows]
    noise = {
        name: {
            'fraction': float(np.mean(labels != truth)),
            'observed_sha256': array_hash(labels),
            'within_selector_covariance': within_group_error_covariance(labels != truth, groups),
            'within_shuffled_covariance': within_group_error_covariance(labels != truth, shuffled_groups),
            'within_stress_covariance': within_group_error_covariance(labels != truth, stress_groups),
        } for name, labels in observed.items()
    }

    eye = np.eye(reference.CLASSES, dtype=np.float32)
    records = []
    for arm, order in arms.items():
        path_counts = baseline.test_paths(neighbors, heldout, order)
        similarity_to_test = test_features @ features[order].T
        for budget in BUDGETS:
            selected = order[:budget]
            nearest = similarity_to_test[:, :budget].argmax(axis=1)
            selected_edges = np.isin(candidate, selected)
            unqueried = ~np.isin(np.arange(len(rows)), selected)
            coverage_count = np.bincount(target[selected_edges], minlength=len(rows))
            pair_ids = target[selected_edges].astype(np.int64) * SELECTOR_GROUPS[0] + groups[candidate[selected_edges]]
            group_count = np.bincount(np.unique(pair_ids) // SELECTOR_GROUPS[0], minlength=len(rows))
            labels = [observed[name][selected] for name in CONDITIONS]
            basis = np.concatenate([eye[annotations] for annotations in labels], axis=1)
            for depth, counts in path_counts.items():
                weights = counts[:, :budget].astype(np.float32)
                scores = (weights @ basis).reshape(len(heldout), len(CONDITIONS), reference.CLASSES)
                if not np.isfinite(scores).all():
                    raise ValueError('Invalid graph scores')
                predictions = scores.argmax(axis=-1)
                supported = int((weights.max(axis=1) > 0).sum())
                for i, (name, annotations) in enumerate(zip(CONDITIONS, labels)):
                    records.append(dict(
                        seed=seed, arm=arm, depth=depth, condition=name, budget=budget,
                        test_count=len(test_truth), test_supported=supported,
                        correct=int((predictions[:, i] == test_truth).sum()),
                        nn_correct=int((annotations[nearest] == test_truth).sum()),
                        wrong_annotations=int((annotations != truth[selected]).sum()),
                        pool_count=int(unqueried.sum()),
                        pool_first=int((coverage_count[unqueried] >= 1).sum()),
                        pool_second=int((coverage_count[unqueried] >= 2).sum()),
                        pool_distinct_group_second=int((group_count[unqueried] >= 2).sum()),
                    ))
            print(f'{seed} {arm} {budget} verified', flush=True)
    result = dict(
        seed=seed, records=records, noise=noise,
        group_sizes=np.bincount(groups).tolist(),
        stress_group_sizes=np.bincount(stress_groups).tolist(),
        landmarks=rows[landmarks].tolist(), stress_landmarks=rows[stress_landmarks].tolist(),
        partitions={name: array_hash(values) for name, values in (
            ('selector', groups), ('stress', stress_groups), ('shuffle', shuffled_groups)
        )},
        geometry_sha256=baseline.GEOMETRY_SHA256,
        revisions={name: reference.sha256(path) for name, path in (
            ('selector', Path(__file__)), ('baseline', Path(baseline.__file__)),
            ('helper', Path(reference.__file__)), ('protocol', Path(__file__).with_name('PROTOCOL.md'))
        )},
        acquisitions={name: rows[indices].tolist() for name, indices in arms.items()},
    )
    temporary = output.with_suffix('.tmp')
    with temporary.open('x') as stream:
        json.dump(result, stream, sort_keys=True)
    temporary.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    opts = parser.parse_args()
    run(opts.seed, opts.geometry, opts.output)
