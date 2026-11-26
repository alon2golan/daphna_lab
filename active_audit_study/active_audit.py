"""Paired clean-label audit and purchase replay on frozen CIFAR-100 BYOL features."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

BANK = Path('/cs/labs/daphna/itai.david/representations_bank/cifar-100_BYOL')
PRIOR = Path('/cs/labs/daphna/tomn/projects/cifar100_herding_20261119/study/inputs')
GROUPS = Path('/cs/labs/daphna/tomn/projects/vast_noisy_inputs_fc280c3/cifar100_superclasses.json')
BANK_HASHES = {
    'train_features': '4e91020dcd15d77dcbe35ad32e63020bd156ad610808866ae49eb696d2035476',
    'train_targets': '233cdd4afc94cd00f3ddec6f8640168bf76a6872168cf6c5a9990ac9f144be3b',
    'test_features': 'd7ba6cfb072039698da1b4e67fe97636ab0d4e6b6158bd9d66b42959bab18e36',
    'test_targets': '7827272d0fbeee517a81b235f35fae2128af64393371c2fdde1c8f9a638e4a1a',
}
CLASSES = 100
BUDGETS = (400, 800)
DEPTH = 3
CONDITIONS = (('clean', 0.), ('sym20', .2), ('sym50', .5), ('asym20', .2), ('asym50', .5))
SEEDS = (0, 1, 2, 3, 4)


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def unit_rows(features):
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm <= 0):
        raise ValueError('Missing or invalid feature rows')
    return np.ascontiguousarray(features / norm, dtype=np.float32)


def coverage_order(neighbors, size, seed):
    """Greedy binary coverage of reverse 13NN plus self, with seeded ties."""
    n = len(neighbors)
    links = np.column_stack((np.arange(n), neighbors))
    edge_sources = np.argsort(links.ravel(), kind='stable') // links.shape[1]
    gains = np.bincount(links.ravel(), minlength=n)
    offsets = np.r_[0, np.cumsum(gains)]
    covered = np.zeros(n, dtype=bool)
    tie = np.random.default_rng(seed).random(n)
    order = np.empty(size, dtype=np.int64)
    for rank in range(size):
        anchor = int(np.argmax(gains + tie))
        order[rank] = anchor
        targets = edge_sources[offsets[anchor]:offsets[anchor + 1]]
        fresh = targets[~covered[targets]]
        covered[fresh] = True
        np.add.at(gains, links[fresh].ravel(), -1)
        gains[anchor] = -1
    return order


def path_counts(neighbors, evaluation_neighbors, anchors, depth):
    active = np.zeros((len(neighbors), len(anchors)), dtype=np.int16)
    active[anchors, np.arange(len(anchors))] = 1
    next_round = np.empty_like(active)
    heldout = np.empty((len(evaluation_neighbors), len(anchors)), dtype=np.int16)
    for step in range(depth):
        if step == depth - 1:
            for start in range(0, len(heldout), 256):
                heldout[start:start + 256] = active[evaluation_neighbors[start:start + 256]].sum(1, dtype=np.int16)
        for start in range(0, len(active), 256):
            next_round[start:start + 256] = active[neighbors[start:start + 256]].sum(1, dtype=np.int16)
        active, next_round = next_round, active
    return active, heldout


def leave_one_out(anchors_to_anchors, observed, eyes):
    weights = anchors_to_anchors.copy()
    np.fill_diagonal(weights, 0)
    scores = weights.astype(np.float32) @ eyes[observed]
    rows = np.arange(len(observed))
    self_score = scores[rows, observed].copy()
    scores[rows, observed] = -1
    alt = scores.argmax(1)
    support = weights.sum(1)
    margin = (scores[rows, alt] - self_score) / np.maximum(support, 1)
    return margin, alt


def counterfactual_footprints(votes, scores, observed, alternative):
    """For each audit, find unqueried rows that would stop predicting its label."""
    rows = np.arange(len(scores))
    best = scores.argmax(1)
    without_best = scores.copy()
    without_best[rows, best] = -1
    second = without_best.argmax(1)
    footprints = []
    for anchor, (original, replacement) in enumerate(zip(observed, alternative)):
        candidates = np.flatnonzero((best == original) & (votes[:, anchor] > 0))
        contribution = votes[candidates, anchor]
        old = scores[candidates, original] - contribution
        proposed = scores[candidates, replacement] + contribution
        runner = scores[candidates, second[candidates]]
        changed = (
            (proposed > old) | ((proposed == old) & (replacement < original))
            | (runner > old) | ((runner == old) & (second[candidates] < original))
        )
        footprints.append(candidates[changed])
    return footprints


def rank_audits(suspicion, footprints, pool_size, count):
    gate = suspicion.clip(min=0)
    indices = np.arange(len(gate))
    initial = gate * np.array([len(group) for group in footprints])
    independent = np.lexsort((indices, -suspicion, -initial))[:count]
    covered = np.zeros(pool_size)
    ideal_order = np.lexsort((indices, -suspicion))
    chosen = []
    remaining = set(indices.tolist())
    for _ in range(count):
        best, best_delta = None, -1.
        for candidate in ideal_order:
            if candidate in remaining:
                gain = np.maximum(gate[candidate] - covered[footprints[candidate]], 0.).sum()
                if gain > best_delta:
                    best, best_delta = candidate, gain
        chosen.append(best)
        remaining.remove(best)
        group = footprints[best]
        covered[group] = np.maximum(covered[group], gate[best])
    return independent, np.asarray(chosen)


def corrupt_labels(clean, kind, rate, seed, successor):
    """Match the lab's exact symmetric and superclass-cycle RNG draw order."""
    rng = np.random.default_rng(seed)
    corrupted = clean.copy()
    if kind.startswith('sym'):
        sample = rng.choice(len(clean), int(rate * len(clean)), replace=False)
        draws = rng.integers(CLASSES - 1, size=len(sample))
        corrupted[sample] = draws + (draws >= clean[sample])
    if kind.startswith('asym'):
        shifted = successor[clean]
        stay = np.float64(np.float32(1. - rate))
        flip = np.float64(np.float32(rate))
        stay /= stay + flip
        cutoff = np.where(clean < shifted, stay, 1. - stay)
        first = rng.random(len(clean)) < cutoff
        corrupted = np.where(first, np.minimum(clean, shifted), np.maximum(clean, shifted))
    return corrupted


def prepare(graph_path, destination):
    import faiss

    if destination.exists():
        raise FileExistsError(destination)
    for name, expected in BANK_HASHES.items():
        if sha256(BANK / f'{name}.npy') != expected:
            raise ValueError(f'BYOL checksum mismatch for {name}')
    with np.load(graph_path, allow_pickle=False) as graph:
        rows = graph['row_indices']
        neighbors = graph['neighbors'].astype(np.int32)
        if json.loads(graph['metadata'].item())['feature_sha256'] != BANK_HASHES['train_features']:
            raise ValueError('Graph and BYOL bank disagree')
    np.testing.assert_array_equal(rows, np.load(PRIOR / 'eligible.npy', allow_pickle=False))
    if (rows.shape != (45000,) or neighbors.shape != (45000, 13)
            or neighbors.min() < 0 or neighbors.max() >= len(rows)
            or np.any(neighbors == np.arange(len(rows))[:, None])):
        raise ValueError('Invalid exported 13NN graph')
    train = unit_rows(np.load(BANK / 'train_features.npy', mmap_mode='r')[rows])
    test = unit_rows(np.load(BANK / 'test_features.npy', allow_pickle=False))
    if train.shape != (45000, 512) or test.shape != (10000, 512):
        raise ValueError('Unexpected BYOL array shape')
    index = faiss.IndexFlatIP(train.shape[1])
    index.add(train)
    _, heldout_neighbors = index.search(test, neighbors.shape[1])
    metadata = dict(graph_sha256=sha256(graph_path), bank_sha256=BANK_HASHES, faiss=faiss.__version__,
                    groups_sha256=sha256(GROUPS), test_rows=len(test), train_rows=len(train))
    temporary = destination.with_suffix('.tmp')
    with temporary.open('xb') as target:
        np.savez_compressed(target, rows=rows, neighbors=neighbors,
                            heldout_neighbors=heldout_neighbors, metadata=json.dumps(metadata))
    temporary.rename(destination)
    print(json.dumps(metadata), flush=True)


def replay(seed, prepared, output):
    if output.exists():
        raise FileExistsError(output)
    with np.load(prepared, allow_pickle=False) as payload:
        rows, neighbors, test_neighbors = payload['rows'], payload['neighbors'], payload['heldout_neighbors']
        provenance = json.loads(payload['metadata'].item())
    if provenance['bank_sha256'] != BANK_HASHES or provenance['groups_sha256'] != sha256(GROUPS):
        raise ValueError('Prepared data provenance mismatch')
    clean = np.load(BANK / 'train_targets.npy', allow_pickle=False)
    truth = np.load(BANK / 'test_targets.npy', allow_pickle=False)
    eyes = np.eye(CLASSES, dtype=np.float32)
    groups = list(json.loads(GROUPS.read_text()).values())
    if len(groups) != 20 or any(len(group) != 5 for group in groups) or sorted(sum(groups, [])) != list(range(CLASSES)):
        raise ValueError('Expected 20 disjoint groups of five CIFAR-100 classes')
    successor = np.empty(CLASSES, dtype=np.int64)
    for group in groups:
        successor[group] = np.roll(group, -1)
    corrupted = {name: corrupt_labels(clean, name, rate, seed, successor) for name, rate in CONDITIONS}
    if seed == 0:
        for name, observed in corrupted.items():
            np.testing.assert_array_equal(observed, np.load(PRIOR / f'{name}_observed.npy', allow_pickle=False))
    order = coverage_order(neighbors, max(BUDGETS) * 11 // 10, seed)
    if len(np.unique(order)) != len(order):
        raise ValueError('Repeated acquisition')
    train_votes, test_votes = path_counts(neighbors, test_neighbors, order, DEPTH)
    feature_anchors = unit_rows(np.load(BANK / 'train_features.npy', mmap_mode='r')[rows[order]])
    test_features = unit_rows(np.load(BANK / 'test_features.npy', allow_pickle=False))
    nearest = test_features @ feature_anchors.T
    if not np.isfinite(nearest).all() or np.any(train_votes < 0) or np.any(test_votes < 0):
        raise ValueError('Nonfinite similarities or path count overflow')
    results = []
    for budget in BUDGETS:
        extra = budget // 10
        used = budget + extra
        warm, fresh = order[:budget], order[budget:used]
        target_mask = np.ones(len(rows), dtype=bool)
        target_mask[order[:used]] = False
        influence = train_votes[target_mask, :budget].astype(np.float32)
        local = train_votes[warm, :budget]
        nearest_base = nearest[:, :budget].argmax(axis=1)
        nearest_full = nearest[:, :used].argmax(axis=1)
        train_matrices = (influence, train_votes[target_mask, :used].astype(np.float32))
        test_matrices = (test_votes[:, :budget].astype(np.float32), test_votes[:, :used].astype(np.float32))
        for condition, _ in CONDITIONS:
            started = time.monotonic()
            observed = corrupted[condition][rows[warm]]
            scores = influence @ eyes[observed]
            suspicion, alternative = leave_one_out(local, observed, eyes)
            footprints = counterfactual_footprints(influence, scores, observed, alternative)
            single, diverse = rank_audits(suspicion, footprints, len(influence), extra)
            random = np.random.default_rng(100_000 + seed * 10_000 + budget).choice(budget, extra, replace=False)
            loo = np.argsort(-suspicion, kind='stable')[:extra]
            for policy, audits in (
                ('no_audit', np.empty(0, dtype=int)), ('random_audit', random), ('loo_audit', loo),
                ('single_footprint', single), ('diverse_footprint', diverse), ('fresh_clean', np.empty(0, dtype=int)),
            ):
                # The clean oracle is accessed only after all actions are fixed.
                labels = observed.copy()
                labels[audits] = clean[rows[warm[audits]]]
                expand = policy == 'fresh_clean'
                if expand:
                    labels = np.r_[labels, clean[rows[fresh]]]
                onehot = eyes[labels]
                test_scores = test_matrices[expand] @ onehot
                pool_scores = train_matrices[expand] @ onehot
                if not np.isfinite(test_scores).all() or not np.isfinite(pool_scores).all():
                    raise ValueError('Nonfinite prediction')
                result = dict(seed=seed, budget=budget, condition=condition, policy=policy,
                              clean_requests=extra if expand else len(audits),
                              initial_wrong=int((observed != clean[rows[warm]]).sum()),
                              corrected=int((labels[:budget] != observed).sum()),
                              pool_correct=int((pool_scores.argmax(1) == clean[rows[target_mask]]).sum()),
                              pool_count=int(target_mask.sum()), test_count=len(truth),
                              test_correct=int((test_scores.argmax(1) == truth).sum()),
                              test_1nn_correct=int((labels[nearest_full if expand else nearest_base] == truth).sum()),
                              request_rows=rows[fresh if expand else warm[audits]].tolist())
                results.append(result)
            print(seed, budget, condition, round(time.monotonic() - started, 2), flush=True)
    result = dict(rows=results, acquisition_rows=rows[order].tolist(), source_sha256=sha256(Path(__file__)),
                  geometry_sha256=sha256(prepared), provenance=provenance, depth=DEPTH, numpy=np.__version__)
    temporary = output.with_suffix('.tmp')
    with temporary.open('x') as handle:
        json.dump(result, handle, sort_keys=True)
    temporary.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    run = commands.add_parser('prepare')
    run.add_argument('graph', type=Path)
    run.add_argument('output', type=Path)
    run = commands.add_parser('replay')
    run.add_argument('seed', type=int, choices=SEEDS)
    run.add_argument('prepared', type=Path)
    run.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.graph, args.output)
    else:
        replay(args.seed, args.prepared, args.output)


if __name__ == '__main__':
    main()
