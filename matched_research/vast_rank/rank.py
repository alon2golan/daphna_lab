"""Rank-only protected VAST targets selected using fold-excluded acquired labels."""

import argparse
import json
from pathlib import Path

import numpy as np

BUDGETS = (800, 1200)
CONDITIONS = ('sym0', 'sym30', 'sym50', 'asym50')
CASES = tuple(f'b{budget}_{condition}' for budget in BUDGETS for condition in CONDITIONS)
DEPTHS = (0, 1, 3)
FOLDS = 5
FOLD_SEED = 20261127
FIELDS = ('probabilities', 'confidence', 'admitted')


def align_rank(raw, proposed, depth):
    """Move raw values onto the proposed prefix using successive transpositions."""
    revised = raw.copy()
    if not depth:
        return revised
    rows = np.arange(len(raw))
    holders = np.argsort(-raw, axis=1, kind='stable')[:, :depth].copy()
    desired = np.argsort(-proposed, axis=1, kind='stable')[:, :depth]
    for rank, target in enumerate(desired.T):
        source = holders[:, rank].copy()
        revised[rows, source], revised[rows, target] = revised[rows, target], revised[rows, source]
        later = holders[:, rank + 1:]
        np.copyto(later, source[:, None], where=later == target[:, None])
    return revised


def fold_ids(size):
    assignment = np.empty(size, dtype=np.int64)
    assignment[np.random.default_rng(FOLD_SEED).permutation(size)] = np.arange(size) % FOLDS
    return assignment


def replay_folds(labeled, observed, assignment, predict):
    """Collect each row's predictions from a teacher that excluded its fold."""
    blocks = []
    for fold in range(FOLDS):
        heldout = assignment == fold
        blocks.append(predict(labeled[~heldout], observed[~heldout], labeled[heldout]))
    restore = np.argsort(np.argsort(assignment, kind='stable'))
    raw, proposed = zip(*blocks)
    return np.concatenate(raw)[restore], np.concatenate(proposed)[restore]


def crossfit_depth(raw, proposed, observed):
    row = np.arange(len(observed))
    loss = np.stack([-np.log(align_rank(raw, proposed, depth)[row, observed].astype(np.float64))
                     for depth in DEPTHS], axis=1)
    means = loss.mean(axis=0)
    if not np.isfinite(means).all():
        raise ValueError('Nonfinite acquired-label cross-entropy')
    return DEPTHS[int(means.argmin())], loss


def write_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, allow_nan=False, indent=2) + '\n')
    tmp.replace(path)
    print(json.dumps(data, allow_nan=False), flush=True)


def load_targets(path):
    with np.load(path, allow_pickle=False) as saved:
        return {field: saved[field] for field in FIELDS}


def prepare(archive, root_targets, output, index):
    from scipy.sparse import load_npz
    from deep_al.vast_noisy.submission20261019.adapters import seed_evidence
    from source_support_study.replay import DEPTH, checked_file, checksum, read, teacher
    from transfer import compress

    name = CASES[index]
    plan = read(archive / 'plan.json')
    case, saved_root = archive / 'cells' / name, root_targets / name
    fixed = read(case / 'fixed_targets.json')
    original_name = f'A_{DEPTH}.npz'
    original_sha = fixed['hashes'][original_name]
    raw = load_targets(checked_file(case / original_name, original_sha))
    graph_info = plan['graph']
    graph = load_npz(checked_file(Path(graph_info['path']), graph_info['kernel_sha256'])).tocsr()
    prepared = case / 'inputs'
    digests = read(prepared / 'preparation.json')['prepared_sha256']

    def acquired_file(filename):
        return np.load(checked_file(prepared / filename, digests[filename]), allow_pickle=False)

    labeled = acquired_file('partitions/labeled_indices.npy')
    unlabeled = acquired_file('partitions/unlabeled_indices.npy')
    observed = acquired_file('observed_labeled_labels.npy')
    if not np.isin(np.flatnonzero(raw['admitted']), unlabeled).all():
        raise ValueError('Protected admission includes ineligible rows')
    classes = raw['probabilities'].shape[1]
    np.testing.assert_array_equal(observed.shape, labeled.shape)
    assignment = fold_ids(len(labeled))
    errors = []

    def predict(keep_rows, keep_labels, heldout_rows):
        initial = seed_evidence(graph, keep_rows, keep_labels, classes)
        evidence, gates = teacher(graph, initial, keep_rows, keep_labels)
        original = evidence / evidence.sum(axis=1, keepdims=True)
        proposal, error = compress(graph, keep_rows, keep_labels, original, gates)
        errors.append(error)
        print(f'{name} fold {len(errors)}/{FOLDS}: {len(heldout_rows)} held out, reconstruction {error:.3g}', flush=True)
        return original[heldout_rows], proposal[heldout_rows]

    raw_oof, proposed_oof = replay_folds(labeled, observed, assignment, predict)
    selected, loss = crossfit_depth(raw_oof, proposed_oof, observed)
    # The full-data target is opened after the OOF decision has been made.
    root_info = read(saved_root / 'result.json')
    if root_info['original_sha256'] != original_sha or root_info['depth'] != DEPTH:
        raise ValueError('Full proposed ranks use a different protected teacher')
    root = load_targets(checked_file(saved_root / 'targets.npz', root_info['revised_sha256']))
    gate = raw['admitted']
    for field in ('admitted', 'confidence'):
        np.testing.assert_array_equal(raw[field], root[field])
    np.testing.assert_array_equal(raw['probabilities'][labeled], root['probabilities'][labeled])
    revised = raw['probabilities'].copy()
    revised[gate] = align_rank(raw['probabilities'][gate], root['probabilities'][gate], selected)
    np.testing.assert_array_equal(np.sort(revised[gate], axis=1), np.sort(raw['probabilities'][gate], axis=1))
    np.testing.assert_array_equal(revised[~gate], raw['probabilities'][~gate])
    output.mkdir(parents=True, exist_ok=False)
    target_path = output / 'targets.npz'
    np.savez_compressed(target_path, probabilities=revised, admitted=gate, confidence=raw['confidence'])
    np.savez_compressed(output / 'oof.npz', labeled=labeled, observed=observed, fold=assignment, loss=loss)
    result = dict(case=name, depth=DEPTH, folds=FOLDS, fold_seed=FOLD_SEED, chosen=selected,
                  mean_acquired_ce=dict(zip(DEPTHS, loss.mean(axis=0).tolist())),
                  fold_acquired_ce=[loss[assignment == k].mean(axis=0).tolist() for k in range(FOLDS)],
                  candidates=DEPTHS, labeled=len(labeled), admitted=int(gate.sum()),
                  changed_rows=int((revised[gate] != raw['probabilities'][gate]).any(axis=1).sum()),
                  changed_winners=int((revised[gate].argmax(1) != raw['probabilities'][gate].argmax(1)).sum()),
                  oof_reconstruction_error=max(errors), original_sha256=original_sha,
                  graph_sha256=graph_info['kernel_sha256'], proposed_sha256=root_info['revised_sha256'],
                  revised_sha256=checksum(target_path), oof_sha256=checksum(output / 'oof.npz'),
                  source_sha256=checksum(Path(__file__)))
    write_json(output / 'result.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', type=int, choices=range(len(CASES)))
    parser.add_argument('archive', type=Path)
    parser.add_argument('root_targets', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    prepare(args.archive, args.root_targets, args.output / CASES[args.index], args.index)


if __name__ == '__main__':
    main()
