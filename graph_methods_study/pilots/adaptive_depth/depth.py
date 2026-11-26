"""Compare fixed and confidence-selected graph walk depths with the same seed labels."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

BUDGETS = (200, 400, 800, 1200)
DEPTHS = (1, 2, 3, 4, 5, 6)
CONDITIONS = (('clean', 0.),) + tuple(
    (f'{kind}{rate}', rate / 100) for kind in ('sym', 'asym') for rate in (30, 40, 50)
)
SEEDS = tuple(range(10, 15))


def neighbors_matrix(neighbors, columns):
    rows, degree = neighbors.shape
    return csr_matrix(
        (np.ones(rows * degree, dtype=np.float32), neighbors.ravel(),
         np.arange(rows + 1, dtype=np.int32) * degree),
        shape=(rows, columns),
    )


def margin_score(evidence, scale):
    """Standardized top-two gap under the chosen scale, with no rating for empty rows."""
    top = np.partition(evidence, -2, axis=1)[:, -2:]
    difference = top[:, 1] - top[:, 0]
    available = scale > 0
    margin = np.divide(difference, scale, out=np.full(len(scale), -np.inf), where=available)
    return margin, available


def run(geometry, saved_order, human, seed, output):
    import active_audit as source

    with np.load(geometry, allow_pickle=False) as archive:
        rows, neighbors, heldout = (archive[name] for name in ('rows', 'neighbors', 'heldout_neighbors'))
        info = json.loads(archive['metadata'].item())
    if info['bank_sha256'] != source.BANK_HASHES or info['groups_sha256'] != source.sha256(source.GROUPS):
        raise ValueError('Expected verified BYOL graph and class mapping')
    order_rows = json.loads(saved_order.read_text())['acquisition_rows']
    order = source.replay_order(neighbors, rows, max(BUDGETS), seed, order_rows)
    clean = np.load(source.BANK / 'train_targets.npy', allow_pickle=False)
    truth = np.load(source.BANK / 'test_targets.npy', allow_pickle=False)
    groups = list(json.loads(source.GROUPS.read_text()).values())
    successor = np.empty(source.CLASSES, dtype=int)
    for group in groups:
        successor[group] = np.roll(group, -1)
    labels = {name: source.corrupt_labels(clean, name, rate, seed, successor)[rows[order]]
              for name, rate in CONDITIONS}
    labels['cifar100n'] = np.load(human, allow_pickle=False)[rows[order]]
    eyes = np.eye(source.CLASSES, dtype=np.float32)
    onehot = np.concatenate([eyes[y] for y in labels.values()], axis=1)
    features = np.load(source.BANK / 'train_features.npy', mmap_mode='r')[rows[order]]
    test_features = source.unit_rows(np.load(source.BANK / 'test_features.npy', allow_pickle=False))
    similarities = test_features @ source.unit_rows(features).T
    fallback = {b: similarities[:, :b].argmax(axis=1) for b in BUDGETS}

    walk = neighbors_matrix(neighbors, len(rows))
    predict = neighbors_matrix(heldout, len(rows))
    basis = np.zeros((len(rows), len(order)), dtype=np.float32)
    basis[order, np.arange(len(order))] = 1
    cases = {(b, c): {'fixed': {}, 'best': {}, 'oracle': np.zeros(len(truth), dtype=bool)}
             for b in BUDGETS for c in labels}
    for d in DEPTHS:
        weights = predict @ basis
        if d != DEPTHS[-1]:
            basis = walk @ basis
        if d == 3:
            # On a uniformly drawn training point, this is proportional to the
            # probability that the three-step walk reaches each anchor.
            inverse_mass = np.reciprocal(np.maximum(basis.sum(axis=0), 1))
        for budget in BUDGETS:
            window = weights[:, :budget]
            standard = np.sqrt(np.einsum('ij,ij->i', window, window, dtype=np.float64))
            mass = window.sum(axis=1, dtype=np.float64)
            votes = (window @ onehot[:budget]).reshape(len(truth), len(labels), source.CLASSES)
            if d == 3:
                balanced = ((window * inverse_mass[:budget]) @ onehot[:budget]).reshape(votes.shape)
            for k, (name, observed) in enumerate(labels.items()):
                case = cases[budget, name]
                view = votes[:, k]
                raw, available = margin_score(view, mass)
                standardized, _ = margin_score(view, standard)
                winner = view.argmax(axis=1)
                predicted = np.where(available, winner, observed[fallback[budget]])
                correct = predicted == truth
                case['fixed'][str(d)] = int(correct.sum())
                case['oracle'] |= correct
                if d == 3:
                    case['published_3'] = int((winner == truth).sum())
                    balanced_winner = balanced[:, k].argmax(axis=1)
                    balanced_predicted = np.where(available, balanced_winner, observed[fallback[budget]])
                    case['equal_source_mass'] = int((balanced_predicted == truth).sum())
                for method, score in (('raw_margin', raw), ('standardized_margin', standardized)):
                    if d == DEPTHS[0]:
                        case['best'][method] = (score, correct, np.full(len(truth), d))
                    else:
                        previous, prior_correct, selected_depth = case['best'][method]
                        update = score > previous + 1e-12
                        previous[update] = score[update]
                        prior_correct[update] = correct[update]
                        selected_depth[update] = d
        print(f'Finished depth {d}', flush=True)

    results = []
    for (b, condition), case in cases.items():
        methods = {key: {'correct': int(correct.sum()), 'selected_depths': np.bincount(selected, minlength=7)[1:].tolist()}
                   for key, (_, correct, selected) in case['best'].items()}
        results.append(dict(budget=b, condition=condition, fixed=case['fixed'], methods=methods,
                            published_3=case['published_3'], equal_source_mass=case['equal_source_mass'], oracle=int(case['oracle'].sum())))
    data = dict(seed=seed, test_count=len(truth), geometry_sha256=source.sha256(geometry),
                acquisition_sha256=source.sha256(saved_order), human_sha256=source.sha256(human),
                source_sha256=source.sha256(Path(__file__)), rows=results)
    temporary = output.with_suffix('.tmp')
    with temporary.open('x') as stream:
        json.dump(data, stream, indent=2)
    temporary.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('geometry', type=Path)
    parser.add_argument('order', type=Path)
    parser.add_argument('human', type=Path)
    parser.add_argument('seed', type=int)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.geometry, args.order, args.human, args.seed, args.output)
