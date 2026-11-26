"""Screen per-annotation square roots on saved protected VAST posteriors."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

from source_support_study.replay import DEPTH, checked_file, checksum, read, teacher

CASES = tuple(product((800, 1200), ('sym0', 'sym30', 'sym50', 'asym50')))


def compress(graph, labeled, observed, probabilities, gates, block_size=32):
    """Preserve each row's excess mass while changing how sources split it."""
    rows, classes = probabilities.shape
    raw = np.zeros((rows, classes), dtype=np.float64)
    sqrt = np.zeros_like(raw)
    for start in range(0, len(labeled), block_size):
        anchors = labeled[start:start + block_size]
        labels = observed[start:start + block_size]
        direct = np.ascontiguousarray(graph[anchors].T.toarray())
        current = direct.copy()
        for send, receive in gates:
            update = direct + graph @ (current * send[:, None])
            current[receive] = update[receive]
        for col, cls in enumerate(labels):
            raw[:, cls] += current[:, col]
            sqrt[:, cls] += np.sqrt(current[:, col])
    total = raw.sum(axis=1)
    sqrt_total = sqrt.sum(axis=1)
    unprotected = np.ones(rows, dtype=bool)
    unprotected[labeled] = False
    prior = 1 / classes
    expected = (raw + prior) / (1 + total[:, None])
    error = float(np.max(np.abs(expected[unprotected] - probabilities[unprotected])))
    if not np.isfinite(error) or error > 3e-6:
        raise ValueError(f'Per-source decomposition failed to reproduce archive ({error:g})')
    scale = np.divide(total, sqrt_total, out=np.zeros_like(total), where=sqrt_total > 0)
    result = (sqrt * scale[:, None] + prior) / (1 + total[:, None])
    result[labeled] = probabilities[labeled]
    return result.astype(probabilities.dtype), error


def screen(archive, output, budget, condition):
    plan = read(archive / 'plan.json')
    cell = archive / 'cells' / f'b{budget}_{condition}'
    if not (cell / 'verified.json').is_file():
        raise ValueError('Expected completed fixed-depth cell')
    prepared = cell / 'inputs'
    provenance = read(prepared / 'provenance.json')
    prep = read(prepared / 'preparation.json')['prepared_sha256']

    def inp(name):
        return np.load(checked_file(prepared / name, prep[name]), allow_pickle=False)

    labeled = inp('partitions/labeled_indices.npy')
    unlabeled = inp('partitions/unlabeled_indices.npy')
    observed = inp('observed_labeled_labels.npy')
    graph_info = plan['graph']
    graph = load_npz(checked_file(Path(graph_info['path']), graph_info['kernel_sha256'])).tocsr()
    name = f'A_{DEPTH}.npz'
    target = checked_file(cell / name, read(cell / 'fixed_targets.json')['hashes'][name])
    with np.load(target, allow_pickle=False) as saved:
        probabilities, admitted, confidence = (saved[k] for k in ('probabilities', 'admitted', 'confidence'))
    evidence, gates = teacher(graph, np.load(prepared / 'initial_evidence.npy', allow_pickle=False),
                              labeled, observed)
    rebuilt = evidence / evidence.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(rebuilt, probabilities, atol=3e-6, rtol=0)
    revised, error = compress(graph, labeled, observed, probabilities, gates)
    truth = np.load(checked_file(Path(provenance['files']['train.labels']),
                                 plan['features']['train.labels']), allow_pickle=False)
    baseline, prediction = probabilities.argmax(axis=1), revised.argmax(axis=1)
    if not np.array_equal(np.flatnonzero(admitted), np.intersect1d(unlabeled, np.flatnonzero(admitted))):
        raise ValueError('Unexpected admitted split')
    pool = {}
    for label, ids in (('unlabeled', unlabeled), ('admitted', np.flatnonzero(admitted))):
        pool[label] = dict(samples=len(ids), raw=int((baseline[ids] == truth[ids]).sum()),
                           sqrt=int((prediction[ids] == truth[ids]).sum()),
                           changed=int((prediction[ids] != baseline[ids]).sum()))
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'targets.npz', probabilities=revised, admitted=admitted, confidence=confidence)
    details = dict(case=cell.name, budget=budget, condition=condition, depth=DEPTH, seed=0,
                   graph_sha256=graph_info['kernel_sha256'], original_sha256=checksum(target),
                   revised_sha256=checksum(output / 'targets.npz'), reconstruction_error=error, rows=pool)
    (output / 'result.json').write_text(json.dumps(details, indent=2, allow_nan=False) + '\n')
    print(json.dumps(details), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', type=int, choices=range(len(CASES)))
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    budget, condition = CASES[args.index]
    screen(args.archive, args.output / f'b{budget}_{condition}', budget, condition)
