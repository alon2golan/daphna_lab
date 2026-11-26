"""Exact expected backup coverage on the archived MaxHerding similarities."""

import argparse
import heapq
import hashlib
import json
from pathlib import Path

import numpy as np


def gain(rows, best, second, backup):
    first_gain = np.maximum(np.subtract(rows, best, dtype=np.float64), 0)
    if backup:
        second_gain = np.maximum(np.subtract(rows, second, dtype=np.float64), 0)
        first_gain *= 1 - backup
        first_gain += backup * second_gain
    return first_gain.sum(axis=-1)


def greedy(matrix, warm, count, backup, progress=False):
    """Use decreasing marginal gains as lazy bounds, breaking ties by row ID."""
    n = len(matrix)
    warm = np.asarray(warm, dtype=int)
    best = np.zeros(n)
    second = np.zeros(n)

    def update(row):
        np.maximum(second, np.minimum(best, row), out=second)
        np.maximum(best, row, out=best)

    for i in warm:
        update(matrix[i])
    allowed = np.ones(n, dtype=bool)
    allowed[warm] = False
    heap = []
    for start in range(0, n, 128):
        stop = min(start + 128, n)
        values = np.asarray(matrix[start:stop])
        scores = gain(values, best, second, backup)
        heap.extend((-float(scores[j]), j + start, -1) for j in np.flatnonzero(allowed[start:stop]))
    heapq.heapify(heap)
    chosen = []
    for iteration in range(count):
        while True:
            negative, row, stamp = heapq.heappop(heap)
            if stamp == iteration:
                break
            value = gain(matrix[row], best, second, backup)
            heapq.heappush(heap, (-float(value), row, iteration))
        chosen.append(row)
        update(matrix[row])
        if progress and (iteration + 1) % 200 == 0:
            print(f'Committed {len(warm) + iteration + 1} picks', flush=True)
    return np.array(chosen)


def run(reference, order_file, output):
    plan = json.loads((reference / 'plan.json').read_text())
    geometry = Path(plan['geometry'])
    manifest_path = geometry / 'geometry.json'
    manifest = json.loads(manifest_path.read_text())
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_sha256 != plan['geometry_ready'] or manifest_sha256 != (geometry / 'READY').read_text().strip():
        raise ValueError('Coverage geometry manifest changed')
    artifact = manifest['files']['rbf']
    path = geometry / artifact['file']
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != artifact['sha256']:
            raise ValueError('Archived similarity bytes changed')
    print('Verified archive matrix', flush=True)
    matrix = np.load(path, mmap_mode='r', allow_pickle=False)
    rows = np.load(reference / 'inputs/eligible.npy', allow_pickle=False)
    archived = np.asarray(json.loads(order_file.read_text())['rows'])
    if matrix.shape != (len(rows), len(rows)) or matrix.dtype != np.float32 or len(archived) != 2000:
        raise ValueError('Unexpected coverage inputs')
    local = np.searchsorted(rows, archived)
    np.testing.assert_array_equal(rows[local], archived)
    warm = local[:100]
    np.testing.assert_array_equal(greedy(matrix, warm, 100, 0), local[100:200])
    print('Zero-backup parity matched the archived second batch', flush=True)
    backup = greedy(matrix, warm, len(archived) - len(warm), .5, progress=True)
    alternate = rows[np.r_[warm, backup]]
    assert len(np.unique(alternate)) == len(archived)
    result = dict(maxherding=archived.tolist(), backup_coverage=alternate.tolist(), common_prefix=len(warm),
                  backup_weight=.5, geometry_sha256=artifact['sha256'],
                  baseline_parity_count=100, eligible_count=len(rows),
                  selector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with output.open('x') as stream:
        json.dump(result, stream)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('order', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.reference, args.order, args.output)
