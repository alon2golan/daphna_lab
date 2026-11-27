"""Matched CIFAR-100 sweeps with Netta's three NAS acquisition rules."""

import argparse
import csv
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np

CONDITIONS = ('clean', 'sym20', 'sym30', 'sym50', 'asym20', 'asym30', 'asym50', 'cifar100n')
METHODS = ('probcover', 'npc', 'weighted_npc', 'maxherding_nas', 'maxherding_reference')
BUDGETS = (400, 800, 1200, 1600, 2000)
BATCH = 100
CLASSES = 100
PROJECTS = Path('/cs/labs/daphna/tomn/projects')
HERDING = PROJECTS / 'cifar100_herding_20261119/study'
TEST_GRAPH = PROJECTS / 'cifar_active_audit_20261126/geometry.npz'
TEST_GRAPH_SHA256 = '486509610f2947676c2bf5dfc7d3e6950e79805580290f9ca0d54790058e682b'
DETECTOR_SHA256 = '2eaa5f51747396250e29e459afbc45a84a349eb2e630a18acb5db325e6b5af58'
GAUSSIAN_SHA256 = '31f8c13336b52bd31cca6c21abf9dc4ecc39d13283243d4dc6581a8206fffb4f'
NETTA = '733651bb0943a96392895103eb2ffc4e99a7c7d1'


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def checked(path, expected):
    if sha256(path) != expected:
        raise ValueError(f'Checksum changed: {path}')
    return path


def radius_demand(graph, labeled, noisy, weighted):
    """Authors' fixed-radius rule. Clean coverage wins noisy/clean overlaps."""
    weight = np.ones(graph.shape[0])
    if weighted and noisy.any():
        weight[graph[labeled[noisy]].indices] = 1 - noisy.mean()
    weight[graph[labeled[~noisy]].indices] = 0
    return weight


def radius_batch(graph, incoming, labeled, noisy, weighted, rng, batch_size=BATCH):
    weights = radius_demand(graph, labeled, noisy, weighted)
    gain = graph @ weights
    gain[labeled] = -np.inf
    chosen = []
    for _ in range(batch_size):
        # Ignore summation-order noise when multiple candidates have equal degree.
        ties = np.flatnonzero(np.isclose(gain, gain.max(), rtol=0, atol=1e-10))
        row = int(rng.choice(ties))
        chosen.append(row)
        gain[row] = -np.inf
        targets = graph.indices[graph.indptr[row]:graph.indptr[row + 1]]
        targets = targets[weights[targets] > 0]
        for col, weight in zip(targets, weights[targets]):
            left, right = incoming.indptr[col:col + 2]
            gain[incoming.indices[left:right]] -= weight
        weights[targets] = 0
    return np.asarray(chosen)


def gaussian_batch(matrix, labeled, noisy, batch_size=BATCH, block_size=128):
    """Exact lazy MaxHerding gain, with only predicted-clean prior anchors covering."""
    n = len(matrix)
    best = np.zeros(n, dtype=np.float64)
    allowed = np.ones(n, dtype=bool)
    allowed[labeled] = False
    for row in labeled[~noisy]:
        np.maximum(best, matrix[row], out=best)

    def gains(values):
        score = np.subtract(values, best, dtype=np.float64)
        np.maximum(score, 0, out=score)
        return score.sum(axis=-1)

    heap = []
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        inside = np.flatnonzero(allowed[start:stop])
        if len(inside):
            score = gains(matrix[start:stop])
            heap.extend((-float(score[j]), j + start, -1) for j in inside)
    heapq.heapify(heap)
    chosen = []
    for iteration in range(batch_size):
        while True:
            _, row, stamp = heapq.heappop(heap)
            if stamp == iteration:
                break
            heapq.heappush(heap, (-float(gains(matrix[row])), row, iteration))
        chosen.append(row)
        np.maximum(best, matrix[row], out=best)
    return np.asarray(chosen)


def tests():
    from scipy.sparse import csr_matrix

    matrix = np.array([[1, 0, 1, 0, 1], [0, 1, 1, 1, 0], [1, 1, 1, 0, 0],
                       [0, 1, 0, 1, 1], [1, 0, 0, 1, 1]], dtype=float)
    graph = csr_matrix(matrix)
    labeled = np.array([0, 1])
    noisy = np.array([False, True])
    np.testing.assert_array_equal(radius_demand(graph, labeled, noisy, False), [0, 1, 0, 1, 0])
    np.testing.assert_array_equal(radius_demand(graph, labeled, noisy, True), [0, .5, 0, .5, 0])
    for weighted in (False, True):
        seed = 31
        actual = radius_batch(graph, graph.tocsc(), labeled, noisy, weighted,
                              np.random.RandomState(seed), batch_size=3)
        demand = radius_demand(graph, labeled, noisy, weighted)
        reference = []
        rng = np.random.RandomState(seed)
        for _ in range(3):
            gain = matrix @ demand
            gain[np.r_[labeled, reference].astype(int)] = -np.inf
            chosen = int(rng.choice(np.flatnonzero(gain == gain.max())))
            reference.append(chosen)
            demand[matrix[chosen] > 0] = 0
        np.testing.assert_array_equal(actual, reference)
    rng = np.random.default_rng(918)
    similarities = rng.uniform(.001, .99, (19, 19))
    similarities = (similarities + similarities.T) / 2
    np.fill_diagonal(similarities, 1)
    from_naive = []
    best = similarities[0].copy()
    for _ in range(7):
        gain = np.maximum(similarities - best, 0).sum(axis=1)
        gain[np.r_[labeled, from_naive].astype(int)] = -np.inf
        index = int(gain.argmax())
        from_naive.append(index)
        np.maximum(best, similarities[index], out=best)
    np.testing.assert_array_equal(gaussian_batch(similarities, labeled, noisy, 7), from_naive)


def inputs(condition):
    import active_audit as protocol
    import scipy.sparse as sparse

    plan = load_json(HERDING / 'plan.json')
    if (plan['eligible_count'] != 45000 or plan['test_count'] != 10000
            or plan['seed'] != 0 or plan['batch_size'] != BATCH):
        raise ValueError('Unexpected herding protocol')
    population = np.load(checked(HERDING / 'inputs/eligible.npy', plan['input_sha256']['eligible.npy']), allow_pickle=False)
    folder = Path(plan['geometry'])
    manifest = load_json(checked(folder / 'geometry.json', plan['geometry_ready']))
    if (folder / 'READY').read_text().strip() != plan['geometry_ready']:
        raise ValueError('Archived geometry is not ready')
    radius_meta = manifest['files']['probcover']
    graph = sparse.load_npz(checked(folder / radius_meta['file'], radius_meta['sha256'])).tocsr()
    graph.sort_indices()
    rbf_meta = manifest['files']['rbf']
    rbf = np.load(folder / rbf_meta['file'], mmap_mode='r', allow_pickle=False)
    if (graph.shape != (len(population),) * 2 or rbf.shape != graph.shape
            or rbf_meta['sha256'] != GAUSSIAN_SHA256 or not np.all(graph.data == 1)
            or not np.all(graph.diagonal() == 1)):
        raise ValueError('Coverage geometry differs')
    track = HERDING / 'trajectories/clean_maxherding/acquisitions.csv'
    with track.open(newline='') as stream:
        baseline_global = np.asarray([int(row['row_index']) for row in csv.DictReader(stream)])
    baseline = np.searchsorted(population, baseline_global)
    np.testing.assert_array_equal(population[baseline], baseline_global)
    if len(np.unique(baseline)) != 2000:
        raise ValueError('Invalid reference order')
    with np.load(checked(TEST_GRAPH, TEST_GRAPH_SHA256), allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['rows'], population)
        neighbors, heldout = archive['neighbors'], archive['heldout_neighbors']
        graph_meta = json.loads(archive['metadata'].item())
    if graph_meta['bank_sha256'] != protocol.BANK_HASHES:
        raise ValueError('Test graph and BYOL bank differ')
    if condition == 'cifar100n':
        clean = np.load(checked(protocol.BANK / 'train_targets.npy', protocol.BANK_HASHES['train_targets']), allow_pickle=False)
        pool = protocol.human_annotations(clean)
        noise_hash = protocol.HUMAN_SHA256
    else:
        item = plan['noises'][condition]
        pool = np.load(checked(Path(item['observed_path']), item['observed_sha256']), allow_pickle=False)
        noise_hash = item['observed_sha256']
    return population, neighbors, heldout, graph, rbf, baseline, pool, dict(
        archive_revision=plan['revision'], geometry_manifest_sha256=plan['geometry_ready'],
        radius_graph_sha256=radius_meta['sha256'], rbf_sha256=rbf_meta['sha256'],
        head_graph_sha256=TEST_GRAPH_SHA256, warm_order_sha256=sha256(track), observed_sha256=noise_hash)


def trajectory(method, features, population, pool, graph, rbf, baseline, detector):
    order = baseline[:BATCH].copy() if method != 'maxherding_reference' else baseline.copy()
    known = np.full(len(features), -1, dtype=int)
    known[order] = pool[population[order]]
    incoming = graph.tocsc()
    rng = np.random.RandomState(0)
    masks = {}
    for budget in range(BATCH, max(BUDGETS) + BATCH, BATCH):
        current = order[:budget]
        candidates = np.setdiff1d(np.arange(len(features)), current)
        adapt = method in ('npc', 'weighted_npc', 'maxherding_nas')
        if adapt or budget in BUDGETS:
            noisy, stats = detector(features, known, current, candidates, num_classes=CLASSES,
                                    seed=0, device='cpu')
            if len(noisy) != budget or not np.isfinite(stats.get('threshold', 0)):
                raise ValueError('Invalid LowBudgetAUM result')
        if budget in BUDGETS:
            masks[budget] = dict(noisy_rows=population[current[noisy]].tolist(), info=stats)
        if budget == max(BUDGETS) or method == 'maxherding_reference':
            continue
        if method == 'maxherding_nas':
            query = gaussian_batch(rbf, current, noisy)
        else:
            if not adapt:
                noisy = np.zeros(budget, dtype=bool)
            query = radius_batch(graph, incoming, current, noisy, method == 'weighted_npc', rng)
        if len(np.unique(query)) != BATCH or not np.isin(query, candidates).all():
            raise ValueError('Invalid acquisition batch')
        known[query] = pool[population[query]]
        order = np.r_[order, query]
        print(f'{method} acquired {len(order)}', flush=True)
    return dict(rows=population[order].tolist(), labels=known[order].tolist(), masks=masks)


def path_matrix(neighbors, width):
    from scipy.sparse import csr_matrix

    n, k = neighbors.shape
    return csr_matrix((np.ones(n * k, dtype=np.float32), neighbors.ravel(), np.arange(n + 1) * k), shape=(n, width))


def evaluate(artifact, population, neighbors, heldout, features, test_features, truth):
    from active_audit import unit_rows

    positions = np.searchsorted(population, artifact['rows'])
    labels = np.asarray(artifact['labels'])
    basis = np.zeros((len(population), len(positions)), dtype=np.float32)
    basis[positions, np.arange(len(positions))] = 1
    walk = path_matrix(neighbors, len(population))
    project = path_matrix(heldout, len(population))
    outputs = {}
    for depth in range(1, 7):
        if depth in (3, 6):
            outputs[f'raw{depth}'] = project @ basis
        if depth < 6:
            basis = walk @ basis
    cosine = test_features @ unit_rows(features[positions]).T
    eye = np.eye(CLASSES, dtype=np.float32)
    records = []
    for budget in BUDGETS:
        kept = ~np.isin(np.asarray(artifact['rows'][:budget]), artifact['masks'][budget]['noisy_rows'])
        for training_filter, used in (('all', np.ones(budget, dtype=bool)), ('aum', kept)):
            scores = {}
            if used.any():
                class_vectors = eye[labels[:budget][used]]
                for metric, counts in outputs.items():
                    pred = (counts[:, :budget][:, used] @ class_vectors).argmax(axis=1)
                    scores[metric] = int((pred == truth).sum())
                nearest = cosine[:, :budget][:, used].argmax(axis=1)
                scores['1nn'] = int((labels[:budget][used][nearest] == truth).sum())
            records.append(dict(budget=budget, training_filter=training_filter,
                                retained=int(used.sum()), test_count=len(truth), correct=scores))
    return records


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
    tmp.rename(path)


def run(condition, output):
    import active_audit as protocol
    from deep_al.pycls.al import lowbudget_aum
    import torch

    detector = lowbudget_aum.lowbudget_aum_noisy_mask
    checked(Path(lowbudget_aum.__file__), DETECTOR_SHA256)
    torch.set_num_threads(2)
    population, neighbors, heldout, graph, rbf, baseline, pool, provenance = inputs(condition)
    output.mkdir(exist_ok=True)
    features = np.load(checked(protocol.BANK / 'train_features.npy', protocol.BANK_HASHES['train_features']),
                       mmap_mode='r')[population]
    test_x = protocol.unit_rows(np.load(checked(protocol.BANK / 'test_features.npy', protocol.BANK_HASHES['test_features']), allow_pickle=False))
    truth = np.load(checked(protocol.BANK / 'test_targets.npy', protocol.BANK_HASHES['test_targets']), allow_pickle=False)
    provenance.update(detector_sha256=DETECTOR_SHA256, netta_revision=NETTA, source_sha256=sha256(__file__),
                      torch=torch.__version__, numpy=np.__version__, seed=0, radius=.5,
                      detector_device='cpu', query_noise_dropout=False)
    for method in METHODS:
        path = output / f'{method}.json'
        if path.is_file():
            if load_json(path)['provenance'] != provenance:
                raise ValueError('Refusing to reuse unrelated outputs')
            continue
        artifact = trajectory(method, features, population, pool, graph, rbf, baseline, detector)
        if (len(set(artifact['rows'])) != max(BUDGETS)
                or artifact['rows'][:BATCH] != population[baseline[:BATCH]].tolist()):
            raise ValueError('Completed trajectory failed budget or warm-start check')
        artifact['scores'] = evaluate(artifact, population, neighbors, heldout, features, test_x, truth)
        artifact.update(method=method, condition=condition, provenance=provenance)
        write_json(path, artifact)
        print(f'{condition} {method} verified', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=CONDITIONS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--parity', action='store_true')
    args = parser.parse_args()
    tests()
    if args.parity:
        _, _, _, _, rbf, baseline, _, _ = inputs('clean')
        np.testing.assert_array_equal(gaussian_batch(rbf, baseline[:BATCH], np.zeros(BATCH, dtype=bool)),
                                      baseline[BATCH:2*BATCH])
        print('Archived Gaussian second-batch parity passed', flush=True)
    elif args.condition:
        run(args.condition, args.output)


if __name__ == '__main__':
    main()
