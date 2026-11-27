"""Re-evaluate immutable seed-0 acquisition orders with shared held-out heads."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

CONDITIONS = ('clean', 'sym20', 'sym30', 'sym50', 'asym20', 'asym30', 'asym50')
BUDGETS = (400, 800, 1200, 1600, 2000)
ARCHIVED = ('maxherding', 'uherding', 'js', 'observed_10', 'soft_10')
METHODS = ARCHIVED + ('backup_coverage',)
HEADS = ('raw3', 'raw6', 'sqrt3', 'sqrt6', '1nn')
HISTORY_REVISION = '2659c225dd223bdfc623b6d88c1ac993e0316a27'
JS_REVISION = '870f380aa29d6c18fd256955db08ee70ee495531'
READY = 'f8e9ebc64ca3f82189a365c874aa3d613e7d34ee3a55a1651afd9a4588b3b998'
GEOMETRY_SHA256 = '486509610f2947676c2bf5dfc7d3e6950e79805580290f9ca0d54790058e682b'
BACKUP_SHA256 = 'eff8bdb0a55bce77fa66eb6061c815efa5608b60c047500fe119894b252237e4'
SELECTOR_SHA256 = 'e0e8fd4928f586b7e67205992d2c0684859f0b8f6cd548cc4ae5a5afab8ca0d7'
EVALUATOR_SHA256 = 'bd86279047547c9ae30b86bea409a7c26938a2443617d949774358f4e7e14fca'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def local_indices(eligible, rows):
    if rows.ndim != 1 or len(np.unique(rows)) != len(rows):
        raise ValueError('Acquisition rows must be one-dimensional and unique')
    local = np.searchsorted(eligible, rows)
    if np.any(local >= len(eligible)):
        raise ValueError('An archived row is outside the eligible split')
    np.testing.assert_array_equal(eligible[local], rows)
    return local


def ordered_csv(path, eligible, observed, ids, count):
    with path.open(newline='') as stream:
        records = list(csv.DictReader(stream))
    if len(records) != count:
        raise ValueError(f'Expected {count} acquired rows in {path}')
    positions = np.array([int(record['order']) for record in records])
    np.testing.assert_array_equal(positions, np.arange(1, count + 1))
    rows = np.array([int(record['row_index']) for record in records], dtype=np.int64)
    local = local_indices(eligible, rows)
    np.testing.assert_array_equal(observed[rows], [int(record['observed_annotation']) for record in records])
    np.testing.assert_array_equal(ids[rows], [record['sample_id'] for record in records])
    return rows, local


def verify_queries(folder, rows, local, observed, ids):
    for offset in range(0, len(rows), 100):
        prefix = slice(offset, offset + 100)
        with np.load(folder / f'query_{offset // 100 + 1:02d}.npz', allow_pickle=False) as query:
            np.testing.assert_array_equal(query['acquisition_positions'], np.arange(offset + 1, offset + 101))
            np.testing.assert_array_equal(query['indices'], rows[prefix])
            np.testing.assert_array_equal(query['local_indices'], local[prefix])
            np.testing.assert_array_equal(query['previous_local_rows'], local[:offset])
            np.testing.assert_array_equal(query['observed_annotations'], observed[rows[prefix]])
            np.testing.assert_array_equal(query['sample_ids'], ids[rows[prefix]])


def replay(condition, projects, output):
    import evaluate as evaluator

    source = evaluator.source
    if digest(Path(evaluator.__file__)) != EVALUATOR_SHA256:
        raise ValueError('Wrong frozen confirmation evaluator')
    herding = projects / 'cifar100_herding_20261119/study'
    followup = projects / 'cifar100_js_followup_20261120/study'
    geometries = projects / 'cifar_active_audit_20261126/geometry.npz'
    backup_path = projects / 'cifar_exact_backup_20261126/orders.json'
    selector_path = projects / 'cifar_exact_backup_20261126/dense_backup.py'
    if digest(geometries) != GEOMETRY_SHA256 or digest(backup_path) != BACKUP_SHA256:
        raise ValueError('Wrong graph or exact backup acquisition')
    baseline_plan, js_plan = read(herding / 'plan.json'), read(followup / 'plan.json')
    for revision, plan in ((HISTORY_REVISION, baseline_plan), (JS_REVISION, js_plan)):
        if (plan['revision'] != revision or plan['geometry_ready'] != READY or plan['seed'] != 0
                or plan['budgets'] != list(range(100, 2001, 100))):
            raise ValueError('Wrong historical replay protocol')
    geom_manifest = Path(baseline_plan['geometry']) / 'geometry.json'
    if digest(geom_manifest) != READY or js_plan['geometry'] != baseline_plan['geometry']:
        raise ValueError('The archival acquisition geometry changed')
    backup = read(backup_path)
    if (backup['selector_sha256'] != SELECTOR_SHA256
            or backup['geometry_sha256'] != read(geom_manifest)['files']['rbf']['sha256']
            or backup['common_prefix'] != 100 or backup['backup_weight'] != .5):
        raise ValueError('Unexpected backup coverage source')
    if selector_path.is_file() and digest(selector_path) != SELECTOR_SHA256:
        raise ValueError('Archival backup selector bytes changed')
    with np.load(geometries, allow_pickle=False) as graph:
        eligible, neighbors, heldout = (graph[name] for name in ('rows', 'neighbors', 'heldout_neighbors'))
        graph_meta = json.loads(graph['metadata'].item())
    np.testing.assert_array_equal(eligible, np.load(herding / 'inputs/eligible.npy', allow_pickle=False))
    validation = np.load(herding / 'inputs/partitions/validation_indices.npy', allow_pickle=False)
    np.testing.assert_array_equal(np.sort(np.r_[eligible, validation]), np.arange(50000))
    if (neighbors.shape != (45000, 13) or heldout.shape != (10000, 13)
            or graph_meta['bank_sha256'] != source.BANK_HASHES):
        raise ValueError('Unexpected graph or bank provenance')
    for name, sha in source.BANK_HASHES.items():
        if digest(source.BANK / f'{name}.npy') != sha:
            raise ValueError(f'The BYOL bank changed at {name}')
    observed_path = herding / 'inputs' / f'{condition}_observed.npy'
    observed_sha = digest(observed_path)
    if (observed_sha != baseline_plan['noises'][condition]['observed_sha256']
            or observed_sha != digest(followup / 'inputs' / observed_path.name)):
        raise ValueError('The shared annotation vector changed')
    observed = np.load(observed_path, allow_pickle=False)
    canonical = np.load(source.BANK / 'train_targets.npy', allow_pickle=False)
    truth = np.load(source.BANK / 'test_targets.npy', allow_pickle=False)
    ids_info = read(herding / 'inputs/provenance.json')
    ids_path = Path(ids_info['files']['train.sample_ids'])
    if digest(ids_path) != ids_info['sha256']['train.sample_ids']:
        raise ValueError('Canonical train IDs changed')
    ids = np.load(ids_path, allow_pickle=False)
    if observed.shape != canonical.shape or truth.shape != (10000,):
        raise ValueError('Unexpected train or test label shape')
    train = np.load(source.BANK / 'train_features.npy', mmap_mode='r', allow_pickle=False)
    test = source.unit_rows(np.load(source.BANK / 'test_features.npy', allow_pickle=False))
    walk, predict = evaluator.walk_matrix(neighbors, len(eligible)), evaluator.walk_matrix(heldout, len(eligible))
    result, chosen, order_files = [], {}, {}
    for method in METHODS:
        if method == 'backup_coverage':
            rows = np.asarray(backup[method], dtype=np.int64)
            if len(rows) != max(BUDGETS):
                raise ValueError('Unexpected backup order length')
            order, path = local_indices(eligible, rows), backup_path
        else:
            archive = followup if method in ('observed_10', 'soft_10') else herding
            folder = archive / 'trajectories' / f'{condition}_{method}'
            path = folder / 'acquisitions.csv'
            rows, order = ordered_csv(path, eligible, observed, ids, max(BUDGETS))
            verify_queries(folder, rows, order, observed, ids)
        np.testing.assert_array_equal(rows[:100], backup['maxherding'][:100])
        if method == 'maxherding':
            np.testing.assert_array_equal(rows, backup['maxherding'])
        chosen[method], order_files[method] = rows.tolist(), {'path': str(path), 'sha256': digest(path)}
        records = evaluator.evaluate_order(train, test, {condition: observed}, canonical, truth,
                                           BUDGETS, order, eligible, walk, predict, method)
        for record in records:
            budget = record['budget']
            record['correct'] = {head: record['correct'][head] for head in HEADS}
            if method != 'backup_coverage':
                historic = read(folder / f'budget_{budget:04d}/metrics.json')
                record['archived_1nn_correct'] = round(historic['accuracy_1nn_percent'] * len(truth) / 100)
            result.append(record)
    artifact = dict(condition=condition, seed=0, test_count=len(truth), rows=result, acquisitions=chosen,
                    orders=order_files, observed_sha256=observed_sha, geometry_sha256=GEOMETRY_SHA256,
                    historical_revision=HISTORY_REVISION, js_revision=JS_REVISION, backup_sha256=BACKUP_SHA256,
                    evaluator_sha256=EVALUATOR_SHA256, helper_sha256=digest(Path(source.__file__)),
                    source_sha256=digest(Path(__file__)), numpy=np.__version__)
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_suffix('.tmp')
    with temporary.open('x') as stream:
        json.dump(artifact, stream, sort_keys=True)
    temporary.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('condition_index', type=int, choices=range(len(CONDITIONS)))
    parser.add_argument('projects', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    replay(CONDITIONS[args.condition_index], args.projects, args.output)


if __name__ == '__main__':
    main()
