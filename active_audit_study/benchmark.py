"""Verify signed audits over a wider CIFAR-100 grid and an archived MaxHerding order."""

import argparse
import json
from pathlib import Path

from active_audit import replay, sha256
from collect import POLICIES, collect, write_csv

HERE = Path(__file__).resolve().parent
ARCHIVE_ORDER = HERE / 'maxherding_rows.json'
SYNTHETIC = tuple((f'{kind}{rate}', rate / 100) for kind in ('sym', 'asym') for rate in (30, 40, 50))
COVERAGE = (('clean', 0),) + SYNTHETIC + (('cifar100n', .402),)
GRID = {
    'coverage': (COVERAGE, tuple(range(10, 20)), (400, 600, 800, 1200)),
    'maxherding': (SYNTHETIC, tuple(range(20, 25)), (800, 1200)),
}
METHODS = POLICIES + ('signed_peer',)


def archive_order():
    return json.loads(ARCHIVE_ORDER.read_text())['rows']


def run(seed, geometry, output):
    cohort = 'coverage' if seed in GRID['coverage'][1] else 'maxherding'
    conditions, _, budgets = GRID[cohort]
    acquired = archive_order() if cohort == 'maxherding' else None
    replay(seed, geometry, output, signed=True, budgets=budgets, conditions=conditions,
           acquired_rows=acquired)


def export(root):
    rows = []
    manifest = {}
    for cohort, (conditions, seeds, budgets) in GRID.items():
        acquired = archive_order() if cohort == 'maxherding' else None
        indexed = collect(root, conditions, seeds, budgets, METHODS, acquired_rows=acquired)
        for _, result in sorted(indexed.items()):
            rows.append({'acquisition': cohort, **{k: v for k, v in result.items() if k != 'request_rows'}})
        raw = json.loads((root / f'seed_{seeds[0]}.json').read_text())
        manifest[cohort] = {k: raw[k] for k in (
            'geometry_sha256', 'source_sha256', 'provenance', 'depth', 'numpy',
        )}
        manifest[cohort].update(seeds=seeds, budgets=budgets, evaluations=len(indexed))
    if len({entry['geometry_sha256'] for entry in manifest.values()}) != 1:
        raise ValueError('Verification grids used different graph artifacts')
    manifest['maxherding']['order_sha256'] = sha256(ARCHIVE_ORDER)
    write_csv(root / 'benchmark_metrics.csv', rows)
    (root / 'benchmark_provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Verified {len(rows)} matched policy evaluations')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    replay_parser = actions.add_parser('run')
    replay_parser.add_argument('seed', type=int, choices=range(10, 25))
    replay_parser.add_argument('geometry', type=Path)
    replay_parser.add_argument('output', type=Path)
    export_parser = actions.add_parser('collect')
    export_parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    if args.action == 'run':
        run(args.seed, args.geometry, args.output)
    else:
        export(args.directory)


if __name__ == '__main__':
    main()
