"""Run the same protected VAST fit after locking the rank permutation."""

import argparse
import math
from pathlib import Path

import numpy as np

from rank import CASES, DEPTHS, fold_ids, load_targets, write_json


def run(archive, targets, output, index):
    from deep_al.vast_noisy.balanced_compare20261120 import worker as balanced
    from deep_al.vast_noisy.factorial20261031 import core
    from deep_al.vast_noisy.submission20261019 import run as source
    from source_support_study.replay import DEPTH, checked_file, checksum, read
    from train_student import validate_fit

    name = CASES[index]
    case = archive / 'cells' / name
    selected_dir = targets / name
    selection = read(selected_dir / 'result.json')
    checked_file(selected_dir / 'oof.npz', selection['oof_sha256'])
    checked_file(Path(__file__).with_name('rank.py'), selection['source_sha256'])
    plan = read(archive / 'plan.json')
    revision, epochs = plan['code']['revision'], plan['reference_epochs']
    original_name = f'A_{DEPTH}.npz'
    original_sha = read(case / 'fixed_targets.json')['hashes'][original_name]
    if (selection['case'] != name or selection['depth'] != DEPTH or selection['chosen'] not in DEPTHS
            or selection['original_sha256'] != original_sha
            or selection['graph_sha256'] != plan['graph']['kernel_sha256']):
        raise ValueError('Study targets do not match the archived protected teacher')
    raw = load_targets(checked_file(case / original_name, original_sha))
    target = load_targets(checked_file(selected_dir / 'targets.npz', selection['revised_sha256']))
    for field in ('admitted', 'confidence'):
        np.testing.assert_array_equal(target[field], raw[field])
    gate = raw['admitted']
    np.testing.assert_array_equal(target['probabilities'][~gate], raw['probabilities'][~gate])
    np.testing.assert_array_equal(np.sort(target['probabilities'][gate], axis=1),
                                  np.sort(raw['probabilities'][gate], axis=1))
    changed = (target['probabilities'] != raw['probabilities']).any(axis=1)
    if changed.sum() != selection['changed_rows'] or (selection['chosen'] == 0 and changed.any()):
        raise ValueError('Applied permutation differs from the recorded decision')

    prepared = case / 'inputs'
    step = read(prepared / 'preparation.json')['rounds'][0]
    report, _, arrays, cfg, _, labeled, unlabeled, validation, observed, val_labels = source._inputs(prepared, step)
    if (report['code']['revision'] != revision or report['config_sha256'] != plan['config_sha256']
            or cfg.OPTIM.MAX_EPOCH != epochs or epochs != 300 or cfg.TRAIN.BATCH_SIZE != 100
            or cfg.TRAIN.EVAL_PERIOD != 2 or cfg.RNG_SEED != 0):
        raise ValueError('The archived student configuration differs')
    with np.load(selected_dir / 'oof.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['labeled'], labeled)
        np.testing.assert_array_equal(saved['observed'], observed)
        np.testing.assert_array_equal(saved['fold'], fold_ids(len(labeled)))
        means = saved['loss'].mean(axis=0)
        np.testing.assert_array_equal(means, [selection['mean_acquired_ce'][str(k)] for k in DEPTHS])
        if DEPTHS[int(means.argmin())] != selection['chosen']:
            raise ValueError('Recorded depth does not minimize acquired-label cross-entropy')
    selected = np.r_[labeled, unlabeled]
    group = core.membership(raw['probabilities'], labeled, observed, unlabeled, selected, gate,
                            gate=True, hybrid=False)
    batches = math.ceil(len(group['indices']) / cfg.TRAIN.BATCH_SIZE)
    steps = epochs * batches
    reference = case / str(DEPTH) / 'A'
    paired = validate_fit(reference, group, steps, epochs, revision)
    if changed.any():
        destination = output / 'rank'
        fit = balanced.fit(destination, prepared, plan, arrays, cfg, target, group, validation,
                           val_labels, steps, 2 * batches)
        runtime_fields = set(fit['training']) - {'best_epoch', 'best_validation_accuracy_percent'}
        reference_training = read(reference / 'result.json')['training']
        for field in runtime_fields:
            if fit['training'][field] != reference_training[field]:
                raise ValueError(f'Student training differed on {field}')
        candidate = validate_fit(destination, group, steps, epochs, revision)
    else:
        output.mkdir(parents=True, exist_ok=True)
        candidate = paired
    write_json(output / 'comparison.json', dict(
        case=name, archive=paired, candidate=candidate, permutation=selection['chosen'],
        candidate_is_exact_reference=not changed.any(),
        delta_pp=candidate['test_accuracy_percent'] - paired['test_accuracy_percent'],
        seed=cfg.RNG_SEED, revision=revision, optimizer_steps=steps,
        epochs=epochs, labeled=len(labeled), admitted=int(gate.sum()), test_count=10000,
        config_sha256=plan['config_sha256'], original_sha256=original_sha,
        target_sha256=selection['revised_sha256'], selection_sha256=checksum(selected_dir / 'result.json'),
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', type=int, choices=range(len(CASES)))
    parser.add_argument('archive', type=Path)
    parser.add_argument('targets', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.archive, args.targets, args.output / CASES[args.index], args.index)


if __name__ == '__main__':
    main()
