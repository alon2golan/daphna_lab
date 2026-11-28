"""Verify and tabulate matched LowBudgetAUM, peer-only and fused fits."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from collect import paired_rows as verified_fusion, read, save_csv
from run_official import BUDGETS, CONDITIONS, SEEDS, file_sha256


def paired_rows(reference, replay):
    fusion = {(row['condition'], row['seed'], row['budget']): row for row in verified_fusion(reference)}
    planned = len(CONDITIONS) * len(SEEDS) * len(BUDGETS)
    if len(fusion) != planned:
        raise ValueError('The original NPC/fusion grid is incomplete')
    candidate_hash = file_sha256(Path(__file__).with_name('peer_only.py'))
    for condition in CONDITIONS:
        for seed in SEEDS:
            case = f'{condition}_s{seed}'
            for budget in BUDGETS:
                snapshot = reference / 'runs' / case / 'fits' / f'{budget:05d}'
                fit_dir = replay / 'peer_only' / case / f'{budget:05d}'
                previous = fusion[condition, seed, budget]
                base = read(snapshot / 'baseline_result.json')
                candidate = read(fit_dir / 'result.json')
                identity = read(fit_dir / 'identity_result.json')
                input_path = snapshot / 'fit_inputs.npz'
                digest = file_sha256(input_path)
                if (candidate['condition'], candidate['seed'], candidate['budget']) != (condition, seed, budget):
                    raise ValueError(f'Wrong peer-only fit in {fit_dir}')
                if (candidate['mask_function_sha256'] != candidate_hash
                        or candidate['fit_inputs_sha256'] != digest or base['fit_inputs_sha256'] != digest
                        or candidate['kept'] != base['kept'] or previous['retained'] != base['kept']
                        or identity['fit_inputs_sha256'] != digest):
                    raise ValueError(f'Wrong snapshot, retained count, or mask source in {fit_dir}')
                for field in ('prediction_sha256', 'fitted_model_sha256'):
                    if identity[field] != base[field] or candidate[f'baseline_{field}'] != base[field]:
                        raise ValueError(f'Identity differs from the original {field} in {fit_dir}')
                if identity['accuracy'] != base['accuracy'] or base['accuracy'] != previous['base_accuracy_pct']:
                    raise ValueError(f'Identity test accuracy differs in {fit_dir}')
                if candidate['baseline_retained_clean'] != previous['base_clean_retained']:
                    raise ValueError(f'The original clean retention differs in {fit_dir}')
                with np.load(input_path, allow_pickle=False) as saved:
                    observed, original = saved['observed_labels'], saved['base_keep']
                keep = np.load(fit_dir / 'keep.npy', allow_pickle=False)
                if (keep.shape != (budget,) or keep.dtype != np.bool_
                        or (original & ~keep).sum() != candidate['changed_pairs']):
                    raise ValueError(f'Peer-only membership differs from its result in {fit_dir}')
                np.testing.assert_array_equal(np.bincount(observed[original], minlength=100),
                                              np.bincount(observed[keep], minlength=100))
                accuracies = [base['accuracy'], candidate['accuracy'], previous['peer_accuracy_pct']]
                if not all(0 <= value <= 100 and math.isfinite(value) for value in accuracies):
                    raise ValueError(f'Invalid held-out test score in {fit_dir}')
                base_acc, peer_acc, fusion_acc = accuracies
                yield dict(condition=condition, budget=budget, seed=seed, retained=base['kept'],
                           base_accuracy_pct=base_acc, peer_only_accuracy_pct=peer_acc,
                           fusion_accuracy_pct=fusion_acc,
                           peer_only_minus_base_pp=round(peer_acc - base_acc, 8),
                           fusion_minus_peer_only_pp=round(fusion_acc - peer_acc, 8),
                           fusion_minus_base_pp=round(fusion_acc - base_acc, 8),
                           base_clean_retained=previous['base_clean_retained'],
                           peer_only_clean_retained=candidate['retained_clean'],
                           fusion_clean_retained=previous['peer_clean_retained'],
                           peer_only_replaced=candidate['changed_pairs'], fusion_replaced=previous['replaced'])


ACCURACIES = ('base_accuracy_pct', 'peer_only_accuracy_pct', 'fusion_accuracy_pct')
DIFFERENCES = ('peer_only_minus_base_pp', 'fusion_minus_peer_only_pp', 'fusion_minus_base_pp')


def setting_rows(pairs):
    for condition in CONDITIONS:
        for budget in BUDGETS:
            rows = [row for row in pairs if (row['condition'], row['budget']) == (condition, budget)]
            if len(rows) != len(SEEDS):
                raise ValueError(f'Missing seeds for {condition} at {budget}')
            yield dict(condition=condition, budget=budget, seeds=len(rows),
                       **{key: mean(row[key] for row in rows) for key in (*ACCURACIES, *DIFFERENCES)},
                       **{f'{key}_winning_seeds': sum(row[key] > 0 for row in rows) for key in DIFFERENCES})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'replay', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    pairs = list(paired_rows(args.reference, args.replay))
    settings = list(setting_rows(pairs))
    args.output.mkdir(parents=True, exist_ok=False)
    save_csv(args.output / 'per_seed.csv', pairs)
    save_csv(args.output / 'by_setting.csv', settings)
    groups = dict(all=settings, clean=[row for row in settings if row['condition'] == 'clean'],
                  symmetric=[row for row in settings if row['condition'].startswith('sym')],
                  cyclic=[row for row in settings if row['condition'].startswith('asym')])
    summary = {group: dict(settings=len(rows), **{
        key: dict(mean=mean(row[key] for row in rows), wins=sum(row[key] > 1e-8 for row in rows))
        for key in DIFFERENCES}) for group, rows in groups.items()}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(pairs=len(pairs), settings=len(settings), summary=summary), allow_nan=False))


if __name__ == '__main__':
    main()
