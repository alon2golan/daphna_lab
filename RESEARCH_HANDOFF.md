# Research Handoff

This repository contains the code and compact artifacts from the active learning project on locality, graph degree, and MaxHerding.

## Final Reports

The latest report versions are in the repository root:

- `Research Summary final 2026-09-08 daphna edits.docx`
- `Research Summary final 2026-09-08 daphna edits.pdf`
- `Research Summary short 2026-09-08 daphna edits.docx`
- `Research Summary short 2026-09-08 daphna edits.pdf`

The editable Markdown sources are:

- `session_notes/research_summary_final_updated_daphna_edits_2026_09_08.md`
- `session_notes/research_summary_short_updated_daphna_edits_2026_09_08.md`

## Main Research Artifacts

Useful compact result tables are under `results_summaries/`.

Useful generated figures are under `visual_outputs/`.

The most relevant final-analysis directories are:

- `results_summaries/cifar100_degree_evidence_2026_06_27/`
- `results_summaries/cifar100_degree_controlled_2026_06_27/`
- `visual_outputs/cifar100_degree_evidence_2026_06_27/`
- `visual_outputs/cifar100_degree_controlled_2026_06_27/`

## Main Code Changes

The core implementation changes are in:

- `deep_al/pycls/al/coverage_matrix_methods.py`
- `deep_al/tools/train_al.py`
- `deep_al/pycls/datasets/tiny_imagenet.py`
- `deep_al/pycls/datasets/utils/features.py`
- `deep_al/pycls/datasets/data.py`

These include local-σ/cutoff support, density normalization schedules, deterministic tie handling for sparse graph cases, and Tiny ImageNet feature-loading fixes.

## Reusable Scripts

Analysis scripts are under `scripts/`.

Reusable cluster scripts are organized under `scripts/sbatch/`.

The old ad-hoc `scripts/sbatch/legacy/` directory was removed during cleanup.

## Deleted During Cleanup

The following were removed because they were generated, stale, or too large for handoff:

- Raw experiment checkpoint directory `output/`
- Slurm/stdout log directory `outputs/`
- Python caches and local IDE metadata
- Duplicate older root-level report copies
- Root-level stray diagnostic PNGs
- Stale checkout `/cs/labs/daphna/ido.dotan/projects/bayes_al_2025`

The compact CSV summaries, figures, final reports, code changes, configs, and reusable scripts were kept.

## Notes For Future Work

The final research direction is to study graph degree and density control more directly than local K alone. The strongest evidence came from CIFAR100, where useful MaxHerding behavior appeared in a low-to-moderate effective-degree regime rather than at the most local or most global settings.
