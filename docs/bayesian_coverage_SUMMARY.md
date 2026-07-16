---
title: "Bayesian Coverage — summary (SPLIT 2026-07-12)"
date: 2026-07-12
---

This file was **split into three documents on 2026-07-12** (section numbering preserved):

| File | Contents (original §) |
|---|---|
| `bayesian_coverage_THEORY.md` | §1 probabilistic model, §2 `TRAIN_LABEL_MODE` training plane, §3 NPC/AUM baseline |
| `bayesian_coverage_RESULTS.md` | §0 TL;DR + recommended config, §4 result tables, §4.5 posterior replay, §4.6 pseudo-labeling, §5 takeaways, §6 reproduction |
| `bayesian_coverage_NEXT_STEPS.md` | status board, §7 next steps (incl. done/rejected items 7.5, 7.9), open questions & new research directions (2026-07-12 weekend notes) |

The pre-split single file is in git history (last full version: this file at commit `e7f3a79`
plus the then-uncommitted "recommended configuration" section, now in RESULTS §0).
`bayesian_coverage_SUMMARY.html` / `.pdf` are renders of the **pre-split** file — regenerate
per-file with pandoc if needed.

**Interactive report:** `bayes_coverage_summary.html` (same directory) is a self-contained
offline page telling the whole story with interactive curves — the R2–R5 tables as a
click-to-toggle curve explorer over all 34 arms and 4 noise levels. It is *generated*
(never hand-edit) by `python3 batch_utils/build_summary_report.py` from
`results/cifar100/results_master.csv`; its prose mirrors the three md files above, which
remain the source of record.
