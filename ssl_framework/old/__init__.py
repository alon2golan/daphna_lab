"""Archived SimCLR<->SupCon concept work (shelved, not under active development).

Kept runnable -- these still import the live shared modules (labeling,
propagation, evaluation, backends) from the ssl_framework package, so nothing
here was edited beyond its folder. Invoke with e.g.
    python -m ssl_framework.old.supcon_finetune ...

    simclr_supcon     -- iterate between unsupervised SimCLR (full pool) and
                         supervised SupCon (labeled subset) on a shared backbone.
    supcon_finetune   -- fine-tune an EXISTING SimCLR/BYOL encoder with SupCon on
                         a labeled set (+ optional kernel-propagated pseudo-labels),
                         comparing conditions on val_acc1/5 + val_knn_acc1/5.
    propagation_sweep -- cheap, training-free preview of how the propagation knobs
                         change the pseudo-labeled set, to pick a grid.
    threshold_search  -- rank (kernel, bandwidth, sim/conf) combinations by the NET
                         correct pseudo-supervision they add (no training).

The current line of work lives in algorithms.joint_finetune.
"""
