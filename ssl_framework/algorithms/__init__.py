"""Research methods built on top of the backends.

    joint_finetune -- jointly fine-tune an existing SimCLR encoder with a
                      supervised SupCon term (labeled subset) and an unsupervised
                      SimCLR term (full pool), sweeping the unsupervised weight,
                      and compare conditions on val_acc1/5 + val_knn_acc1/5.

The earlier SimCLR<->SupCon concept work (simclr_supcon, supcon_finetune,
propagation_sweep, threshold_search) is shelved under ssl_framework.old.
"""
