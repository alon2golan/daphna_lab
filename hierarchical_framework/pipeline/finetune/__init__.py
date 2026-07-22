# The finetune package: compares FROZEN vs FINETUNED SimCLR embeddings at predicting the 20
# CIFAR-100 superclasses, as a function of how many training points are treated as coarse-labeled.
# Kept separate from the frozen-embedding evaluators -- it needs the checkpoint + raw images.
#
#   solo_backbone.py             -- load the backbone, build index-aligned datasets, extract features
#   labeled_budgets.py           -- build the labeled subset (pluggable selector; random default)
#   two_head_finetune.py         -- finetune with a coarse head (+ fine head as anti-collapse)
#   probes.py                    -- the shared 20-way MLP probe used by both curves
#   run_coarse_budget_sweep.py   -- the sweep runner: frozen vs finetuned across budgets (+ wandb),
#                                   writes CSV + PNG under outputs/finetune/. --shard names outputs
#                                   per parallel worker (own live wandb run in the shared group).
#   run_coarse_budget_merge.py   -- optional: combine the parallel shard CSVs into one CSV + PNG.
#
# Slurm launchers in ../../scripts/:
#   run_coarse_budget_sweep.sbatch        -- single job (1-2 GPUs, DataParallel), whole sweep.
#   run_coarse_budget_sweep_array.sbatch  -- job ARRAY: one GPU per shard (any type), each a LIVE
#                                            wandb run in the shared group -> overlay to watch live.
#   run_coarse_budget_merge.sbatch        -- optional CPU merge into the single combined CSV + PNG.
