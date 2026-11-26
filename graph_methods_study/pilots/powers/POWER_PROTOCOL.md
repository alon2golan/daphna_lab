# Compression of repeated paths to the same annotation

This follow-up changes only the weight assigned to each annotation in the three-hop vote. Raw path counts treat paths ending at the same annotation as separate votes. Square-root compression gives diminishing weight to additional paths. The 0/1 control gives each annotation one vote wherever any three-hop path reaches it. Both preserve exactly the nonzero support of the ordinary three-hop vote.

We hold all feature and graph inputs, acquisition orders, label noise and test scoring fixed. The same seeds 5–9, four label budgets, six noise conditions, and earlier fixed-depth controls are evaluated. The square-root transform and binary endpoint are fixed in advance and use the same rule at every label budget and noise condition.

Run in a fresh directory with the same numerical environment, geometry and audit helper files as the earlier non-backtracking experiment.
