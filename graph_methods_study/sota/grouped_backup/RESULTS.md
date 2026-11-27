# Grouped backup coverage pilot

Requiring the second representative to come from a different feature group did not improve ordinary backup coverage in the primary clustered-noise test. Across budgets 800 and 1,600, grouped backup changed six-hop accuracy by **−0.48 percentage points** on average, with seed-level changes of −0.345, −0.495 and −0.605 pp. Under IID flips matched to each seed's exact clustered corruption count, the change was −1.70 pp. The smaller penalty under cluster-correlated flips gives a +1.22 pp interaction in this pilot. Ordinary backup remained stronger in both conditions.

| Noise and annotations | Max-only | Ordinary backup | Grouped backup | Shuffled groups |
|---|---:|---:|---:|---:|
| Clean, 800 | 57.37% | 57.72% | 57.32% | 57.71% |
| Clean, 1,600 | 58.38% | 58.63% | 58.45% | 58.63% |
| IID 50%, 800 | 33.54% | 36.94% | 35.67% | 36.94% |
| IID 50%, 1,600 | 39.13% | 44.77% | 42.69% | 44.64% |
| Cyclic 50%, 800 | 31.73% | 32.49% | 32.85% | 32.46% |
| Cyclic 50%, 1,600 | 31.67% | 33.23% | 32.82% | 32.89% |
| Cluster-correlated, 800 | 30.91% | 32.17% | 31.33% | 32.20% |
| Cluster-correlated, 1,600 | 33.74% | 34.56% | 34.44% | 34.52% |
| Rate-matched IID, 800 | 34.96% | 37.64% | 36.68% | 37.56% |
| Rate-matched IID, 1,600 | 40.28% | 45.75% | 43.32% | 45.61% |
| CIFAR-100N, 800 | 48.38% | 49.86% | 49.57% | 49.87% |
| CIFAR-100N, 1,600 | 50.86% | 52.85% | 51.99% | 52.82% |

Grouping did change the intended mechanism. At 1,600 labels, the share of eligible unqueried rows with support from at least two real groups increased from 70.80% to 83.82%, while the size-preserving shuffled partition reached 70.58%. Under clustered noise the acquired wrong-label rate was 48.98% for ordinary backup and 48.96% for grouped backup. Pairs sampled within a selector group had error covariance 0.175–0.180 in the cluster-correlated condition and below 0.0004 in absolute value under rate-matched IID. The stress-test corruptions used a separate landmark partition.

The motivating expected-coverage model assumes a failed group contributes no usable similarity. The classifier here can instead be misled by a wrong label. A feature cell also approximates a shared failure source. The result covers this partition rule, three seeds and one held-out CIFAR-100 test set. The evaluation uses one frozen representation, six- and three-hop graph votes, and cosine 1NN.

## Reproduce

See [frozen protocol](PROTOCOL.md) and [paired summary](results/summary.csv). The three Slurm tasks in job **32691496** completed in 53 seconds each. Their result records contain source, graph, partition and observed-label hashes, aligned acquisitions, and test counts. The source and Slurm logs are archived in the lab path `/cs/labs/daphna/tomn/projects/cifar_grouped_backup_20261127.KRmC6x`.

In an isolated cluster folder, stage the files listed in `SOURCE_SHA256`. Dereference the link to the unchanged sparse backup module if staging only this directory. Run `sha256sum -c SOURCE_SHA256`, then submit `sbatch --array=0-2 grouped_backup.slurm.sh "$PWD"`. The supplied Slurm script verifies the four unit tests before each fit-free evaluation. Collect finished seed files into an empty output directory and run `python collect.py path/to/output`.
