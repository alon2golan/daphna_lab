# Historical acquisition orders with matched graph heads

We scored six fixed seed-0 BYOL acquisition orders on the independent 10,000-image CIFAR-100 test split. Each method used the same archived noisy annotation vector, eligible rows, first 100 queries and evaluation heads. The study included budget prefixes of 400, 800, 1,200, 1,600 and 2,000 for seven archived clean and synthetic-noise settings. All 175 historical cosine 1NN test counts reproduced exactly under the new replay.

## Acquisition at a fixed evaluation head

The reference in each row is the highest same-head test accuracy among archived MaxHerding, fixed-bandwidth U-Herding, one-step JS, observed-broadcast JS-10 and soft-broadcast JS-10, calculated separately per condition and budget. The reference is a descriptive upper envelope across existing controls.

| Head | Mean backup gain vs strongest archived order | Backup wins |
|---|---:|---:|
| Raw three-hop vote | +0.404 pp | 21/35 |
| Raw six-hop vote | +0.829 pp | 26/35 |
| Square-root three-hop vote | +0.445 pp | 22/35 |
| Square-root six-hop vote | +0.667 pp | 27/35 |
| Cosine 1NN | −0.420 pp | 7/35 |

With square-root six-hop voting, backup coverage averaged 47.626% across all matched cells. Observed-broadcast JS-10 was the strongest fixed historical acquisition at 46.505%, followed by MaxHerding at 46.427%. Backup improved by an average of +1.079 pp versus the same-head historical envelope on symmetric noise, +0.453 pp on asymmetric noise and +0.070 pp on clean data.

### Symmetric 50% noise with square-root six-hop voting

| Budget | Backup coverage | Strongest archived order | Gain |
|---:|---:|---|---:|
| 400 | 31.91% | Soft JS-10, 33.26% | −1.35 pp |
| 800 | 41.60% | Soft JS-10, 42.07% | −0.47 pp |
| 1,200 | 47.25% | Observed JS-10, 46.00% | +1.25 pp |
| 1,600 | 49.22% | Observed JS-10, 47.78% | +1.44 pp |
| 2,000 | 51.28% | Observed JS-10, 48.96% | +2.32 pp |

Backup coverage worked best with graph voting in this cohort. The advantage over the strongest archived acquisitions appeared at higher budgets under severe symmetric noise. Soft JS-10 remained stronger at 400 and 800. Under asymmetric 50% noise, the backup result ranged from +1.15 pp at 1,200 to −0.78 pp at 2,000 under the same square-root six-hop head.

## Weighting at a fixed acquisition order

Square-root six-hop weighting improved on raw six-hop voting for every acquisition order when averaged over the 15 symmetric condition/budget cells. The mean gains were +3.14 pp for MaxHerding, +3.49 for U-Herding, +3.15 for one-step JS, +3.43 for observed JS-10, +3.40 for soft JS-10 and +3.14 for backup coverage. This brings the weighting result across several distinct acquisition algorithms while keeping the selected rows and observed labels fixed within each comparison.

These comparisons use a single acquisition and corruption seed, with a shared test split and nested budget prefixes. The U-Herding comparator is the lab's archived fixed-bandwidth version using its acquired-label temperature calibration. The published U-Herding setup includes adaptive radius. CIFAR-100N and fresh noise seeds require their own adaptive acquisition trajectories. [Published UHerding paper](https://openreview.net/pdf/0b53a1002679bae2e0fe9413650fffef50d01b4d.pdf)

## Verification

Slurm array `32691453` completed all seven tasks. The collector verified all 1,050 test counts from 210 matched acquisitions. The replay validates every original 100-query snapshot against its ordered CSV and checks IDs and answers against the shared label vector. The graph and all BYOL bank arrays matched their content hashes. Frozen hashes are `7e41274c74029483eb8b52dbab96b493e71a722bfeac66265f0067d096f6ad4f` for the replay, `bd86279047547c9ae30b86bea409a7c26938a2443617d949774358f4e7e14fca` for the trusted evaluator, `72df76b08512438890caa902f8df8f81f0c9f61fe6b5e652833e9b401c10337d` for the shared data helper and `eff8bdb0a55bce77fa66eb6061c815efa5608b60c047500fe119894b252237e4` for the backup order.
