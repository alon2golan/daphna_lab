# Clean-label audits and expansion on CIFAR-100 BYOL

This replay compares actions given a small budget of clean-label requests after acquiring noisy labels. The primary metric is accuracy on the independent 10,000-image test split with three-hop inductive graph voting. Cosine 1NN on acquired labels is a second test classifier. Pool accuracy excludes all initial annotations and all rows any arm could buy at that budget. Test images have outgoing nearest-neighbor edges to the acquisition pool and send no evidence to it.

Five seeds vary greedy sparse coverage tie-breaking and the noise draws. At each seed, every policy shares the same 45,000 eligible rows and ordered initial annotations. The 5,000 historical validation rows are excluded from the acquisition and propagation graph. Acquisition uses reverse 13NN coverage including self and no labels. After 400 or 800 initial annotations, each intervention uses 40 or 80 clean requests. The settings are clean, symmetric 20% and 50%, and superclass-cycle asymmetric 20% and 50%. The seed-0 corruption vectors must agree exactly with the previously saved vectors. This protocol uses the lab's BYOL bank and exported study graph. Its three-hop votes and sparse coverage selector are a faster learner and acquisition setup than the historical MLP and full Gaussian MaxHerding.

The controls are the initial budget alone, a uniform random audit, auditing by normalized leave-one-out disagreement, and buying the next unseen coverage selections clean. All clean replies have unit weight. Policies see only their queried answers.

For each acquired label, the leave-one-out score is the margin of its highest-scoring alternative class over its observed class, divided by total evidence from other seeds. Its footprint is the set of unqueried pool rows whose prediction would stop matching that label if the anchor were changed to the alternative. We rank either by positive margin times footprint size or greedily by the uncovered contribution to the pointwise maximum of positive margins. This pair isolates overlap discounting. Zero-gain ties fall back to leave-one-out order. All settings and both budgets use the same objective and exactly three graph hops.

We compare held-out paired deltas per condition against leave-one-out and clean expansion. A general improvement needs both heavy-noise settings to improve over leave-one-out across seeds. Interpret a clean decrease against expansion as the cost of auditing clean labels. The replay uses a perfect simulated verifier; repeat-annotator reliability and CIFAR-100N are separate follow-ups.

## Run on the lab cluster

Use a fresh directory and copy the files in this folder plus the previously exported `cifar100_herding_20261119_neighbors.npz` as `neighbors.npz`. The preparation job verifies the four laboratory feature and target hashes and historical eligible row indices, runs the unit tests, and calculates exact test-to-pool cosine neighbors with FAISS.

```sh
cd YOUR_FRESH_DIRECTORY
root=$PWD
prep=$(sbatch --parsable --job-name=c100-audit-prepare --output="$root/prepare_%j.log" run.slurm.sh "$root")
sbatch --parsable --dependency="afterok:$prep" --array=0-4%3 --job-name=c100-audit-replay --output="$root/replay_%A_%a.log" run.slurm.sh "$root"
```

There are 300 result rows with five seeds, two initial budgets, five noise settings and six policies. `seed_*.json` records request rows, acquisition order, correct counts, software versions, and input and source digests. Resolved observed labels, test labels, images and raw features remain in their existing locations.

## CIFAR-100N follow-up

The follow-up uses the same method, acquisition orders, test targets and geometry with the fixed CIFAR-100N human annotation vector. Seed variation here comes from acquisition. The loader checks the historical annotation file checksum, uses restricted weights-only PyTorch deserialization and verifies all 50,000 clean labels against the frozen BYOL training order before using the annotation vector. To run, place the source and `run_human.slurm.sh` in a fresh child directory of the synthetic replay. Use a 5-task array and limit concurrency to three tasks. Collect results with `python collect.py PATH --human-labels`.

The signed follow-up used new acquisition seeds and added a signed estimate of the downstream utility. See the [follow-up protocol](SIGNED_PROTOCOL.md) and the [research report](../docs/NOISY_LABEL_RESEARCH_20261126.md).
