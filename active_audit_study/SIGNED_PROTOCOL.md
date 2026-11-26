# Signed peer utility for clean audits

This prespecified follow-up uses the frozen geometry and budget of 800 noisy annotations plus 80 clean requests. Five new seeds vary acquisition and synthetic noise independently of the original pilot. The conditions are symmetric 50%, class-cycle asymmetric 50%, and CIFAR-100N. Both test graph voting accuracy and cosine 1NN are readouts. The primary decision criterion is a mean human-annotation graph-accuracy gain of at least 0.25 percentage points over leave-one-out auditing.

The proposed audit uses the existing leave-one-out rival and suspicion score. For every unqueried training point touched by the annotation, move its votes to its rival and find the new winning class. Estimate the gain as the change in the predicted class's probability under a soft distribution formed by removing all votes from the candidate annotation. Sum the gains with their signs and multiply by positive leave-one-out suspicion. If another label would lose peer support, the contribution is negative. Rows without peer evidence contribute zero. Test data and clean audit answers are unused in this ranking. The ranking takes the 80 highest scores, breaking ties by leave-one-out suspicion and initial acquisition order.

The original random audit, leave-one-out audit, no-audit, fresh clean expansion, unsigned independent footprint and greedy footprint policies serve as equal-cost controls except for the no-audit reference. This isolates one scoring change, while holding the acquired set, features, graph, depth, annotation costs, and downstream learners fixed. The protocol uses neither the known noise-family name nor the true noise rate in its scoring.

Use the validated `geometry.npz` from the original preparation job. Place this source in an isolated subdirectory beside that file and run the launcher below. Collect the results with `python collect.py PATH --signed-study`.

```sh
root=$PWD
sbatch --parsable --array=5-9%3 --job-name=c100-signed-audit --output="$root/replay_%A_%a.log" run_signed.slurm.sh "$root"
```
