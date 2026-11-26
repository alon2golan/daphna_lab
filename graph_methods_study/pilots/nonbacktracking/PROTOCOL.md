# Non-backtracking graph voting

This pilot tests removing immediate return walks from three-hop votes on the frozen CIFAR-100 BYOL graph. A returning walk can bring a seed's evidence back to the same seed through a reciprocal edge. That extra vote depends on neighbor reciprocity. It does not represent another source annotation.

The directed evaluation graph has edges from held-out test images to training neighbors. Training rows have no edges to test rows. If `W3` is the ordinary three-hop test vote matrix and `W1` is the direct test-to-anchor indicator, the exact non-backtracking three-hop vote matrix is `W3 - W1 * reciprocal_degree`. The only excluded path has the form test → u → v → u. The correction counts each reciprocal edge of anchor u exactly once. The rest of the learner still predicts the class with the most votes.

The controls count ordinary paths at depths one through four. All policies use the same noisy acquisition orders, known labels, eligible training geometry, and independent 10,000 test targets. The five acquisition/noise seeds are 5–9. Budgets are 100, 400, 800, and 1,200. Conditions are clean, symmetric 20% and 50%, asymmetric superclass-cycle 20% and 50%, and the same human CIFAR-100N annotation vector. The main comparison is non-backtracking depth 3 versus ordinary depth 3. Changes in test coverage and vote mass are recorded alongside paired accuracy.

The runner uses the original active audit preparation file and the safe noise and acquisition helpers from its human-annotation replay source. It reads the historical files in place. From a fresh cluster directory containing the pilot source, run `sbatch --array=5-9%3 --output=replay_%A_%a.log run_graph_kernel.slurm.sh "$PWD"`.
