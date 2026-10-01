# Run VAST

GPU is optional; CPU works too. These scripts request a cluster GPU using the
repo's Slurm conventions. Graph construction uses CPU FAISS; the MLP uses GPU.

Install once from the repo root (Python 3.11 or 3.12):

```bash
python3.12 -m venv .venv-vast
.venv-vast/bin/python -m pip install -r deep_al/vast/requirements.txt
```

1. **Short check:** `sbatch onboarding/VAST_smoke.sh`
   Runs the complete pipeline with 400 synthetic training examples and 2 epochs.
   It checks FAISS, propagation, GPU training and output writing; it does not
   validate real feature files or scientific accuracy.
2. **Full run:** `sbatch onboarding/VAST_full.sh`
   Runs all 3 datasets × 1/2/4 labels per class, 300 epochs each, one seed (0).
   First create `data/vast/{cifar100,imagenet100,tinyimagenet}.yaml` using
   [the manifest template](manifest.example.yaml) and your aligned features,
   labels and sample IDs. These data/metadata are not included in this checkout.

Both scripts default to `/cs/labs/daphna/$USER/projects/daphna_lab` and save under
`results/vast/`. Edit their top variables or export `REPO_DIR`, `VAST_PYTHON`,
`VAST_OUTPUT_DIR`, `SEED`, or `VAST_MANIFEST_DIR` as needed. Output folders must be
new. Full-run validation without training: `VAST_DRY_RUN=1 bash onboarding/VAST_full.sh`.
Local CPU smoke: `REPO_DIR="$PWD" VAST_DEVICE=cpu bash onboarding/VAST_smoke.sh`.
Cluster time/memory requests are starting allocations, not measured guarantees.

**Both scripts explicitly use experimental `literal-paper` propagation.**
The paper's propagation discrepancy is unresolved; completing a full run does
not establish reproduction of its numbers. Current lab embeddings also differ.

[Technical run details](RUN_DETAILS.md) · [Code provenance](PROVENANCE.md) ·
[Specification and open question](../../docs/VAST_REPRODUCTION_SPEC.md)

To replay a new run after training (no GPU needed for viewing):

```bash
.venv-vast/bin/python -m pip install -r deep_al/vast/requirements-viewer.txt
.venv-vast/bin/python -m streamlit run deep_al/tools/view_vast.py -- --run-dir /path/to/run
```

Move the iteration slider to inspect senders, receivers and newly frozen samples.
For a smoke run, use its `run/` subfolder. You can copy that run folder to your
laptop and view it there; original feature files are not needed. Older runs must
be rerun to create the sample-level replay logs.
