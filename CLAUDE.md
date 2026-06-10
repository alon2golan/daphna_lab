# daphna_lab — working notes for Claude

## SLURM resource etiquette (READ BEFORE ANY sbatch SUBMIT)

The `daphna` SLURM account shares a **24 `g0`-class GPU cap** with **no per-user limit**, so
an unthrottled array can starve labmates (their jobs go `PENDING` / `AssocGrpGRES`). Before
launching any sweep / `sbatch` array:

1. **Check current group usage first** — run `gpus` on takara (alias in `~/.bash_aliases`), or:
   ```bash
   ssh phoenix 'bash -lc "squeue -A daphna -t RUNNING -h -o \"%u|%b\""'
   ```
2. **Cap array concurrency** with the `%N` throttle — default to **`--array=0-N%6`**
   (~6-8 GPUs, a quarter of the pool). Do NOT use high values like `%60`.
3. Request **`--gres=gpu:1`** per task unless the job genuinely uses multiple GPUs.
4. Prefer a **preemptible partition** (`killable`) so jobs yield instead of blocking labmates.

SLURM commands only run via `ssh phoenix` (not on the local node).
