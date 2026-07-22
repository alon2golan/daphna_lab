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
   (~6-8 GPUs, a quarter of the pool) for NON-killable jobs. Do NOT use high values
   like `%60`. **Exception (Alon, 2026-07-22): jobs submitted with `--killable` need
   NO throttle** — run the full array; they yield to labmates via preemption instead
   of hogging quota. (A running throttled array can be lifted live with
   `scontrol update JobId=<id> ArrayTaskThrottle=0`.)
3. Request **`--gres=gpu:1`** per task unless the job genuinely uses multiple GPUs.
4. Prefer a **preemptible partition** (`killable`) so jobs yield instead of blocking labmates.

SLURM commands only run via `ssh phoenix` (not on the local node).

## train_al run tagging & dashboard visibility

train_al records two dashboard-facing knobs in `result.json` (via `reporting/result_writer.py`):

- **`--track_progress`** (store_true, **default OFF**) → `params.track_progress`. Marks a run
  whose accuracy-vs-episode curve belongs in the dashboard **Comparison** tab. Runs WITHOUT it
  are treated as figure-ingredients and surface only under **Additional data → Figure-ingredient
  runs**. **Any sweep whose curves you want to compare MUST pass `--track_progress`** in its
  launcher (the 3000cap family launchers already do). Older runs with no recorded flag fall back
  to category (`reg.TRACK_PROGRESS_CATEGORIES = {train_noise_nu_prime, aum_filter}`); the recorded
  flag always wins, so a new tracked sweep that forgets the flag will be hidden from Comparison.
- **`--tag KEY=VALUE`** (repeatable) → `result.json["tags"]`. Free-form facets; the scanner
  expands each to a `p_<key>` column that auto-appears as a groupable/filterable axis. Convention
  for ν' provenance: `--tag nu_tag=<optimizer>_b<nu_cache_budget>` (e.g. `sgd_b100`, `adam_b300`;
  analytical→`nu_analytical`, oracle→derived `none`). If a run carries no explicit `nu_tag`, the
  scanner derives one optimizer-aware from the exp-name (`nucache<N>` + `sgd`/`adam`). Tags never
  enter `config_key`, so adding one won't fragment seed-grouping or change `about=<hash>` URLs.
