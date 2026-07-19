#!/usr/bin/env bash
# Submit the FULL SupCon-fine-tuning sweep to SLURM, structured for fast, live,
# well-organised results:
#   embeddings  {simclr, byol}
#   datasets    {cifar10, cifar100}
#   budgets     per-class L (n_labeled = L * num_classes; cifar10 x10, cifar100 x100)
#   kernels     {gaussian, laplacian, cosine}  (all run inside each budget job)
#
# Two phases per (embedding, dataset):
#   1) ONE baseline job  -- extracts + caches the frozen features and scores the
#      raw encoder (no labels; identical across budgets). Fast, submitted first.
#   2) One budget job per L, each depending (afterok) on that baseline job so it
#      reuses the cached features. Budgets submitted small -> large, so points
#      appear quickly in wandb and grow.
# => 4 baseline jobs + 4 x len(BUDGETS) budget jobs.
#
# Results tree:  ssl_framework/results/<dataset>/<embedding>/{baseline,supcon_only,supcon_prop/...}
# Slurm logs:    ssl_framework/slurm_logs/<jobname>_<jobid>.{out,err}
# wandb:         project semi-supervised-supcon, group=<embedding>-<dataset>,
#                job_type=baseline|supcon_only|supcon_prop, tags + rich config so
#                you can plot X=per_class_budget vs Y=metric, split by embedding,
#                dataset, condition, kernel, hyperparameters.
#
# Usage (submit from solo-learn):
#   ./run_supcon_sweep_all.sh                 # budgets 2,4,6,10,25,50
#   ./run_supcon_sweep_all.sh 2,4,6,10
#   BUDGETS=2,4,6 SEED=3 ./run_supcon_sweep_all.sh
# Env:
#   SEED (default 1); BUDGETS (default 2,4,6,10,25,50; positional wins);
#   EXTRA extra flags forwarded to budget jobs (e.g. "--sim_threshold 0.1 --conf_threshold 0.8");
#   WANDB_OFFLINE=1; DRY=1 (print, do not submit).

set -e

BUDGETS=${1:-${BUDGETS:-2,4,6,10,25,50}}
SEED=${SEED:-1}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # .../ssl_framework/scripts
ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")
SBATCH="$SCRIPT_DIR/run_supcon_finetune.sbatch"
RESULTS_ROOT="$ROOT/ssl_framework/results"
KERNELS="gaussian,laplacian,cosine"
mkdir -p "$ROOT/ssl_framework/slurm_logs"

submit() {  # prints jobid on stdout; honours DRY / WANDB_OFFLINE
  local jobname=$1 extra=$2 dep=$3 ckpt=$4 dataset=$5
  local depflag=""; [[ -n "$dep" ]] && depflag="--dependency=afterok:$dep"
  if [[ "${DRY:-0}" == "1" ]]; then
    echo "DRY submit $jobname ${depflag:+[$depflag]} :: EXTRA=$extra" >&2
    echo "DRYID_$jobname"
  else
    EXTRA="$extra" WANDB_OFFLINE="${WANDB_OFFLINE:-0}" \
      sbatch --parsable --job-name="$jobname" $depflag --export=ALL \
        "$SBATCH" "$ckpt" "$dataset"
  fi
}

njobs=0
for METHOD in simclr byol; do
  for DATASET in cifar10 cifar100; do
    case "$DATASET" in cifar10) NCLASSES=10 ;; cifar100) NCLASSES=100 ;; esac
    CKPT="trained_models/$METHOD/$DATASET/1000ep"

    # phase 1: baseline (cache features), submitted first. It logs its label-free
    # score at every budget so it shows as a flat reference line across the plots.
    BASE_EXTRA="--mode baseline --results_root $RESULTS_ROOT --seed $SEED --reference_budgets $BUDGETS"
    BID=$(submit "scft_${METHOD}_${DATASET}_baseline" "$BASE_EXTRA" "" "$CKPT" "$DATASET")
    echo "baseline $METHOD/$DATASET -> job $BID"
    njobs=$(( njobs + 1 ))

    # phase 2: budget jobs, small -> large, each afterok the baseline
    IFS=',' read -ra LS <<< "$BUDGETS"
    for L in "${LS[@]}"; do
      N_LABELED=$(( L * NCLASSES ))
      # Thresholds are chosen automatically per case+budget: the job ranks the
      # grid on the cached features (no training) and trains the top-2 per kernel
      # that add positive, pure net supervision -- so gaussian/laplacian/cosine
      # each get their own best thresholds, and you see multiple attempts ranked.
      # Override via EXTRA (e.g. --auto_thresholds 1, or pin explicit lists).
      B_EXTRA="--mode budget --results_root $RESULTS_ROOT --seed $SEED \
--n_labeled $N_LABELED --per_class_budget $L --with_propagation \
--kernels $KERNELS --auto_thresholds 2 ${EXTRA:-}"
      submit "scft_${METHOD}_${DATASET}_L${L}" "$B_EXTRA" "$BID" "$CKPT" "$DATASET" >/dev/null
      echo "  budget $METHOD/$DATASET L=$L (n=$N_LABELED) -> afterok:$BID"
      njobs=$(( njobs + 1 ))
    done
  done
done

echo "$njobs jobs (4 baseline + budgets) for BUDGETS=$BUDGETS SEED=$SEED"
echo "track: squeue -u \$USER   |   results: $RESULTS_ROOT   |   logs: $ROOT/ssl_framework/slurm_logs"
