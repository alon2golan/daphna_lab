#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=00:35:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail
cd "$1"
project="$(dirname "$PWD")"
python="$project/daphna_lab/.venv-vast/bin/python"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
printf 'Exact backup coverage on %s\n' "$(hostname)"
"$python" -u dense_backup.py "$project/cifar100_herding_20261119/study" maxherding_rows.json orders.json
