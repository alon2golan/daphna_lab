#!/bin/bash
# Run once on the cluster gateway. Keep Python on shared storage for Slurm nodes.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VAST_BOOTSTRAP_PYTHON="${VAST_BOOTSTRAP_PYTHON:-/usr/bin/python3}"
export UV_PYTHON_INSTALL_DIR="$REPO_DIR/.vast-python"
export UV_PYTHON_BIN_DIR="$REPO_DIR/.vast-python-bin"
export UV_CACHE_DIR="$REPO_DIR/.vast-cache"
VAST_UV="$REPO_DIR/.vast-bootstrap/bin/uv"

cd "$REPO_DIR"
if [[ ! -x "$VAST_UV" ]]; then
  "$VAST_BOOTSTRAP_PYTHON" -m pip install --only-binary=:all: \
    --target "$REPO_DIR/.vast-bootstrap" uv==0.8.22
fi
"$VAST_UV" python install 3.13.5
shared_python="$("$VAST_UV" python find --managed-python 3.13.5)"

# Recreate the interpreter links while retaining installed packages.
if [[ -f .venv-vast/pyvenv.cfg ]]; then
  backup_dir="results/vast/env_backup_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$backup_dir"
  cp -a .venv-vast/pyvenv.cfg "$backup_dir/"
  for executable in python python3 python3.13; do
    if [[ -e ".venv-vast/bin/$executable" || -L ".venv-vast/bin/$executable" ]]; then
      cp -a ".venv-vast/bin/$executable" "$backup_dir/"
    fi
  done
fi
"$VAST_UV" venv --allow-existing --python "$shared_python" .venv-vast
"$VAST_UV" pip install --python "$REPO_DIR/.venv-vast/bin/python" \
  --only-binary=:all: -r deep_al/vast/requirements.txt
"$REPO_DIR/.venv-vast/bin/python" - <<'PY'
import sys
import faiss
import numpy
import torch
print(f"Python: {sys.version.split()[0]} | base: {sys.base_prefix}")
print(f"NumPy: {numpy.__version__} | FAISS: {faiss.__version__}")
print(f"PyTorch: {torch.__version__} | CUDA build: {torch.version.cuda}")
PY
echo "Environment ready. Submit: sbatch onboarding/VAST_smoke.sh"
