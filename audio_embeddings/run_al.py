###################################
# Endpoint for running AAL algs
# See README.md for detaild
# @yael-ben.shalom@mail.huji.ac.il
###################################

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from audio_embeddings.cli import _load_yaml
    from audio_embeddings.training.trainer_al import run_experiment
else:
    from .cli import _load_yaml
    from .training.trainer_al import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run configured audio active learning")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run_experiment(_load_yaml(args.config)))


if __name__ == "__main__":
    main()
