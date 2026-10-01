"""Run three datasets by three label budgets with a single selected seed."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deep_al.vast.run import grid_main

if __name__ == "__main__":
    grid_main()
