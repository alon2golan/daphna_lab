"""Dedicated VAST CLI; supports invocation from any working directory."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deep_al.vast.run import main

if __name__ == "__main__":
    main()
