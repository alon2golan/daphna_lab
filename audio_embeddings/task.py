"""Task types shared by audio dataset, training, and acquisition code."""
from enum import Enum


class Task(str, Enum):
    multiclass = "multiclass"
    multilabel = "multilabel"
