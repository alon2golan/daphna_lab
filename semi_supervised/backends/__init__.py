"""Backends: adapters to external training frameworks.

Currently only solo_learn, which wraps the vendored solo-learn repo. Algorithms
import from here rather than importing solo-learn directly, so the rest of
semi_supervised stays independent of any one framework.
"""
