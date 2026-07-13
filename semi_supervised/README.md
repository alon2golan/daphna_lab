# Semi- and Self-Supervised

Workspace for semi-supervised and self-supervised learning ideas.

The layout separates two concerns:

- models/      base representation-learning encoders (SupCon, SimCLR, BYOL, ...).
               These are the backbones adapted from their upstream repos.
- algorithms/  methods built on top of those encoders. New research ideas go here.

## Status

Early scaffold. Base models will be ported in from their original GitHub
repositories one at a time; algorithms are implemented on top of them.

## Relation to the rest of the repo

This folder is intended to stand on its own for now, but be import-friendly so
it can later interoperate with the active-learning code under deep_al/ and the
self-supervised code under scan/ (e.g. sharing features, datasets, or encoders).
The directory uses an underscore (semi_supervised) rather than a hyphen so it is
a valid Python package.
