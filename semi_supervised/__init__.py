"""Semi- and self-supervised learning workspace.

Layout:
    backends    -- adapters to external training frameworks (solo-learn).
    algorithms  -- research methods built on top of those backends.
    labeling    -- strategies for choosing which pool samples to label.

Base encoders (SimCLR, SupCon, BYOL, ...) and their training come from the
vendored solo-learn repo, reached only through backends.solo_learn.
"""
