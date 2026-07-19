"""Reorganize tiny-imagenet's val split into an ImageFolder-compatible tree.

The raw download stores validation images flat in val/images/*.JPEG with a
val/val_annotations.txt mapping each filename to its wnid class. torchvision's
ImageFolder needs one subdir per class, so we build val_sorted/<wnid>/<img>
alongside it. Images are symlinked (not copied) to avoid duplicating ~400MB.

Usage:
    python prepare_tinyimagenet_val.py --root ../data/tiny-imagenet-200
Produces <root>/val_sorted/<wnid>/<image>.JPEG
"""

import argparse
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    # dir holding train/ and val/ (the unzipped tiny-imagenet-200)
    p.add_argument("--root", required=True)
    args = p.parse_args()

    root = Path(args.root).resolve()
    val_dir = root / "val"
    ann = val_dir / "val_annotations.txt"
    out = root / "val_sorted"

    # Parse "filename<TAB>wnid<TAB>bbox..." -> keep only filename + wnid.
    mapping = {}
    with open(ann) as f:
        for line in f:
            parts = line.split("\t")
            mapping[parts[0]] = parts[1]

    # One symlink per image under its class folder; skip if already linked so the
    # script is safe to re-run.
    made = 0
    for fname, wnid in mapping.items():
        class_dir = out / wnid
        class_dir.mkdir(parents=True, exist_ok=True)
        link = class_dir / fname
        src = val_dir / "images" / fname
        if not link.exists():
            os.symlink(src, link)
            made += 1

    n_classes = sum(1 for _ in out.iterdir())
    print(f"val_sorted ready: {len(mapping)} images across {n_classes} classes "
          f"({made} new symlinks) -> {out}")


if __name__ == "__main__":
    main()
