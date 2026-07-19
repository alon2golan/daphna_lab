# Build a coarse (superclass) map for TinyImageNet-200, the CIFAR-100 analogue.
# TinyImageNet ships no superclasses, so we derive them from WordNet: each of the
# 200 classes is a WordNet noun synset (its wnid encodes the synset offset). We
# walk each synset UP its hypernym paths and assign it to the first matching
# anchor category from a curated list -> a readable ~20-group taxonomy.
#
# Label ordering: deep_al's TinyImageNet uses torchvision ImageFolder, which
# assigns label i to the i-th ALPHABETICALLY-SORTED wnid. We reproduce that here.

import json
from pathlib import Path

import numpy as np
from nltk.corpus import wordnet as wn

TI = "/cs/labs/daphna/itai.david/py_repos/tiny-imagenet-200"

# curated anchor synsets, most specific first: a class is assigned to the FIRST
# of these it can reach by climbing hypernyms. order matters (dog before mammal).
ANCHORS = [
    ("dog", "dog.n.01"), ("feline", "feline.n.01"), ("bear", "bear.n.01"),
    ("primate", "primate.n.02"), ("hoofed", "ungulate.n.01"),
    ("rodent", "rodent.n.01"), ("marine mammal", "aquatic_mammal.n.01"),
    ("mammal", "mammal.n.01"),
    ("bird", "bird.n.01"), ("fish", "fish.n.01"),
    ("reptile", "reptile.n.01"), ("amphibian", "amphibian.n.03"),
    ("insect", "insect.n.01"), ("other invertebrate", "invertebrate.n.01"),
    ("vehicle", "vehicle.n.01"), ("musical instrument", "musical_instrument.n.01"),
    ("clothing", "clothing.n.01"), ("furniture", "furniture.n.01"),
    ("container", "container.n.01"), ("structure", "structure.n.01"),
    ("device/instrument", "device.n.01"), ("appliance", "appliance.n.02"),
    ("food", "food.n.01"), ("fruit/veg", "produce.n.01"),
    ("plant", "plant.n.02"), ("person", "person.n.01"),
    ("geological", "geological_formation.n.01"),
]


def wnid_to_synset(wnid):
    # wnid n01443537 -> noun synset at offset 1443537
    return wn.synset_from_pos_and_offset("n", int(wnid[1:]))


def coarse_of(syn, anchor_syns):
    # all ancestors of this synset (union over every hypernym path), as a set
    anc = set()
    for path in syn.hypernym_paths():
        anc.update(path)
    # first anchor (in priority order) that appears among the ancestors
    for name, a in anchor_syns:
        if a in anc:
            return name
    return "other"


def main():
    # torchvision label order = sorted wnids
    wnids = sorted(l.strip() for l in open(f"{TI}/wnids.txt") if l.strip())
    assert len(wnids) == 200, len(wnids)
    anchor_syns = [(name, wn.synset(s)) for name, s in ANCHORS]

    # coarse name for each of the 200 fine labels, in label order
    names_per_label, fine_names = [], []
    for w in wnids:
        syn = wnid_to_synset(w)
        names_per_label.append(coarse_of(syn, anchor_syns))
        fine_names.append(syn.lemma_names()[0])   # a human-readable class name

    # map the string groups to integer ids (stable: sorted by group name)
    groups = sorted(set(names_per_label))
    gid = {g: i for i, g in enumerate(groups)}
    coarse = np.array([gid[n] for n in names_per_label], dtype=np.int64)

    out = Path(__file__).resolve().parents[2] / "embeddings" / "old_embeddings" / "tinyimagenet_simclr"
    np.save(out / "coarse_map.npy", coarse)                 # (200,) fine-label -> coarse id
    json.dump(groups, open(out / "coarse_names.json", "w")) # coarse id -> name
    json.dump(fine_names, open(out / "fine_names.json", "w"))

    # report: how many fine classes fell into each coarse group
    counts = np.bincount(coarse, minlength=len(groups))
    print(f"{len(groups)} superclasses over 200 classes:")
    for i, g in enumerate(groups):
        print(f"  {i:2d} {g:20s} {counts[i]:3d} classes")
    print(f"saved -> {out}/coarse_map.npy (+ coarse_names.json, fine_names.json)")


if __name__ == "__main__":
    main()
