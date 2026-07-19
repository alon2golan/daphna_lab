# Fine -> coarse maps shared across the pipeline and analysis tools.

from pathlib import Path
import numpy as np

# Maps 100 fine labels (0-99) to 20 coarse superclasses (0-19).
# Array index = Fine Label ID, Value at index = Coarse Superclass ID.
# Example: CIFAR100_COARSE[0] is 4 -> Fine class 0 belongs to superclass 4.
CIFAR100_COARSE = np.array([
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
], dtype=np.int64)

# Human-readable names for the 20 CIFAR-100 superclasses, indexed by coarse id.
CIFAR100_COARSE_NAMES = [
    "aquatic mammals", "fish", "flowers", "food containers", "fruit/veg",
    "electrical devices", "furniture", "insects", "large carnivores",
    "man-made outdoor", "natural outdoor", "large omni/herbiv",
    "medium mammals", "non-insect inverts", "people", "reptiles",
    "small mammals", "trees", "vehicles 1", "vehicles 2",
]

# Names for the 100 CIFAR-100 fine classes, in torchvision (alphabetical) label order.
CIFAR100_FINE_NAMES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]


def load_coarse_map(embeddings_dir, dataset="cifar100"):
    """Loads custom map from disk, falls back to CIFAR-100 defaults, or returns None."""
    p = Path(embeddings_dir) / "coarse_map.npy"

    # 1. Try loading external map file if present
    if p.exists():
        return np.load(p)

    # 2. Use built-in map for standard CIFAR-100 runs
    if dataset.lower().startswith("cifar100"):
        return CIFAR100_COARSE

    return None