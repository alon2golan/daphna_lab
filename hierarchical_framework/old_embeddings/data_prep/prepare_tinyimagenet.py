# Build the TinyImageNet embeddings dir ProbCover/accuracy_vs_budget expect.
#
# Feature source: itai.david's representation bank. NOTE the bank holds several
# inconsistent SimCLR extractions; only features_seed1.npy is a strong, coherent
# train embedding (33% train leave-one-out 1-NN). Its matching val features are
# NOT in the bank (every val file there is from a different model, giving chance
# cross-set 1-NN), so we evaluate on a held-out slice of the coherent train set.
#
# Labels: deep_al's TinyImageNet uses torchvision ImageFolder, which assigns
# label i to the i-th ALPHABETICALLY-SORTED wnid and lays samples out contiguously
# (500 per class). So the label vector is repeat(arange(200), 500). This matches
# the sorted-wnid order used by build_tinyimagenet_coarse.py's coarse_map.

from pathlib import Path

import numpy as np

BANK = "/cs/labs/daphna/itai.david/representations_bank/tiny-imagenet_simclr/pretext"
# old_embeddings/data/ holds itai's SimCLR embeddings, L2-normalized (data_prep/ -> up one)
OUT = Path(__file__).resolve().parents[1] / "data" / "itai_embeddings" / "tinyimagenet_simclr"
PER_CLASS = 500          # images per class in TinyImageNet train
N_CLASSES = 200
HOLDOUT = 50             # images per class held out as the eval set


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X = np.load(f"{BANK}/features_seed1.npy").astype(np.float32)          # (100000, 512)
    X /= np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)    # L2-normalize
    assert X.shape[0] == PER_CLASS * N_CLASSES, X.shape
    # contiguous class blocks -> label = block index
    labels = np.repeat(np.arange(N_CLASSES), PER_CLASS)
    # hold out the last HOLDOUT of each class block as the eval split
    test_idx = np.concatenate([np.arange(c * PER_CLASS + PER_CLASS - HOLDOUT,
                                         c * PER_CLASS + PER_CLASS) for c in range(N_CLASSES)])
    mask = np.zeros(len(X), bool); mask[test_idx] = True

    np.save(OUT / "train_features_normalized.npy", X[~mask])   # pool ProbCover selects from
    np.save(OUT / "train_labels.npy", labels[~mask])
    np.save(OUT / "test_features_normalized.npy", X[mask])     # held-out eval
    np.save(OUT / "test_labels.npy", labels[mask])
    print(f"pool={int((~mask).sum())} test={int(mask.sum())} classes={N_CLASSES}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
