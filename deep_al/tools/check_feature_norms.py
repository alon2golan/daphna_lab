import argparse
import numpy as np

import deep_al.pycls.datasets.utils as ds_utils


def l2_normalize(x, eps=1e-12):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def print_norm_stats(name, x):
    norms = np.linalg.norm(x, axis=1)
    print(f"\n{name}")
    print(f"shape: {x.shape}")
    print(f"norm min:  {norms.min():.8f}")
    print(f"norm mean: {norms.mean():.8f}")
    print(f"norm max:  {norms.max():.8f}")
    print(f"all close to 1: {np.allclose(norms, 1.0, atol=1e-5)}")
    print(f"max |norm-1|: {np.max(np.abs(norms - 1.0)):.8e}")


def check_split(dataset, seed, train, features_tag):
    split = "train" if train else "test"

    raw = ds_utils.load_features(
        dataset,
        seed=seed,
        train=train,
        normalized=False,
        features_tag=features_tag,
    )

    loaded_normed = ds_utils.load_features(
        dataset,
        seed=seed,
        train=train,
        normalized=True,
        features_tag=features_tag,
    )

    manual_normed = l2_normalize(raw)

    print_norm_stats(f"{split} raw features", raw)
    print_norm_stats(f"{split} load_features(normalized=True)", loaded_normed)

    print(f"\n{split} check")
    print("loaded_normed == manual_normed:",
          np.allclose(loaded_normed, manual_normed, atol=1e-6))
    print("max abs diff:",
          np.max(np.abs(loaded_normed - manual_normed)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CIFAR100")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--features_tag", default=None)
    args = parser.parse_args()

    check_split(args.dataset, args.seed, train=True, features_tag=args.features_tag)
    check_split(args.dataset, args.seed, train=False, features_tag=args.features_tag)


if __name__ == "__main__":
    main()