"""Iterate between SimCLR and SupCon on a shared backbone.

The idea: alternate an unsupervised SimCLR phase (trained on the whole image
pool, no labels) with a supervised SupCon phase (trained only on a small labeled
subset), repeatedly, warm-starting each phase from the backbone the previous
phase produced. SimCLR keeps the representation general; SupCon pulls same-class
samples together using the few labels available.

Both phases are real solo-learn training runs executed through
ssl_framework.backends.solo_learn; the only thing carried between them is the
backbone's weights. The labeled subset is chosen once by a Selector (random for
now -- swap in an active-learning selector later without changing this file).

Everything runs in a single process, so the whole iteration is one job. Run it
directly:

    python -m ssl_framework.algorithms.simclr_supcon \
        --dataset cifar100 --n_labeled 1000 \
        --num_iterations 3 --simclr_epochs 200 --supcon_epochs 100 \
        --output_dir runs/simclr_supcon_cifar100_l1000
"""

import argparse
from pathlib import Path

import numpy as np

from ..backends import solo_learn
from ..labeling import RandomSelector


def run(
    dataset,
    n_labeled,
    output_dir,
    num_iterations=3,
    simclr_epochs=200,
    supcon_epochs=100,
    data_dir=solo_learn.DEFAULT_DATA_DIR,
    seed=1,
    num_workers=4,
    start_with="simclr",
):
    """Run the SimCLR<->SupCon iteration and save the backbone + embeddings.

    Args mirror the CLI below. `start_with` picks which phase opens each
    iteration; the default starts unsupervised so the very first backbone is a
    SimCLR representation before any labels are used.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    solo_learn.set_seed(seed)

    # Choose the labeled subset once, up front, and record it.
    n_pool = solo_learn.train_pool_size(dataset, data_dir)
    labeled = RandomSelector().select(n_pool, n_labeled, seed)
    np.save(out / "labeled_indices.npy", np.array(labeled))
    print(f"pool={n_pool} labeled={len(labeled)} dataset={dataset} seed={seed}")

    order = ["simclr", "supcon"] if start_with == "simclr" else ["supcon", "simclr"]
    epochs = {"simclr": simclr_epochs, "supcon": supcon_epochs}

    backbone_state = None
    result = None
    for it in range(num_iterations):
        for method in order:
            # SupCon trains on the labeled subset; SimCLR on the whole pool.
            labeled_indices = labeled if method == "supcon" else None
            n_imgs = len(labeled) if method == "supcon" else n_pool
            print(f"[iter {it}] {method} phase: {epochs[method]} epochs on {n_imgs} images")
            result = solo_learn.run_phase(
                method,
                dataset,
                epochs[method],
                data_dir=data_dir,
                init_backbone_state=backbone_state,
                labeled_indices=labeled_indices,
                num_workers=num_workers,
            )
            backbone_state = result["backbone_state"]

        solo_learn.save_backbone_state(backbone_state, out / f"backbone_iter{it}.pt")

    solo_learn.save_backbone_state(backbone_state, out / "backbone_final.pt")

    # Export frozen features from the final backbone for downstream use.
    print("extracting embeddings from the final backbone ...")
    emb = solo_learn.extract_embeddings(result["model"].backbone, dataset, data_dir,
                                        num_workers=num_workers)
    for split, (feats, labels) in emb.items():
        np.save(out / f"{split}_features.npy", feats)
        np.save(out / f"{split}_labels.npy", labels)
        print(f"  {split}: features {feats.shape}")
    print("done ->", out)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", default="cifar100", help="cifar10 | cifar100")
    p.add_argument("--n_labeled", type=int, required=True,
                   help="size of the labeled set used by the SupCon phases")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_iterations", type=int, default=3)
    p.add_argument("--simclr_epochs", type=int, default=200)
    p.add_argument("--supcon_epochs", type=int, default=100)
    p.add_argument("--data_dir", default=str(solo_learn.DEFAULT_DATA_DIR))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--start_with", default="simclr", choices=["simclr", "supcon"])
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
