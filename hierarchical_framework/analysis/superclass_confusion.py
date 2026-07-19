# What did the 1-NN classifier actually confuse? Builds superclass confusion
# heatmaps and lists the concrete (true -> guessed) mistakes, for TWO predictors
# side by side:
#   full      = 1-NN fit on ALL train labels  -> the embedding's own ceiling
#               (how good the representation is, ignoring active learning)
#   probcover = 1-NN fit only on the ~budget*rounds points ProbCover selected
#               (what the AL run actually achieves)
# Outputs go to outputs/figures/error_analysis/<tag>/.

import argparse
import sys
from pathlib import Path

import numpy as np

# hierarchical_framework/ root (analysis/ -> up one); make the pipeline package
# importable for the shared selectors, and land figures in outputs/
HF_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = HF_ROOT / "outputs"
if str(HF_ROOT / "pipeline") not in sys.path:
    sys.path.insert(0, str(HF_ROOT / "pipeline"))

from selection_strategies import SELECTORS
from accuracy_vs_budget import CIFAR100_COARSE


# 20 CIFAR-100 superclass names, indexed by coarse id (matches CIFAR100_COARSE)
COARSE_NAMES = [
    "aquatic mammals", "fish", "flowers", "food containers", "fruit/veg",
    "electrical devices", "furniture", "insects", "large carnivores",
    "man-made outdoor", "natural outdoor", "large omni/herbiv",
    "medium mammals", "non-insect inverts", "people", "reptiles",
    "small mammals", "trees", "vehicles 1", "vehicles 2",
]

# 100 CIFAR-100 fine class names in torchvision (alphabetical) label order
FINE_NAMES = [
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


def one_nn(train_feats, train_labels, fit_idx, test_feats):
    # 1-NN fit on the chosen train subset (fit_idx), predict every test point
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(train_feats[fit_idx], train_labels[fit_idx])
    return clf.predict(test_feats)


def probcover_idx(train_feats, strategy, delta, budget, rounds, seed):
    # replay the AL run and return the full set of selected point indices
    sel = SELECTORS[strategy](train_feats, delta=delta, seed=seed)
    idx = []
    for _ in range(rounds):
        idx.extend(sel.select_round(budget))
    return np.array(idx)


def coarse_confusion(preds, truth, coarse_map, n_coarse=20):
    # 20x20 count matrix M[t, p] = #(true superclass t predicted as p)
    tc, pc = coarse_map[truth], coarse_map[preds]
    M = np.zeros((n_coarse, n_coarse), dtype=int)
    np.add.at(M, (tc, pc), 1)
    return M


def heatmap(ax, M, title):
    # row-normalize to percentages so every true-superclass row sums to 100
    row = M.sum(axis=1, keepdims=True)
    pct = 100.0 * M / np.clip(row, 1, None)
    im = ax.imshow(pct, cmap="magma", vmin=0, vmax=100)
    ax.set_xticks(range(len(COARSE_NAMES))); ax.set_yticks(range(len(COARSE_NAMES)))
    ax.set_xticklabels(COARSE_NAMES, rotation=90, fontsize=6)
    ax.set_yticklabels(COARSE_NAMES, fontsize=6)
    ax.set_xlabel("predicted superclass"); ax.set_ylabel("true superclass")
    # diagonal accuracy = trace / total, the coarse accuracy
    acc = 100.0 * np.trace(M) / M.sum()
    ax.set_title(f"{title}\ncoarse acc = {acc:.1f}%  (bright diagonal = good)")
    return im


def mistakes_table(preds, truth, coarse_map, out_csv):
    # every wrong test point: what it was vs what it was guessed as, and whether
    # the guess at least landed in the right superclass
    err = np.where(preds != truth)[0]
    with open(out_csv, "w") as fh:
        fh.write("test_idx,true_fine,true_name,pred_fine,pred_name,"
                 "true_super,pred_super,same_superclass\n")
        for i in err:
            t, p = int(truth[i]), int(preds[i])
            same = int(coarse_map[t] == coarse_map[p])
            fh.write(f"{i},{t},{FINE_NAMES[t]},{p},{FINE_NAMES[p]},"
                     f"{COARSE_NAMES[coarse_map[t]]},{COARSE_NAMES[coarse_map[p]]},{same}\n")
    return len(err)


def confused_pairs(preds, truth, coarse_map, out_csv, topk=30):
    # aggregate the mistakes into (true_fine -> pred_fine) pairs, most frequent first
    err = preds != truth
    pairs = {}
    for t, p in zip(truth[err], preds[err]):
        pairs[(int(t), int(p))] = pairs.get((int(t), int(p)), 0) + 1
    ranked = sorted(pairs.items(), key=lambda kv: kv[1], reverse=True)
    with open(out_csv, "w") as fh:
        fh.write("true_name,pred_name,count,same_superclass\n")
        for (t, p), c in ranked:
            same = int(coarse_map[t] == coarse_map[p])
            fh.write(f"{FINE_NAMES[t]},{FINE_NAMES[p]},{c},{same}\n")
    # return the top-k for printing to the console
    return [((FINE_NAMES[t], FINE_NAMES[p]), c,
             int(coarse_map[t] == coarse_map[p])) for (t, p), c in ranked[:topk]]


def log_wandb_confusion(preds_full, preds_pc, test_labels, coarse_map, wandb_kwargs, png=None):
    import wandb
    wandb.init(**wandb_kwargs)
    # map fine truth/preds up to the 20 superclasses for a readable 20x20 matrix
    tc = coarse_map[test_labels]
    for name, preds in [("full", preds_full), ("probcover", preds_pc)]:
        pc = coarse_map[preds]
        # wandb's interactive confusion matrix: hover a cell to read the count
        wandb.log({f"coarse_confusion_{name}": wandb.plot.confusion_matrix(
            y_true=tc.tolist(), preds=pc.tolist(), class_names=COARSE_NAMES)})
    # also upload the rendered side-by-side figure as an image panel
    if png is not None:
        wandb.log({"confusion_coarse": wandb.Image(str(png))})
    wandb.finish()


def run(embeddings_dir, strategy="probcover", delta=0.65, budget=100, rounds=10,
        split="train", seed=0, topk=20, use_wandb=False, wandb_kwargs=None):
    emb = Path(embeddings_dir)
    train_feats = np.load(emb / f"{split}_features_normalized.npy").astype(np.float32)
    train_labels = np.load(emb / f"{split}_labels.npy")
    test_feats = np.load(emb / "test_features_normalized.npy").astype(np.float32)
    test_labels = np.load(emb / "test_labels.npy")
    coarse_map = CIFAR100_COARSE
    print(f"test set = {len(test_labels)} images, {int(train_labels.max()) + 1} classes", flush=True)

    # predictor 1: the embedding ceiling -- 1-NN using ALL train labels
    full_idx = np.arange(len(train_labels))
    preds_full = one_nn(train_feats, train_labels, full_idx, test_feats)
    # predictor 2: what ProbCover's selected labels achieve
    pc_idx = probcover_idx(train_feats, strategy, delta, budget, rounds, seed)
    preds_pc = one_nn(train_feats, train_labels, pc_idx, test_feats)

    tag = "_".join(emb.parts[-3:])
    out_base = f"{tag}_{strategy}_d{delta}_b{budget}"
    outdir = OUTPUTS / "figures" / "error_analysis" / out_base
    outdir.mkdir(parents=True, exist_ok=True)

    # side-by-side coarse confusion heatmaps: ceiling vs probcover
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    M_full = coarse_confusion(preds_full, test_labels, coarse_map)
    M_pc = coarse_confusion(preds_pc, test_labels, coarse_map)
    heatmap(ax[0], M_full, f"embedding ceiling (1-NN on all {len(full_idx)} labels)")
    im = heatmap(ax[1], M_pc, f"probcover (1-NN on {len(pc_idx)} selected labels)")
    fig.colorbar(im, ax=ax, fraction=0.02, label="% of true-superclass row")
    png = outdir / "confusion_coarse.png"
    fig.savefig(png, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {png}", flush=True)

    # per-predictor: the concrete mistakes + the most confused fine-class pairs
    for name, preds in [("full", preds_full), ("probcover", preds_pc)]:
        n_err = mistakes_table(preds, test_labels, coarse_map, outdir / f"mistakes_{name}.csv")
        top = confused_pairs(preds, test_labels, coarse_map, outdir / f"confusedpairs_{name}.csv", topk)
        fine = 100.0 * float((preds == test_labels).mean())
        print(f"\n[{name}] fine acc={fine:.2f}%  errors={n_err}/{len(test_labels)}", flush=True)
        print(f"  saved -> {outdir / f'mistakes_{name}.csv'}", flush=True)
        print(f"  saved -> {outdir / f'confusedpairs_{name}.csv'}", flush=True)
        print(f"  top confusions (true -> guessed, count, same-superclass?):", flush=True)
        for (t, p), c, same in top[:12]:
            mark = "same-super" if same else "CROSS-super"
            print(f"    {t:>16} -> {p:<16} {c:4d}  [{mark}]", flush=True)

    # optionally push the two coarse confusion matrices + the figure to wandb
    if use_wandb:
        log_wandb_confusion(preds_full, preds_pc, test_labels, coarse_map, wandb_kwargs, png=png)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings_dir", required=True)
    p.add_argument("--strategy", default="probcover", choices=list(SELECTORS))
    p.add_argument("--delta", type=float, default=0.65)
    p.add_argument("--budget", type=int, default=100)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--split", default="train")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--topk", type=int, default=20, help="how many confused pairs to keep")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--project", default="applying_priors")
    p.add_argument("--group", default="confusion")
    args = p.parse_args()
    tag = "_".join(Path(args.embeddings_dir).parts[-3:])
    wandb_kwargs = dict(project=args.project, group=args.group,
                        name=f"{tag}_{args.strategy}_d{args.delta}",
                        config=dict(embeddings_dir=args.embeddings_dir, strategy=args.strategy,
                                    delta=args.delta, budget=args.budget, rounds=args.rounds))
    run(args.embeddings_dir, strategy=args.strategy, delta=args.delta,
        budget=args.budget, rounds=args.rounds, split=args.split,
        seed=args.seed, topk=args.topk, use_wandb=args.wandb, wandb_kwargs=wandb_kwargs)


if __name__ == "__main__":
    main()
