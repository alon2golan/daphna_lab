"""Joint SimCLR+SupCon fine-tuning of a frozen-pretrained encoder, swept over the
unsupervised weight.

Two conditions, scored on the SAME offline yardstick (linear probe + weighted kNN):

    baseline   the base SimCLR encoder as-is (the 1000ep checkpoint), no labels.
    joint      base encoder -> supcon_simclr on the LABELED set only (real labels),
               loss = sup_weight * L_supcon + unsup_weight * L_simclr, for one
               --unsup_weight. One run per weight in --unsup_weights, so the sweep
               shows how much the unsupervised (label-free) term helps.

Everything lands in its OWN wandb project (default semi-supervised-joint) and its
own results subtree, so it never mixes with the old supcon_finetune runs -- filter
by project, or by the group/tags/unsup_weight config.

    python -m ssl_framework.algorithms.joint_finetune \
        --dataset cifar100 \
        --base_checkpoint ../solo-learn/trained_models/simclr/cifar100/1000ep \
        --n_labeled 1000 --per_class_budget 10 --epochs 50 \
        --unsup_weights 1,2,5,10
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ..backends import solo_learn
from ..labeling import RandomSelector, class_counts
from .. import evaluation as ev

_NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


def _floats(s):
    return [float(x) for x in str(s).split(",") if str(x).strip() != ""]


def _g(x):
    return f"{float(x):g}"


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _make_logger(args, *, name, group, job_type, tags, save_dir):
    if not args.wandb:
        return None
    from lightning.pytorch.loggers import WandbLogger
    return WandbLogger(name=name, project=args.wandb_project, entity=args.wandb_entity,
                       group=group, job_type=job_type, tags=tags,
                       offline=args.wandb_offline, save_dir=str(save_dir), reinit=True)


def _log(logger, metrics):
    if logger is not None:
        logger.log_metrics(dict(metrics))


def _log_config(logger, cfg):
    if logger is not None:
        try:
            logger.experiment.config.update({k: v for k, v in cfg.items() if v is not None},
                                            allow_val_change=True)
        except Exception:
            pass


def _finish(logger):
    if logger is not None:
        try:
            logger.experiment.finish()
        except Exception:
            pass


from lightning.pytorch.callbacks import Callback  # noqa: E402


class _OfflineTrajEval(Callback):
    """Every `every` epochs, run the FULL offline scorecard (linear probe +
    weighted kNN over all extracted features) and log it on the epoch axis as
    offline_traj/*, so a mid-training peak is visible -- not just the final epoch.
    Also logs improvement_traj/* against the frozen baseline.
    """

    def __init__(self, dataset, data_dir, args, base_scores, every):
        super().__init__()
        self.dataset, self.data_dir, self.args = dataset, data_dir, args
        self.base_scores, self.every = base_scores, int(every)

    def on_train_epoch_end(self, trainer, pl_module):
        ep = trainer.current_epoch
        # fire every `every` epochs (1-indexed so epoch 0 isn't a wasted probe)
        if self.every <= 0 or (ep + 1) % self.every:
            return
        was_training = pl_module.training
        scores, _ = _evaluate_backbone(pl_module.backbone, self.dataset, self.data_dir,
                                       self.args.num_workers, self.args)
        metrics = {f"offline_traj/{k}": v for k, v in scores.items()}
        metrics.update({f"improvement_traj/{k}": scores[k] - self.base_scores[k] for k in scores})
        metrics["epoch"] = ep
        trainer.logger.log_metrics(metrics, step=trainer.global_step)
        pl_module.backbone.to(pl_module.device)
        if was_training:
            pl_module.backbone.train()


def _evaluate_backbone(backbone, dataset, data_dir, num_workers, args):
    import torch
    backbone = backbone.to("cuda" if torch.cuda.is_available() else "cpu")
    emb = solo_learn.extract_embeddings(backbone, dataset, data_dir, num_workers=num_workers)
    (xtr, ytr), (xva, yva) = emb["train"], emb["val"]
    scores = ev.evaluate_features(xtr, ytr, xva, yva, knn_k=args.knn_k,
                                  knn_distance=args.knn_distance, probe_epochs=args.probe_epochs)
    return scores, emb


def run(args):
    solo_learn.set_seed(args.seed)
    num_classes = _NUM_CLASSES[args.dataset]
    data_dir = args.data_dir
    base_method = solo_learn.checkpoint_method(args.base_checkpoint)  # expect simclr
    case = Path(args.results_root) / args.dataset / base_method / "joint"
    group = f"{base_method}-{args.dataset}-joint"
    print(f"case={case} base={base_method} project={args.wandb_project}")

    base_backbone = solo_learn.load_backbone_from_checkpoint(args.base_checkpoint)
    base_state = {k: v.detach().cpu().clone() for k, v in base_backbone.state_dict().items()}

    # ---- baseline: score the frozen 1000ep encoder once (reference line) ----- #
    base_scores, _ = _evaluate_backbone(base_backbone, args.dataset, data_dir,
                                        args.num_workers, args)
    print(f"[baseline] {base_scores}")
    _write_json(case / "baseline_scores.json",
                {"base_method": base_method, "dataset": args.dataset, "offline": base_scores})
    ref_budgets = [int(x) for x in str(args.reference_budgets).split(",") if x.strip()] \
        or [args.per_class_budget]
    blog = _make_logger(args, name=f"{group}-baseline", group=group, job_type="baseline",
                        tags=[base_method, args.dataset, "baseline"], save_dir=case)
    _log_config(blog, {"base_method": base_method, "dataset": args.dataset, "condition": "baseline"})
    for L in ref_budgets:  # flat reference across the budget axis
        _log(blog, {**{f"offline/{k}": v for k, v in base_scores.items()}, "per_class_budget": L})
    _finish(blog)

    # ---- labeled set (real labels only, seeded) ------------------------------ #
    n_pool = solo_learn.train_pool_size(args.dataset, data_dir)
    targets = solo_learn.train_targets(args.dataset, data_dir)
    labeled = RandomSelector().select(n_pool, args.n_labeled, args.seed)
    counts = class_counts(labeled, targets, num_classes)
    print(f"pool={n_pool} labeled={len(labeled)} L={args.per_class_budget} "
          f"| empty classes={int((counts == 0).sum())}")

    id_cfg = {"base_method": base_method, "dataset": args.dataset, "condition": "joint",
              "per_class_budget": args.per_class_budget, "n_labeled": args.n_labeled,
              "seed": args.seed, "epochs": args.epochs, "sup_weight": args.sup_weight,
              "unsup_source": args.unsup_source, "lr": args.lr,
              "finetune_blocks": args.finetune_blocks}

    # ---- joint condition, one run per unsupervised weight -------------------- #
    for w in _floats(args.unsup_weights):
        leaf = case / f"w{_g(w)}" / f"L{args.per_class_budget}_s{args.seed}"
        leaf.mkdir(parents=True, exist_ok=True)
        name = f"{group}-w{_g(w)}-L{args.per_class_budget}-s{args.seed}"
        logger = _make_logger(args, name=name, group=group, job_type="joint",
                              tags=[base_method, args.dataset, "joint", f"w{_g(w)}"],
                              save_dir=leaf)
        _log_config(logger, {**id_cfg, "unsup_weight": w})

        val_every = args.val_every or max(1, args.epochs // 20)
        traj = [_OfflineTrajEval(args.dataset, data_dir, args, base_scores, args.offline_every)] \
            if args.offline_every > 0 else None
        if args.unsup_source == "unlabeled":
            # SupCon on the labeled batch + SimCLR over the FULL pool each step.
            print(f"[joint w={_g(w)}] SEMI: supcon(labeled={len(labeled)}) + "
                  f"simclr(pool={n_pool}) | epochs={args.epochs} sup_w={args.sup_weight} unsup_w={w}")
            res = solo_learn.run_joint_phase(
                args.dataset, args.epochs, data_dir=data_dir, init_backbone_state=base_state,
                labeled_indices=labeled, sup_batch_size=min(args.batch_size, len(labeled)),
                unsup_batch_size=args.batch_size, num_workers=args.num_workers,
                enable_eval=True, knn_k=args.knn_k, knn_distance=args.knn_distance,
                val_every=val_every, lr=args.lr, classifier_lr=args.classifier_lr,
                sup_weight=args.sup_weight, unsup_weight=w, callbacks=traj,
                finetune_blocks=args.finetune_blocks, logger=logger)
        else:
            # Both terms on the same labeled batch (SimCLR as a regulariser).
            bs = min(args.batch_size, len(labeled))
            print(f"[joint w={_g(w)}] LABELED-ONLY: supcon_simclr on {len(labeled)} images | "
                  f"batch={bs} epochs={args.epochs} sup_w={args.sup_weight} unsup_w={w}")
            res = solo_learn.run_phase(
                "supcon_simclr", args.dataset, args.epochs,
                data_dir=data_dir, init_backbone_state=base_state,
                labeled_indices=labeled, batch_size=bs, num_workers=args.num_workers,
                enable_eval=True, knn_k=args.knn_k, knn_distance=args.knn_distance,
                val_every=val_every, lr=args.lr, classifier_lr=args.classifier_lr,
                sup_weight=args.sup_weight, unsup_weight=w, callbacks=traj,
                finetune_blocks=args.finetune_blocks, logger=logger)

        scores, _ = _evaluate_backbone(res["model"].backbone, args.dataset, data_dir,
                                       args.num_workers, args)
        print(f"[joint w={_g(w)}] {scores}")
        metrics = {**{f"offline/{k}": v for k, v in scores.items()},
                   **{f"baseline/{k}": base_scores[k] for k in scores},
                   **{f"improvement/{k}": scores[k] - base_scores[k] for k in scores},
                   "per_class_budget": args.per_class_budget, "unsup_weight": w}
        _log(logger, metrics)
        _finish(logger)
        solo_learn.save_backbone_state(res["backbone_state"], leaf / "backbone.pt")
        _write_json(leaf / "scores.json",
                    {**id_cfg, "unsup_weight": w, "offline": scores, "baseline": base_scores})

    print("done ->", case)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="cifar100", choices=list(_NUM_CLASSES))
    p.add_argument("--base_checkpoint", required=True,
                   help="the pretrained SimCLR run dir / .ckpt (e.g. .../simclr/cifar100/1000ep)")
    p.add_argument("--results_root",
                   default=str(Path(__file__).resolve().parents[1] / "results"))
    p.add_argument("--data_dir", default=str(solo_learn.DEFAULT_DATA_DIR))
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_labeled", type=int, required=True,
                   help="total labeled-set size (real labels only)")
    p.add_argument("--per_class_budget", type=int, required=True,
                   help="labels-per-class L (metadata / x-axis; n_labeled = L * num_classes)")
    p.add_argument("--seed", type=int, default=1)
    # joint fine-tuning
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.01,
                   help="fine-tune LR. Low by design: 0.1 badly forgets the strong "
                        "pretrained encoder; 0.01/0.001 fine-tunes gently")
    p.add_argument("--finetune_blocks", type=int, default=-1,
                   help="how many of the last resnet stages stay trainable (anti-forgetting): "
                        "-1 all (full fine-tune), k trains only the last k of layer1..4, "
                        "0 fully frozen (control -> backbone features == baseline)")
    p.add_argument("--classifier_lr", type=float, default=0.1)
    p.add_argument("--val_every", type=int, default=0, help="0 = auto (~20 points)")
    p.add_argument("--unsup_source", default="unlabeled", choices=["unlabeled", "labeled"],
                   help="unlabeled: SimCLR term over the FULL pool (true semi-supervised); "
                        "labeled: SimCLR term on the labeled batch only (regulariser)")
    p.add_argument("--sup_weight", type=float, default=1.0,
                   help="fixed weight on the SupCon term")
    p.add_argument("--unsup_weights", default="1,2,5,10",
                   help="comma list of weights on the SimCLR (unsupervised) term; one run each. "
                        "Larger = more emphasis on the unsupervised regulariser")
    # evaluation
    p.add_argument("--knn_k", type=int, default=20)
    p.add_argument("--knn_distance", default="cosine", choices=["cosine", "euclidean"])
    p.add_argument("--probe_epochs", type=int, default=100)
    p.add_argument("--offline_every", type=int, default=5,
                   help="run the full offline scorecard every N epochs and log it on the "
                        "epoch axis (offline_traj/*, improvement_traj/*) so a mid-training "
                        "peak is visible; 0 disables. Each probe extracts all features + "
                        "trains a linear probe, so keep N modest")
    p.add_argument("--reference_budgets", default="",
                   help="budgets L at which to log the baseline flat line (default: this budget)")
    # wandb -- SEPARATE project so it never mixes with supcon_finetune
    p.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    p.add_argument("--no_wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb_project", default="semi-supervised-joint")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_offline", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
