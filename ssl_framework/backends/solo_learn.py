"""Adapter between ssl_framework algorithms and the vendored solo-learn.

This is the ONLY module in ssl_framework that imports solo-learn. Everything
else (the algorithms, the label selectors) talks to solo-learn through the
functions here, so if the training backend is ever swapped this is the single
file to rewrite.

What it provides:
    run_phase(...)          -- train one solo-learn method (e.g. simclr, supcon)
                               for a number of epochs, optionally warm-started
                               from a previous backbone and optionally on a
                               labeled subset only. Returns the trained backbone
                               weights (and the live model).
    extract_embeddings(...) -- run a backbone over a dataset and return features.
    load_backbone_from_checkpoint(...) -- warm-start from a saved solo-learn run
                               (e.g. an existing SimCLR/BYOL checkpoint).
    build_val_loader(...)   -- classification val loader for online eval.
    save/load_backbone_state, train_pool_size, train_targets, set_seed -- helpers.

Two things solo-learn does not expose as config flags, and which this adapter
adds, are exactly what the SimCLR<->SupCon iteration needs:
    * warm-starting one method's backbone from another method's weights
      (its resume is full same-method state), and
    * training a supervised method on a chosen labeled subset of the pool
      (via a torch Subset over selected indices).
"""

import json
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

# Make the vendored solo-learn importable without installing it. Layout:
#   <repo>/ssl_framework/backends/solo_learn.py  ->  parents[2] == <repo>
#   <repo>/solo-learn
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOLO_ROOT = _REPO_ROOT / "solo-learn"
if str(_SOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOLO_ROOT))

# solo-learn imports (available once _SOLO_ROOT is on the path).
from omegaconf import OmegaConf  # noqa: E402
from lightning.pytorch import Trainer, seed_everything  # noqa: E402
from solo.args.pretrain import parse_cfg  # noqa: E402
from solo.methods import METHODS  # noqa: E402
from . import joint as _joint  # noqa: E402,F401  (registers "supcon_simclr" in METHODS)
from solo.data.pretrain_dataloader import (  # noqa: E402
    FullTransformPipeline,
    NCropAugmentation,
    build_transform_pipeline,
    prepare_datasets,
)

# Ready-made cifar configs shipped by solo-learn; we reuse their hyperparameters
# verbatim and only override dataset / epochs / paths.
_CFG_DIR = _SOLO_ROOT / "scripts" / "pretrain" / "cifar"

# solo-learn's cifar backbones are 32x32 resnet variants; resnet18 -> 512 dims.
DEFAULT_DATA_DIR = _REPO_ROOT / "data"


def set_seed(seed):
    """Seed everything once, at the start of an algorithm run."""
    seed_everything(seed, workers=True)


def _augmentations_file(cfg):
    """Pull the augmentations filename out of the config's hydra `defaults` list
    (e.g. simclr -> symmetric.yaml, byol -> asymmetric.yaml)."""
    for item in OmegaConf.to_container(cfg.defaults):
        if isinstance(item, dict) and "augmentations" in item:
            return item["augmentations"]
    raise ValueError("no 'augmentations' entry in config defaults")


def _build_cfg(method, dataset, epochs, data_dir, batch_size=None, *,
               enable_eval=False, knn_k=20, knn_distance="cosine",
               lr=None, classifier_lr=None, warmup_epochs=None,
               sup_weight=None, unsup_weight=None, finetune_blocks=None):
    """Load a solo-learn cifar config by hand (without the Hydra CLI) and apply
    our overrides, then run it through solo-learn's own parse_cfg so every
    derived field (num_classes, crop counts, LR scaling, ...) is filled exactly
    as a normal solo-learn run would.

    enable_eval turns on solo's native online evaluation: it points val_path at
    the dataset (so a classification val loader can be built) and switches on the
    online weighted-kNN, giving the same val_acc1/val_acc5/val_knn_acc1/
    val_knn_acc5 metrics a normal solo-learn pretraining logs. lr / classifier_lr
    / warmup_epochs override the shipped-from-scratch recipe for fine-tuning a
    warm-started backbone (fewer, gentler updates)."""
    # supcon_simclr has no shipped config of its own: reuse supcon's (same
    # backbone + projector), then re-tag the method and add the joint weights.
    cfg_name = "supcon" if method == "supcon_simclr" else method
    cfg = OmegaConf.load(_CFG_DIR / f"{cfg_name}.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.method = method
    if sup_weight is not None:
        cfg.method_kwargs.sup_weight = float(sup_weight)
    if unsup_weight is not None:
        cfg.method_kwargs.unsup_weight = float(unsup_weight)
    if finetune_blocks is not None:
        cfg.method_kwargs.finetune_blocks = int(finetune_blocks)

    # Inline the augmentations list that the defaults block would have pulled in,
    # then drop the Hydra machinery we are bypassing.
    cfg.augmentations = OmegaConf.load(_CFG_DIR / "augmentations" / _augmentations_file(cfg))
    for k in ("defaults", "hydra"):
        if k in cfg:
            del cfg[k]

    # W&B is wired straight into the Lightning Trainer as a logger object (see
    # run_phase), not through this cfg, so keep the cfg's own flag off.
    cfg.wandb = {"enabled": False, "entity": None, "project": "solo-learn", "offline": False}
    cfg.data.dataset = dataset
    cfg.data.train_path = str(data_dir)
    cfg.data.format = "image_folder"
    cfg.name = f"{method}-{dataset}"
    cfg.max_epochs = int(epochs)
    # devices must stay a list: parse_cfg scales LR by len(devices)*batch/256,
    # so [0] with batch 256 reproduces the shipped single-GPU learning rate.
    cfg.devices = [0]
    cfg.num_nodes = 1
    cfg.strategy = "auto"
    cfg.sync_batchnorm = False
    if batch_size is not None:
        cfg.optimizer.batch_size = int(batch_size)

    # Evaluation: with a val_path set, run_phase can build a classification val
    # loader so solo's validation_step computes val_acc1/5; knn_eval adds the
    # online weighted-kNN (val_knn_acc1/5). Without it, no val loader is passed
    # and neither is computed (the pre-existing SimCLR<->SupCon behaviour).
    if enable_eval:
        cfg.data.val_path = str(data_dir)
        cfg.knn_eval = {"enabled": True, "k": int(knn_k), "distance_func": knn_distance}
    else:
        cfg.data.val_path = None

    # Fine-tuning overrides: warm-starting a converged backbone wants a smaller
    # LR and shorter warmup than the shipped from-scratch recipe.
    if lr is not None:
        cfg.optimizer.lr = float(lr)
    if classifier_lr is not None:
        cfg.optimizer.classifier_lr = float(classifier_lr)
    if warmup_epochs is not None:
        cfg.scheduler.warmup_epochs = int(warmup_epochs)

    return parse_cfg(cfg)


def _build_transform(cfg, dataset):
    """Recreate solo-learn's multi-view augmentation pipeline for this config."""
    pipelines = [
        NCropAugmentation(build_transform_pipeline(dataset, aug_cfg), aug_cfg.num_crops)
        for aug_cfg in cfg.augmentations
    ]
    return FullTransformPipeline(pipelines)


class _LabelOverrideDataset(Dataset):
    """Wrap a pretrain dataset and swap the target for chosen pool indices.

    solo's pretrain dataset yields (pool_index, crops, target). SupCon trains on
    `target`, so to feed it kernel-propagated pseudo-labels we override the
    target for the pseudo-labeled indices while leaving genuinely-labeled points
    at their true target. `label_map` is keyed by the ORIGINAL pool index (the
    first element the dataset returns), so it composes correctly under a Subset.
    """

    def __init__(self, base, label_map):
        self.base = base
        self.label_map = {int(k): int(v) for k, v in label_map.items()}

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        index, *rest = self.base[i]
        target = rest[-1]
        if int(index) in self.label_map:
            rest[-1] = self.label_map[int(index)]
        return (index, *rest)


def run_phase(
    method,
    dataset,
    epochs,
    *,
    data_dir=DEFAULT_DATA_DIR,
    init_backbone_state=None,
    labeled_indices=None,
    label_overrides=None,
    batch_size=None,
    num_workers=4,
    precision="16-mixed",
    enable_eval=False,
    knn_k=20,
    knn_distance="cosine",
    val_every=1,
    lr=None,
    classifier_lr=None,
    warmup_epochs=None,
    sup_weight=None,
    unsup_weight=None,
    finetune_blocks=None,
    callbacks=None,
    logger=False,
):
    """Train one solo-learn method for `epochs` and return its backbone.

    Args:
        method: solo-learn method name with a cifar config (e.g. "simclr", "supcon").
        dataset: "cifar10" | "cifar100" (must have a copy under data_dir).
        epochs: number of epochs for this phase.
        data_dir: root holding the extracted torchvision dataset.
        init_backbone_state: optional state_dict to warm-start the backbone from
            (e.g. the previous phase's weights). Loaded strictly, so it must come
            from the same backbone architecture.
        labeled_indices: optional list of pool indices. If given, the method
            trains on that Subset only -- this is how a supervised phase (SupCon)
            sees just the labeled set while an unsupervised phase (SimCLR) uses
            the full pool.
        label_overrides: optional {pool_index: label} map. Points at these indices
            train with the given label instead of their true one -- used to add
            kernel-propagated pseudo-labels to the SupCon set. Indices here should
            also appear in labeled_indices (that is what puts them in the Subset).
        batch_size: override the config batch size (useful when the labeled
            subset is small).
        num_workers, precision: dataloader / trainer knobs.
        enable_eval: build a classification val loader and turn on solo's online
            evaluation so val_acc1/val_acc5 (and, with knn, val_knn_acc1/5) are
            computed and logged during training.
        knn_k, knn_distance: online weighted-kNN settings (only if enable_eval).
        val_every: run validation every N epochs (epochs on a small subset are
            tiny, so validating every one wastes time).
        lr, classifier_lr, warmup_epochs: fine-tuning overrides for the recipe.
        logger: a Lightning logger (e.g. WandbLogger) or False for none.

    Returns:
        dict with:
            backbone_state: CPU copy of the trained backbone's state_dict,
                            ready to warm-start the next phase.
            model:          the live LightningModule (its .backbone is on device).
    """
    cfg = _build_cfg(method, dataset, epochs, data_dir, batch_size,
                     enable_eval=enable_eval, knn_k=knn_k, knn_distance=knn_distance,
                     lr=lr, classifier_lr=classifier_lr, warmup_epochs=warmup_epochs,
                     sup_weight=sup_weight, unsup_weight=unsup_weight,
                     finetune_blocks=finetune_blocks)

    model = METHODS[method](cfg)
    if init_backbone_state is not None:
        model.backbone.load_state_dict(init_backbone_state, strict=True)

    transform = _build_transform(cfg, dataset)
    train_dataset = prepare_datasets(
        dataset,
        transform,
        train_data_path=str(data_dir),
        data_format=cfg.data.format,
        no_labels=False,
        data_fraction=-1.0,
    )
    if label_overrides:
        train_dataset = _LabelOverrideDataset(train_dataset, label_overrides)
    if labeled_indices is not None:
        train_dataset = Subset(train_dataset, list(labeled_indices))

    bs = cfg.optimizer.batch_size
    # A labeled subset at-or-below one batch would be dropped entirely by
    # drop_last; keep that (partial or single full) batch and shrink to fit.
    # Use strict '>' so n_train == bs keeps its one full batch instead of
    # yielding zero training steps.
    loader = DataLoader(
        train_dataset,
        batch_size=min(bs, len(train_dataset)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=len(train_dataset) > bs,
    )

    val_loader = build_val_loader(dataset, data_dir, batch_size=bs,
                                  num_workers=num_workers) if enable_eval else None

    use_gpu = torch.cuda.is_available()
    trainer = Trainer(
        max_epochs=int(epochs),
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        precision=precision if use_gpu else 32,
        strategy="auto",
        logger=logger,
        check_val_every_n_epoch=int(val_every),
        enable_checkpointing=False,
        enable_progress_bar=False,  # quiet per-epoch logging (matches run_pretrain)
        callbacks=callbacks,
    )
    trainer.fit(model, loader, val_loader)

    backbone_state = {k: v.detach().cpu().clone() for k, v in model.backbone.state_dict().items()}
    return {"backbone_state": backbone_state, "model": model}


def run_joint_phase(
    dataset,
    epochs,
    *,
    data_dir=DEFAULT_DATA_DIR,
    init_backbone_state=None,
    labeled_indices,
    sup_batch_size=256,
    unsup_batch_size=256,
    num_workers=4,
    precision="16-mixed",
    enable_eval=True,
    knn_k=20,
    knn_distance="cosine",
    val_every=1,
    lr=None,
    classifier_lr=None,
    warmup_epochs=None,
    sup_weight=1.0,
    unsup_weight=1.0,
    finetune_blocks=None,
    callbacks=None,
    logger=False,
):
    """Semi-supervised joint fine-tuning: SupCon on the labeled subset + SimCLR
    over the FULL unlabeled pool, combined each step by the supcon_simclr method.

    Two loaders run together (max_size_cycle): the labeled loader cycles while an
    epoch is one pass over the full pool, so the SimCLR term sees all data. Returns
    the same {backbone_state, model} dict as run_phase.
    """
    from lightning.pytorch.utilities.combined_loader import CombinedLoader

    cfg = _build_cfg("supcon_simclr", dataset, epochs, data_dir, unsup_batch_size,
                     enable_eval=enable_eval, knn_k=knn_k, knn_distance=knn_distance,
                     lr=lr, classifier_lr=classifier_lr, warmup_epochs=warmup_epochs,
                     sup_weight=sup_weight, unsup_weight=unsup_weight,
                     finetune_blocks=finetune_blocks)

    model = METHODS["supcon_simclr"](cfg)
    if init_backbone_state is not None:
        model.backbone.load_state_dict(init_backbone_state, strict=True)

    transform = _build_transform(cfg, dataset)
    full = prepare_datasets(dataset, transform, train_data_path=str(data_dir),
                            data_format=cfg.data.format, no_labels=False, data_fraction=-1.0)
    labeled = Subset(full, list(labeled_indices))

    sup_bs = min(sup_batch_size, len(labeled))
    sup_loader = DataLoader(labeled, batch_size=sup_bs, shuffle=True,
                            num_workers=num_workers, pin_memory=True,
                            drop_last=len(labeled) > sup_bs)
    unsup_loader = DataLoader(full, batch_size=unsup_batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    # max_size_cycle: epoch length = the (larger) pool loader; labeled loader cycles.
    combined = CombinedLoader({"sup": sup_loader, "unsup": unsup_loader},
                              mode="max_size_cycle")

    val_loader = build_val_loader(dataset, data_dir, batch_size=unsup_batch_size,
                                  num_workers=num_workers) if enable_eval else None

    use_gpu = torch.cuda.is_available()
    trainer = Trainer(
        max_epochs=int(epochs),
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        precision=precision if use_gpu else 32,
        strategy="auto",
        logger=logger,
        check_val_every_n_epoch=int(val_every),
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=callbacks,
    )
    trainer.fit(model, combined, val_loader)

    backbone_state = {k: v.detach().cpu().clone() for k, v in model.backbone.state_dict().items()}
    return {"backbone_state": backbone_state, "model": model}


def build_val_loader(dataset, data_dir=DEFAULT_DATA_DIR, *, batch_size=256, num_workers=4):
    """Classification val loader for online eval (val_acc1/5, online kNN)."""
    from solo.data.classification_dataloader import prepare_data

    _, val_loader = prepare_data(
        dataset,
        train_data_path=str(data_dir),
        val_data_path=str(data_dir),
        data_format="image_folder",
        batch_size=batch_size,
        num_workers=num_workers,
        auto_augment=False,
    )
    return val_loader


@torch.no_grad()
def extract_embeddings(backbone, dataset, data_dir=DEFAULT_DATA_DIR, *,
                       split="both", batch_size=256, num_workers=4):
    """Run a trained backbone over a dataset and return frozen features + labels.

    Returns a dict keyed by split ("train"/"val") mapping to (features, labels)
    numpy arrays, where features is (N, feat_dim). The projector/predictor heads
    are intentionally not used -- downstream tasks consume backbone features.

    Crucially, features come out in DATASET ORDER and COMPLETE: the loaders here
    use the *eval* transform (no random augmentation), shuffle=False and
    drop_last=False, so feature row i corresponds to pool index i -- the same
    index RandomSelector, train_targets, and the pretrain Subset use. (solo's own
    classification train loader shuffles, drops the last batch, and augments,
    which would silently misalign features from pool indices; we must not use it
    for extraction.)
    """
    from solo.data.classification_dataloader import prepare_datasets, prepare_transforms

    # Use the eval transform (T_val) for BOTH splits so train features are
    # deterministic, and build the datasets in native order.
    _, t_val = prepare_transforms(dataset)
    train_dataset, val_dataset = prepare_datasets(
        dataset, t_val, t_val,
        train_data_path=str(data_dir), val_data_path=str(data_dir),
        data_format="image_folder", download=False,
    )

    def ordered_loader(ds):
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True, drop_last=False)

    device = next(backbone.parameters()).device
    backbone.eval()

    def run(loader):
        feats, labels = [], []
        for x, y in loader:
            feats.append(backbone(x.to(device, non_blocking=True)).cpu())
            labels.append(y.clone())
        return torch.cat(feats).numpy(), torch.cat(labels).numpy()

    out = {}
    if split in ("train", "both"):
        out["train"] = run(ordered_loader(train_dataset))
    if split in ("val", "both"):
        out["val"] = run(ordered_loader(val_dataset))
    return out


def train_pool_size(dataset, data_dir=DEFAULT_DATA_DIR):
    """Number of training images in the pool (used to size the labeled set)."""
    import torchvision

    if dataset in ("cifar10", "cifar100"):
        cls = getattr(torchvision.datasets, dataset.upper())
        return len(cls(str(data_dir), train=True, download=False))
    raise NotImplementedError(f"train_pool_size not implemented for {dataset}")


def train_targets(dataset, data_dir=DEFAULT_DATA_DIR):
    """True training labels of the whole pool, index-aligned with the dataset.

    Used to (a) report the per-class makeup of a labeled set, and (b) recover the
    genuine labels of the pool for the SupCon subset. Order matches solo's
    image_folder / torchvision train ordering (same indices the Selector uses).
    """
    import numpy as np
    import torchvision

    if dataset in ("cifar10", "cifar100"):
        cls = getattr(torchvision.datasets, dataset.upper())
        return np.asarray(cls(str(data_dir), train=True, download=False).targets)
    raise NotImplementedError(f"train_targets not implemented for {dataset}")


def _resolve_checkpoint(path):
    """Accept a .ckpt file or a run dir; return (ckpt_path, cfg) for that run.

    Mirrors extract_embeddings.py: a run dir holds one-or-more '*-ep=<N>.ckpt'
    plus an args.json; we pick the highest epoch and read its config.
    """
    path = Path(path)
    if path.is_dir():
        ckpts = [f for f in path.iterdir() if f.suffix == ".ckpt"]
        if not ckpts:
            raise FileNotFoundError(f"no .ckpt found in {path}")
        ckpt_path = max(ckpts, key=lambda p: int(m.group(1)) if (m := re.search(r"ep=(\d+)", p.name)) else -1)
        cfg = OmegaConf.create(json.loads((path / "args.json").read_text()))
    else:
        ckpt_path = path
        cfg = OmegaConf.create(json.loads((path.parent / "args.json").read_text()))
    return ckpt_path, cfg


def checkpoint_method(path):
    """The solo-learn method a checkpoint was trained with (e.g. 'simclr', 'byol').

    Used to tag runs by their base encoder for cross-run plots.
    """
    _, cfg = _resolve_checkpoint(path)
    return str(cfg.method)


def load_backbone_from_checkpoint(path):
    """Load a solo-learn pretraining checkpoint and return its backbone module.

    `path` is a run dir (with args.json + *.ckpt) or a single .ckpt beside an
    args.json -- the layout produced by main_pretrain.py and consumed by
    extract_embeddings.py. Returns the live backbone nn.Module (heads dropped);
    call .state_dict() on it to warm-start run_phase via init_backbone_state.
    """
    ckpt_path, cfg = _resolve_checkpoint(path)
    return METHODS[cfg.method].load_from_checkpoint(str(ckpt_path), strict=False, cfg=cfg).backbone


def save_backbone_state(state, path):
    torch.save(state, str(path))


def load_backbone_state(path):
    return torch.load(str(path), map_location="cpu")
