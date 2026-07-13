"""Adapter between semi_supervised algorithms and the vendored solo-learn.

This is the ONLY module in semi_supervised that imports solo-learn. Everything
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
    save/load_backbone_state, train_pool_size, set_seed -- small helpers.

Two things solo-learn does not expose as config flags, and which this adapter
adds, are exactly what the SimCLR<->SupCon iteration needs:
    * warm-starting one method's backbone from another method's weights
      (its resume is full same-method state), and
    * training a supervised method on a chosen labeled subset of the pool
      (via a torch Subset over selected indices).
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

# Make the vendored solo-learn importable without installing it. Layout:
#   <repo>/semi_supervised/backends/solo_learn.py  ->  parents[2] == <repo>
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


def _build_cfg(method, dataset, epochs, data_dir, batch_size=None):
    """Load a solo-learn cifar config by hand (without the Hydra CLI) and apply
    our overrides, then run it through solo-learn's own parse_cfg so every
    derived field (num_classes, crop counts, LR scaling, ...) is filled exactly
    as a normal solo-learn run would."""
    cfg = OmegaConf.load(_CFG_DIR / f"{method}.yaml")
    OmegaConf.set_struct(cfg, False)

    # Inline the augmentations list that the defaults block would have pulled in,
    # then drop the Hydra machinery we are bypassing.
    cfg.augmentations = OmegaConf.load(_CFG_DIR / "augmentations" / _augmentations_file(cfg))
    for k in ("defaults", "hydra"):
        if k in cfg:
            del cfg[k]

    # No W&B, no separate classifier-val loader (we only need the train loader).
    cfg.wandb = {"enabled": False, "entity": None, "project": "solo-learn", "offline": False}
    cfg.data.dataset = dataset
    cfg.data.train_path = str(data_dir)
    cfg.data.val_path = None
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

    return parse_cfg(cfg)


def _build_transform(cfg, dataset):
    """Recreate solo-learn's multi-view augmentation pipeline for this config."""
    pipelines = [
        NCropAugmentation(build_transform_pipeline(dataset, aug_cfg), aug_cfg.num_crops)
        for aug_cfg in cfg.augmentations
    ]
    return FullTransformPipeline(pipelines)


def run_phase(
    method,
    dataset,
    epochs,
    *,
    data_dir=DEFAULT_DATA_DIR,
    init_backbone_state=None,
    labeled_indices=None,
    batch_size=None,
    num_workers=4,
    precision="16-mixed",
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
        batch_size: override the config batch size (useful when the labeled
            subset is small).
        num_workers, precision: dataloader / trainer knobs.

    Returns:
        dict with:
            backbone_state: CPU copy of the trained backbone's state_dict,
                            ready to warm-start the next phase.
            model:          the live LightningModule (its .backbone is on device).
    """
    cfg = _build_cfg(method, dataset, epochs, data_dir, batch_size)

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
    if labeled_indices is not None:
        train_dataset = Subset(train_dataset, list(labeled_indices))

    bs = cfg.optimizer.batch_size
    # A labeled subset smaller than the batch would be dropped entirely by
    # drop_last; keep the partial batch and shrink the batch to fit.
    loader = DataLoader(
        train_dataset,
        batch_size=min(bs, len(train_dataset)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=len(train_dataset) >= bs,
    )

    use_gpu = torch.cuda.is_available()
    trainer = Trainer(
        max_epochs=int(epochs),
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        precision=precision if use_gpu else 32,
        strategy="auto",
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, loader)

    backbone_state = {k: v.detach().cpu().clone() for k, v in model.backbone.state_dict().items()}
    return {"backbone_state": backbone_state, "model": model}


@torch.no_grad()
def extract_embeddings(backbone, dataset, data_dir=DEFAULT_DATA_DIR, *,
                       split="both", batch_size=256, num_workers=4):
    """Run a trained backbone over a dataset and return frozen features + labels.

    Returns a dict keyed by split ("train"/"val") mapping to (features, labels)
    numpy arrays, where features is (N, feat_dim). The projector/predictor heads
    are intentionally not used -- downstream tasks consume backbone features.
    """
    from solo.data.classification_dataloader import prepare_data

    train_loader, val_loader = prepare_data(
        dataset,
        train_data_path=str(data_dir),
        val_data_path=str(data_dir),
        data_format="image_folder",
        batch_size=batch_size,
        num_workers=num_workers,
        auto_augment=False,
    )

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
        out["train"] = run(train_loader)
    if split in ("val", "both"):
        out["val"] = run(val_loader)
    return out


def train_pool_size(dataset, data_dir=DEFAULT_DATA_DIR):
    """Number of training images in the pool (used to size the labeled set)."""
    import torchvision

    if dataset in ("cifar10", "cifar100"):
        cls = getattr(torchvision.datasets, dataset.upper())
        return len(cls(str(data_dir), train=True, download=False))
    raise NotImplementedError(f"train_pool_size not implemented for {dataset}")


def save_backbone_state(state, path):
    torch.save(state, str(path))


def load_backbone_state(path):
    return torch.load(str(path), map_location="cpu")
