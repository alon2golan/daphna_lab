# Jointly finetunes the SimCLR backbone with TWO linear heads -- one for the 20 superclasses,
# one for the 100 fine classes -- sharing a single trunk. Both losses backprop into the SAME
# backbone, so the coarse task reshapes the fine features. The heads are only a training signal;
# the caller keeps the finetuned backbone and throws the heads away.

import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# --- Capacity Constants (the knobs to sweep) ---

FINETUNE_EPOCHS = 30          # passes over the labeled subset (the UNFROZEN phase)
LR_BACKBONE = 1e-4            # backbone learning rate (ignored when FREEZE_BACKBONE is True)
LR_HEADS = 1e-3              # learning rate for the two linear heads
WEIGHT_DECAY = 1e-4          # L2 regularization on all trainable params
COARSE_LOSS_WEIGHT = 1.0     # weight of the 20-way (coarse) loss -- the only task here
FINE_LOSS_WEIGHT = 0.0       # 100-way (fine) loss OFF: this experiment finetunes with the coarse head only
FREEZE_BACKBONE = False      # True -> train heads only (linear-probe baseline, backbone unchanged)
TRAIN_BATCH_SIZE = 128       # minibatch size for the finetune loop
TRAIN_WORKERS = 4            # dataloader workers for the finetune loop

# --- Low-budget stabilizers (all default to a no-op = the original behavior) ---

FREEZE_BN = False            # keep BatchNorm in eval mode during finetune (freeze running stats)
HEAD_WARMUP_EPOCHS = 0       # LP-FT: epochs to train the HEAD ONLY (backbone frozen) before unfreezing
LR_SCHEDULE = "constant"     # "constant" or "cosine" backbone-lr schedule over the unfrozen phase
LR_WARMUP_EPOCHS = 0         # linear backbone-lr ramp 0->LR_BACKBONE over the first N unfrozen epochs


# --- Trainable Heads ---

def make_head(feature_dim, num_outputs):
    """A single linear classification head mapping backbone features to class logits."""
    return nn.Linear(feature_dim, num_outputs)


def set_backbone_trainable(backbone, trainable):
    """Flips requires_grad on every backbone parameter -- the freeze switch."""
    for parameter in backbone.parameters():
        parameter.requires_grad = trainable
    return backbone


def set_bn_eval(backbone):
    """Forces every BatchNorm layer into eval mode so it uses its PRETRAINED running mean/var
    and stops updating them. At tiny budgets a finetune batch is ~100 non-representative points,
    so letting BN re-estimate its stats on them corrupts the eval-time features -- this freezes
    that. Affine (weight/bias) params stay trainable; only the running-stat drift is stopped."""
    for module in backbone.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    return backbone


def backbone_lr_at(unfrozen_epoch, epochs, lr_backbone, lr_schedule, lr_warmup_epochs):
    """Backbone learning rate for one epoch of the UNFROZEN phase (0-indexed). 'constant' with no
    warmup reproduces the original fixed lr; 'cosine' ramps 0->lr_backbone over lr_warmup_epochs
    then cosine-decays to ~0 over the remaining unfrozen epochs."""
    if lr_schedule == "cosine" and unfrozen_epoch < lr_warmup_epochs:
        return lr_backbone * (unfrozen_epoch + 1) / max(1, lr_warmup_epochs)  # linear warmup ramp
    if lr_schedule == "cosine":
        span = max(1, epochs - lr_warmup_epochs)                              # epochs left to decay over
        progress = (unfrozen_epoch - lr_warmup_epochs) / span                 # 0 -> 1 across the decay
        return 0.5 * lr_backbone * (1.0 + math.cos(math.pi * progress))       # cosine lr_backbone -> 0
    if lr_schedule == "constant" and unfrozen_epoch < lr_warmup_epochs:
        return lr_backbone * (unfrozen_epoch + 1) / max(1, lr_warmup_epochs)  # optional warmup, then flat
    return lr_backbone                                                        # plain constant lr


# --- Granularity-Aware Dataset ---

class GranularityDataset(Dataset):
    """Wraps the augmented train dataset to serve only the labeled rows, each tagged with
    whether its FINE label is known. Coarse-labeled rows still return their fine label (so the
    superclass can be derived for the coarse head), but carry is_fine=False so the training
    loop skips them for the fine head."""

    def __init__(self, base_dataset, indices, is_fine_flags):
        self.base_dataset = base_dataset            # the T_train (augmented) CIFAR dataset
        self.indices = np.asarray(indices)          # which base rows this labeled set uses
        self.is_fine_flags = np.asarray(is_fine_flags)  # per-row: is the fine label known?

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        base_row = int(self.indices[position])           # map our position to the base row
        image, fine_label = self.base_dataset[base_row]  # augmented image + its (true) fine label
        is_fine = bool(self.is_fine_flags[position])     # whether to use it for the fine head
        return image, fine_label, is_fine


# --- Joint Finetune Loop ---

def finetune_two_heads(backbone, feature_dim, train_aug_dataset, labeled_set,
                       coarse_map, num_fine, num_coarse, device, seed=0,
                       epochs=FINETUNE_EPOCHS, lr_backbone=LR_BACKBONE,
                       freeze_bn=FREEZE_BN, head_warmup_epochs=HEAD_WARMUP_EPOCHS,
                       lr_schedule=LR_SCHEDULE, lr_warmup_epochs=LR_WARMUP_EPOCHS,
                       weight_decay=WEIGHT_DECAY):
    """Finetunes `backbone` in place on the labeled set with a coarse head and a fine head.
    Each head trains ONLY on its own group: the fine head on fine-labeled points, the coarse
    head on coarse-labeled points. A point shared between the groups is forwarded once per
    group, so it feeds each head at most once. `epochs` and `lr_backbone` default to the module
    constants but can be overridden per run (e.g. from the CLI). Returns the finetuned backbone.

    Low-budget stabilizers (all no-ops at their defaults, so the original run is unchanged):
      freeze_bn           -- keep BatchNorm running stats frozen throughout (eval-mode BN).
      head_warmup_epochs  -- LP-FT: this many EXTRA leading epochs train the head only (backbone
                             frozen) before the `epochs` unfrozen epochs, so the random head stops
                             distorting the pretrained trunk on step one.
      lr_schedule         -- "cosine" (with `lr_warmup_epochs` ramp) instead of a flat backbone lr.
    """
    torch.manual_seed(seed)  # reproducible head init + minibatch shuffling

    # Two heads on top of the shared trunk: 20-way coarse and 100-way fine
    coarse_head = make_head(feature_dim, num_coarse).to(device)
    fine_head = make_head(feature_dim, num_fine).to(device)

    # Two param groups so the backbone can move slower than the freshly-initialized heads.
    # When FREEZE_BACKBONE, the backbone group is empty and only the two heads get updated.
    # During the LP-FT warmup the backbone params are present but requires_grad=False, so Adam
    # simply skips them until we unfreeze -- no optimizer rebuild needed.
    optimizer = torch.optim.Adam(
        [
            {"params": list(coarse_head.parameters()) + list(fine_head.parameters()), "lr": LR_HEADS},
            {"params": [p for p in backbone.parameters()] if not FREEZE_BACKBONE else [], "lr": lr_backbone},
        ],
        weight_decay=weight_decay,
    )
    cross_entropy = nn.CrossEntropyLoss()  # standard multi-class loss for both heads

    # Combine the two groups into one training set, tagging each row with its granularity:
    # fine-labeled rows first (is_fine=True), then coarse-labeled rows (is_fine=False)
    combined_indices = np.concatenate([labeled_set.fine_indices, labeled_set.coarse_indices])
    is_fine_flags = np.concatenate([
        np.ones(labeled_set.num_fine, dtype=bool),    # fine-labeled -> feed the fine head
        np.zeros(labeled_set.num_coarse, dtype=bool),  # coarse-labeled -> coarse head only
    ])
    labeled_dataset = GranularityDataset(train_aug_dataset, combined_indices, is_fine_flags)
    loader = DataLoader(
        labeled_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
        num_workers=TRAIN_WORKERS, pin_memory=True, drop_last=False,
    )

    # coarse_map is a numpy array (fine id -> superclass id); move it to the device once for lookups
    coarse_lookup = torch.as_tensor(coarse_map, dtype=torch.long, device=device)

    coarse_head.train()
    fine_head.train()

    # Total loop = the LP-FT head-only warmup (backbone frozen) followed by the unfrozen finetune.
    total_epochs = head_warmup_epochs + epochs
    for epoch in range(total_epochs):
        in_warmup = epoch < head_warmup_epochs        # head-only (backbone frozen) phase?

        if in_warmup or FREEZE_BACKBONE:
            # Head-only: backbone stays pretrained-frozen (eval -> BN uses clean pretrained stats)
            set_backbone_trainable(backbone, False)
            backbone.eval()
        else:
            # Unfrozen finetune: backbone trains; optionally hold BN in eval to freeze its stats,
            # and set this epoch's backbone lr from the schedule (heads keep their fixed lr).
            set_backbone_trainable(backbone, True)
            backbone.train()
            if freeze_bn:
                set_bn_eval(backbone)
            unfrozen_epoch = epoch - head_warmup_epochs
            optimizer.param_groups[1]["lr"] = backbone_lr_at(
                unfrozen_epoch, epochs, lr_backbone, lr_schedule, lr_warmup_epochs)

        epoch_loss = 0.0  # accumulate for a per-epoch console line
        for images, fine_targets, is_fine in loader:
            images = images.to(device, non_blocking=True)          # the labeled minibatch
            fine_targets = fine_targets.to(device, non_blocking=True)  # each row's (true) fine label
            coarse_targets = coarse_lookup[fine_targets]           # its superclass (used for coarse rows only)
            is_fine = is_fine.to(device)                           # mask: fine-labeled rows
            is_coarse = ~is_fine                                   # the rest are coarse-labeled rows

            features = backbone(images)                            # ONE shared trunk forward

            # Fine head trains ONLY on the fine-labeled rows; skip the term if the batch has none
            if is_fine.any():
                fine_loss = cross_entropy(fine_head(features[is_fine]), fine_targets[is_fine])
            else:
                fine_loss = torch.zeros((), device=device)         # no fine supervision this batch

            # Coarse head trains ONLY on the coarse-labeled rows -- fine rows never touch it
            if is_coarse.any():
                coarse_loss = cross_entropy(coarse_head(features[is_coarse]), coarse_targets[is_coarse])
            else:
                coarse_loss = torch.zeros((), device=device)       # no coarse supervision this batch

            # Coarse is the primary task; the fine term is a small anti-collapse regularizer
            loss = COARSE_LOSS_WEIGHT * coarse_loss + FINE_LOSS_WEIGHT * fine_loss

            optimizer.zero_grad()  # clear last step's grads
            loss.backward()        # grads flow into heads AND (if unfrozen) the backbone
            optimizer.step()       # update all trainable params

            epoch_loss += float(loss.item())

        # One line per epoch: phase (LP head-only warmup vs unfrozen finetune) + mean minibatch loss
        phase = "warmup" if in_warmup else "finetune"
        print(f"  [{phase}] epoch {epoch + 1}/{total_epochs} loss={epoch_loss / len(loader):.4f}",
              flush=True)

    return backbone  # finetuned trunk; caller re-extracts features and discards the heads
