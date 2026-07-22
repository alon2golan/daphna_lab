# Jointly finetunes the SimCLR backbone with TWO linear heads -- one for the 20 superclasses,
# one for the 100 fine classes -- sharing a single trunk. Both losses backprop into the SAME
# backbone, so the coarse task reshapes the fine features. The heads are only a training signal;
# the caller keeps the finetuned backbone and throws the heads away.

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# --- Capacity Constants (the knobs to sweep) ---

FINETUNE_EPOCHS = 30          # passes over the labeled subset
LR_BACKBONE = 1e-4            # backbone learning rate (ignored when FREEZE_BACKBONE is True)
LR_HEADS = 1e-3              # learning rate for the two linear heads
WEIGHT_DECAY = 1e-4          # L2 regularization on all trainable params
COARSE_LOSS_WEIGHT = 1.0     # weight of the 20-way (coarse) loss -- the only task here
FINE_LOSS_WEIGHT = 0.0       # 100-way (fine) loss OFF: this experiment finetunes with the coarse head only
FREEZE_BACKBONE = False      # True -> train heads only (linear-probe baseline, backbone unchanged)
TRAIN_BATCH_SIZE = 128       # minibatch size for the finetune loop
TRAIN_WORKERS = 4            # dataloader workers for the finetune loop


# --- Trainable Heads ---

def make_head(feature_dim, num_outputs):
    """A single linear classification head mapping backbone features to class logits."""
    return nn.Linear(feature_dim, num_outputs)


def set_backbone_trainable(backbone, trainable):
    """Flips requires_grad on every backbone parameter -- the freeze switch."""
    for parameter in backbone.parameters():
        parameter.requires_grad = trainable
    return backbone


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
                       coarse_map, num_fine, num_coarse, device, seed=0):
    """Finetunes `backbone` in place on the labeled set with a coarse head and a fine head.
    Each head trains ONLY on its own group: the fine head on fine-labeled points, the coarse
    head on coarse-labeled points. A point shared between the groups is forwarded once per
    group, so it feeds each head at most once. Returns the finetuned backbone (heads dropped)."""
    torch.manual_seed(seed)  # reproducible head init + minibatch shuffling

    # Two heads on top of the shared trunk: 20-way coarse and 100-way fine
    coarse_head = make_head(feature_dim, num_coarse).to(device)
    fine_head = make_head(feature_dim, num_fine).to(device)

    # Freeze or unfreeze the backbone according to the constant
    set_backbone_trainable(backbone, not FREEZE_BACKBONE)

    # Two param groups so the backbone can move slower than the freshly-initialized heads.
    # When frozen, the backbone group is empty and only the two heads get updated.
    optimizer = torch.optim.Adam(
        [
            {"params": list(coarse_head.parameters()) + list(fine_head.parameters()), "lr": LR_HEADS},
            {"params": [p for p in backbone.parameters()] if not FREEZE_BACKBONE else [], "lr": LR_BACKBONE},
        ],
        weight_decay=WEIGHT_DECAY,
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

    # backbone in train() only when it is being updated; heads always train()
    backbone.train() if not FREEZE_BACKBONE else backbone.eval()
    coarse_head.train()
    fine_head.train()

    for epoch in range(FINETUNE_EPOCHS):
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

        # One line per epoch: mean minibatch loss, so a run's progress is visible in logs
        print(f"  [finetune] epoch {epoch + 1}/{FINETUNE_EPOCHS} loss={epoch_loss / len(loader):.4f}",
              flush=True)

    return backbone  # finetuned trunk; caller re-extracts features and discards the heads
