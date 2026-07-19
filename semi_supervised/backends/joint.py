"""Joint SimCLR + SupCon fine-tuning as a single solo-learn method.

Both terms are the SAME NT-Xent contrastive loss on the SAME projected features;
only the grouping of positives differs:
    supervised   (SupCon)  -- crops are positives if they share a TRUE label.
    unsupervised (SimCLR)  -- crops are positives only if they are augmentations
                             of the SAME image (labels never used).
so the SimCLR term is a label-free regulariser on the supervised pass:

    loss = sup_weight * L_supcon + unsup_weight * L_simclr + online_class_loss

Sweep unsup_weight upward to lean on the unsupervised regulariser (the point of
the experiment: how much unsupervised signal helps a tiny labeled set).

Registered into solo's METHODS as "supcon_simclr" on import, so run_phase can
build it exactly like any shipped method.
"""

from typing import Any, Sequence

import torch
from solo.losses.simclr import simclr_loss_func
from solo.methods import METHODS
from solo.methods.base import BaseMethod
from solo.methods.supcon import SupCon


class SupConSimCLR(SupCon):
    """SupCon with an added SimCLR (per-image) NT-Xent term on the same z."""

    # resnet stages, shallow -> deep; freezing keeps only the last N trainable.
    _STAGES = ("layer1", "layer2", "layer3", "layer4")

    def __init__(self, cfg):
        super().__init__(cfg)
        mk = cfg.method_kwargs
        self.sup_weight = float(mk.sup_weight) if "sup_weight" in mk else 1.0
        self.unsup_weight = float(mk.unsup_weight) if "unsup_weight" in mk else 1.0
        # finetune_blocks: how many of the last resnet stages stay trainable.
        #   <0  -> all trainable (default, full fine-tune)
        #    0  -> backbone fully frozen (control; backbone features == baseline)
        #    k  -> only the last k stages adapt (anti-forgetting partial fine-tune)
        ft = int(mk.finetune_blocks) if "finetune_blocks" in mk else -1
        self._apply_backbone_freeze(ft)

    def _apply_backbone_freeze(self, finetune_blocks):
        if finetune_blocks is None or finetune_blocks < 0:
            return  # all trainable
        trainable = set(self._STAGES[len(self._STAGES) - finetune_blocks:]) \
            if finetune_blocks > 0 else set()
        n_frozen = 0
        for name, p in self.backbone.named_parameters():
            keep = name.split(".")[0] in trainable
            p.requires_grad = keep
            n_frozen += (not keep)
        print(f"[supcon_simclr] finetune_blocks={finetune_blocks} -> trainable stages "
              f"{sorted(trainable)}, froze {n_frozen} backbone tensors")

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        # Two shapes are accepted:
        #   dict {"sup": labeled_batch, "unsup": pool_batch}  -- SEMI-supervised:
        #       SupCon on the labeled batch, SimCLR over the full unlabeled pool.
        #   plain [idx, [crops], targets]                     -- labeled-only:
        #       both terms on the same labeled batch (SimCLR as a regulariser).
        if isinstance(batch, dict):
            return self._semi_step(batch, batch_idx)

        indexes, targets = batch[0], batch[-1]
        # BaseMethod does the forward for every crop + the online classifier loss;
        # skip SupCon.training_step so its single-term loss is not added twice.
        out = BaseMethod.training_step(self, batch, batch_idx)
        class_loss = out["loss"]
        z = torch.cat(out["z"])
        n_augs = self.num_large_crops + self.num_small_crops

        sup_loss = simclr_loss_func(z, indexes=targets.repeat(n_augs),
                                    temperature=self.temperature)
        unsup_loss = simclr_loss_func(z, indexes=indexes.repeat(n_augs),
                                      temperature=self.temperature)
        joint = self.sup_weight * sup_loss + self.unsup_weight * unsup_loss

        self.log_dict({"train_sup_loss": sup_loss, "train_unsup_loss": unsup_loss,
                       "train_joint_loss": joint}, on_epoch=True, sync_dist=True)
        return joint + class_loss

    def _semi_step(self, batch, batch_idx):
        sup_batch, unsup_batch = batch["sup"], batch["unsup"]

        # Supervised pass on the labeled batch. BaseMethod also trains + logs the
        # online classifier (val_acc*) on these REAL labels.
        sup_out = BaseMethod.training_step(self, sup_batch, batch_idx)
        class_loss = sup_out["loss"]
        z_sup = torch.cat(sup_out["z"])
        y = sup_batch[-1].repeat(self.num_large_crops + self.num_small_crops)
        sup_loss = simclr_loss_func(z_sup, indexes=y, temperature=self.temperature)

        # Unsupervised pass over the full-pool batch: per-image positives, labels
        # never touched. Forward the crops through backbone+projector directly.
        idx_u, crops_u = unsup_batch[0], unsup_batch[1]
        z_unsup = torch.cat([self.projector(self.backbone(x)) for x in crops_u])
        idx_u = idx_u.repeat(len(crops_u))
        unsup_loss = simclr_loss_func(z_unsup, indexes=idx_u, temperature=self.temperature)

        joint = self.sup_weight * sup_loss + self.unsup_weight * unsup_loss
        self.log_dict({"train_sup_loss": sup_loss, "train_unsup_loss": unsup_loss,
                       "train_joint_loss": joint}, on_epoch=True, sync_dist=True)
        return joint + class_loss


METHODS["supcon_simclr"] = SupConSimCLR
