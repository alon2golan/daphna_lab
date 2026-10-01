"""Adapted from Itai's train_mlp_with_distillation and distillation_loss.

See PROVENANCE.md: same MLP/Adam, corrected per-sample reverse KL, balanced
batches, final checkpoint and no early stopping or validation selection.
"""
import json
import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

from .upstream import MLPClassifierTorch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def balanced_batches(labeled, admitted, batch_size, rng):
    labeled, admitted = np.asarray(labeled), np.asarray(admitted)
    if batch_size < 2 or not len(labeled):
        raise ValueError("Need labeled examples and batch_size >= 2")
    if not len(admitted):
        shuffled = rng.permutation(labeled)
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start:start + batch_size]
            yield batch, np.ones(len(batch), dtype=bool)
        return
    quota = max(1, batch_size // 8)
    pool = rng.permutation(admitted)
    cycle, pos = rng.permutation(labeled), 0
    for start in range(0, len(pool), batch_size - quota):
        selected = []
        for _ in range(quota):
            if pos == len(cycle):
                cycle, pos = rng.permutation(labeled), 0
            selected.append(cycle[pos])
            pos += 1
        unlabeled = pool[start:start + batch_size - quota]
        batch = np.r_[selected, unlabeled].astype(np.int64)
        mask = np.arange(len(batch)) < quota
        order = rng.permutation(len(batch))
        yield batch[order], mask[order]


def distillation_loss(logits, targets, confidence):
    """Paper KL(model || evidence), confidence multiplied BEFORE reduction."""
    if len(logits) == 0:
        return logits.sum() * 0.0
    log_p = F.log_softmax(logits, dim=1)
    # Positive Dirichlet priors guarantee positive targets. Float64 prevents
    # conversion of a very small but valid probability into a float32 zero.
    target = targets.to(dtype=torch.float64)
    if not torch.isfinite(target).all() or torch.any(target <= 0):
        raise ValueError("Soft targets must be finite and strictly positive")
    per_sample = (log_p.exp().double() * (log_p.double() - target.log())).sum(dim=1)
    return (per_sample * confidence.double()).mean()


@torch.no_grad()
def predict(model, features, device, batch_size=4096):
    model.eval()
    predictions = []
    for start in range(0, len(features), batch_size):
        batch = torch.tensor(np.asarray(features[start:start + batch_size]), dtype=torch.float32, device=device)
        predictions.append(model(batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def predict_checkpoint(path, features, device="cpu"):
    """Inductive inference on RAW frozen features; no graph/test fitting."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = MLPClassifierTorch(checkpoint["input_dim"], 256, checkpoint["classes"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    if features.shape[1] != checkpoint["input_dim"]:
        raise ValueError("Feature width differs from checkpoint")
    return predict(model, features, device)


def train_classifier(features, labeled, labeled_labels, classes, targets, confidence,
                     admitted, config, output):
    seed_everything(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    model = MLPClassifierTorch(features.shape[1], 256, classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.03)
    # Never pass the unrevealed ground-truth labels into training.
    revealed = np.full(len(features), -1, dtype=np.int64)
    revealed[labeled] = labeled_labels
    rng = np.random.RandomState(config["seed"])
    with (output / "training.jsonl").open("w") as log:
        for epoch in range(config["epochs"]):
            model.train()
            total_loss, steps = 0.0, 0
            for indices, is_labeled in balanced_batches(labeled, np.flatnonzero(admitted), config["batch_size"], rng):
                x = torch.tensor(np.asarray(features[indices]), dtype=torch.float32, device=device)
                mask = torch.tensor(is_labeled, device=device)
                logits = model(x)
                labels = torch.tensor(revealed[indices[is_labeled]], dtype=torch.long, device=device)
                ce = F.cross_entropy(logits[mask], labels)
                target = torch.tensor(targets[indices[~is_labeled]], dtype=torch.float64, device=device)
                weights = torch.tensor(confidence[indices[~is_labeled]], dtype=torch.float64, device=device)
                kd = distillation_loss(logits[~mask], target, weights)
                loss = ce + config["distillation_weight"] * kd
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1
            row = {"epoch": epoch + 1, "mean_batch_loss": total_loss / steps, "steps": steps}
            log.write(json.dumps(row) + "\n")
            log.flush()
            if epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == config["epochs"]:
                print(json.dumps(row), flush=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "input_dim": features.shape[1], "classes": classes, "hidden_dim": 256,
                "epoch": config["epochs"], "seed": config["seed"], "classifier_input": "raw",
                "propagation": config["propagation"], "confirmed_reproduction": False},
               output / "final.pt")
    return model, device
