"""Configurable lightweight PyTorch task heads and training helpers."""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn.functional as F
import random
from .task import Task

def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def build_head(input_dim: int, output_dim: int, config: dict, device: str):
    layers = []
    width = input_dim
    for hidden in config.get("hidden_sizes", []):
        layers.extend([torch.nn.Linear(width, int(hidden)), torch.nn.ReLU()])
        dropout = float(config.get("dropout", 0.0))
        if dropout: layers.append(torch.nn.Dropout(dropout))
        width = int(hidden)
    layers.append(torch.nn.Linear(width, output_dim))
    return torch.nn.Sequential(*layers).to(device)


def _loss(logits, targets, objective: dict):
    try:
        name = objective["name"]
    except:
        raise ValueError("No loss was specified under objective: name key does not exist in objective")
    if name == "cross_entropy":
        return F.cross_entropy(logits, targets.long(), reduction=objective.get("reduction", "mean"))
    if name == "bce_with_logits":
        return F.binary_cross_entropy_with_logits(logits, targets.float(), reduction=objective.get("reduction", "mean"))
    if name == "focal_bce":
        raw = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        prob = logits.sigmoid()
        pt = targets * prob + (1 - targets) * (1 - prob)
        value = raw * (1 - pt).pow(float(objective.get("gamma", 2.0)))
        return value.mean() if objective.get("reduction", "mean") == "mean" else value.sum()
    raise ValueError(f"unregistered objective: {name}")


def fit_head(features: np.ndarray, targets: np.ndarray, output_dim: int, objective: dict,
             config: dict, device: str, seed: int, shadow: bool = False):
    model = build_head(features.shape[1], output_dim, config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-3)),
                                  weight_decay=float(config.get("weight_decay", 1e-4)))
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    y = torch.as_tensor(targets, device=device)
    batch_size = int(config.get("batch_size", 128))
    epochs = int(config.get("epochs", 20))
    dropout = float(config.get("dropout", 0))
    long_shadow = copy.deepcopy(model).eval() if shadow else None
    short_shadow = copy.deepcopy(model).eval() if shadow else None
    alpha_long = float(config.get("cls_alpha_long", 0.9999))
    alpha_short = float(config.get("cls_alpha_short", 0.85))
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(x), device=device)
        for start in range(0, len(x), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model(x[batch]), y[batch], objective)
            loss.backward(); optimizer.step()
        if shadow:
            with torch.no_grad():
                for shadow_model, alpha in ((long_shadow, alpha_long), (short_shadow, alpha_short)):
                    for shadow_p, p in zip(shadow_model.parameters(), model.parameters()):
                        shadow_p.mul_(alpha).add_(p, alpha=1 - alpha)
    return model, long_shadow, short_shadow


def calc_probs(logits: torch.Tensor, task: Task | str):
    return torch.softmax(logits, dim=1) if Task(task) == Task.multiclass else torch.sigmoid(logits)

def predict(model, features: np.ndarray, objective: dict, device: str, mc_passes: int = 1):
    # Default is multiclass
    task = Task(objective.get("task", Task.multiclass))

    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    results = []
    model.eval()
    with torch.inference_mode():
        if mc_passes > 1:
            model.train()  # activate only configured dropout layers
        for _ in range(mc_passes):
            logits = model(x)
            results.append(calc_probs(logits, task))
        model.eval()
        return results[0] if mc_passes == 1 else torch.stack(results)
