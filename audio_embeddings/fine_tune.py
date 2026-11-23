"""Waveform-level fine-tuning and CLS-AL shadow training for official adapters."""
from __future__ import annotations

import copy
import numpy as np

from .classifier import _loss, build_head, seed_everything
from .data.audio_io import load_audio, slice_audio, window_audio


def _windows(adapter, record: dict, preprocessing: dict):
    waveform = load_audio(record["audio_path"], adapter.sample_rate,
                          channel_mode=preprocessing.get("channel_mode", "mean"))
    if "start_seconds" in record and "end_seconds" in record:
        waveform = slice_audio(waveform, adapter.sample_rate, float(record["start_seconds"]),
                               float(record["end_seconds"]))
    seconds = float(preprocessing.get("window_seconds", adapter.window_seconds))
    if adapter.name == "ssast" and seconds != adapter.window_seconds:
        raise ValueError("SSAST fine-tuning requires fixed 10.24-second input windows")
    return window_audio(waveform, adapter.sample_rate, seconds,
                        pad_mode=preprocessing.get("pad_mode", "constant"))


def encode_records(adapter, records: list[dict], indices: np.ndarray, preprocessing: dict,
                   encoder=None, batch_size: int = 16) -> np.ndarray:
    """Compute one mean-pooled representation per indexed manifest record."""
    import torch
    model = encoder if encoder is not None else adapter.model
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            windows, owners = [], []
            for owner, index in enumerate(batch_indices):
                chunks = _windows(adapter, records[int(index)], preprocessing)
                windows.extend(chunks); owners.extend([owner] * len(chunks))
            waves = torch.stack(windows).to(adapter.device, dtype=torch.float32)
            features = adapter.encode_tensor(waves, encoder=model)
            owner_tensor = torch.as_tensor(owners, dtype=torch.long, device=features.device)
            pooled = torch.zeros((len(batch_indices), features.shape[-1]), dtype=features.dtype, device=features.device)
            pooled.index_add_(0, owner_tensor, features)
            counts = torch.bincount(owner_tensor, minlength=len(batch_indices)).clamp_min(1).unsqueeze(1)
            output.append(pooled / counts)
    return torch.cat(output, dim=0) if output else torch.empty((0, 0), device=adapter.device)


def _encode_training_batch(adapter, records, selected, preprocessing):
    import torch
    windows, owners = [], []
    for owner, index in enumerate(selected):
        chunks = _windows(adapter, records[int(index)], preprocessing)
        windows.extend(chunks); owners.extend([owner] * len(chunks))
    waves = torch.stack(windows).to(adapter.device, dtype=torch.float32)
    features = adapter.encode_tensor(waves)
    owner_tensor = torch.as_tensor(owners, dtype=torch.long, device=features.device)
    pooled = torch.zeros((len(selected), features.shape[-1]), dtype=features.dtype, device=features.device)
    pooled = pooled.index_add(0, owner_tensor, features)
    counts = torch.bincount(owner_tensor, minlength=len(selected)).clamp_min(1).unsqueeze(1)
    return pooled / counts


def _clone_shadow(adapter, head):
    clone = copy.copy(adapter)
    clone.model = copy.deepcopy(adapter.model).eval()
    clone_head = copy.deepcopy(head).eval()
    for param in clone.model.parameters(): param.requires_grad_(False)
    return clone, clone_head


def _ema(shadow_encoder, shadow_head, adapter, head, alpha: float):
    import torch
    with torch.no_grad():
        for target, source in zip(shadow_encoder.model.parameters(), adapter.model.parameters()):
            target.mul_(alpha).add_(source, alpha=1 - alpha)
        for target, source in zip(shadow_head.parameters(), head.parameters()):
            target.mul_(alpha).add_(source, alpha=1 - alpha)


def fit_finetune(adapter, records: list[dict], selected: list[int], targets: np.ndarray,
                 output_dim: int, objective: dict, config: dict, preprocessing: dict,
                 seed: int, head=None, shadows=None, optimizer_state=None,
                 start_epoch: int = 0, checkpoint_callback=None, rng_state=None):
    """Fine-tune the selected pretrained encoder and task head on revealed labels."""
    import torch
    seed_everything(seed)
    adapter.model.train()
    if head is None:
        head = build_head(int(config["embedding_dim"]), output_dim, config, adapter.device)
    head.train()
    train_targets = torch.as_tensor(targets, device=adapter.device)
    lr = float(config.get("learning_rate", 1e-5))
    optimizer = torch.optim.AdamW(list(adapter.model.parameters()) + list(head.parameters()), lr=lr,
                                  weight_decay=float(config.get("weight_decay", 1e-4)))
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if rng_state is not None:
        import random
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and rng_state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng_state["cuda"])
    batch_size = int(config.get("batch_size", 8))
    total_epochs = int(config.get("epochs", 1))
    for epoch in range(start_epoch, total_epochs):
        adapter.model.train(); head.train()
        order = np.random.default_rng(seed).permutation(selected)
        for start in range(0, len(order), batch_size):
            batch_ids = order[start:start + batch_size].tolist()
            features = _encode_training_batch(adapter, records, batch_ids, preprocessing)
            y = train_targets[torch.as_tensor([selected.index(i) for i in batch_ids], device=adapter.device)]
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(head(features), y, objective)
            loss.backward(); optimizer.step()
        if shadows is not None:
            _ema(shadows[0], shadows[1], adapter, head, float(config.get("cls_alpha_long", 0.9999)))
            _ema(shadows[2], shadows[3], adapter, head, float(config.get("cls_alpha_short", 0.85)))
        if checkpoint_callback is not None and ((epoch + 1) % 5 == 0 or epoch + 1 == total_epochs):
            shadow_state = None
            if shadows is not None:
                shadow_state = {"long_backbone": shadows[0].model.state_dict(),
                                "long_head": shadows[1].state_dict(),
                                "short_backbone": shadows[2].model.state_dict(),
                                "short_head": shadows[3].state_dict()}
            import random
            checkpoint_callback({"backbone": adapter.model.state_dict(), "head": head.state_dict(),
                                 "shadows": shadow_state, "optimizer": optimizer.state_dict(),
                                 "epoch": epoch + 1,
                                 "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                                               "torch": torch.get_rng_state(),
                                               "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}})
    adapter.model.eval(); head.eval()
    if shadows is None:
        shadows = (*_clone_shadow(adapter, head), *_clone_shadow(adapter, head))
    return head, shadows


def predict_records(adapter, head, records: list[dict], indices: np.ndarray, preprocessing: dict,
                    objective: dict, batch_size: int = 8, encoder=None, head_override=None):
    import torch
    from .classifier import predict
    features = encode_records(adapter, records, indices, preprocessing, encoder=encoder, batch_size=batch_size)
    return predict(head_override if head_override is not None else head, features, objective, adapter.device)
