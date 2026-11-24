"""Model-specific official checkpoint loading and waveform preprocessing."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
import numpy as np
from ..data.device import resolve_device
from ..data.audio_io import window_audio


class AudioModel:
    name = "base"
    sample_rate = 16000
    window_seconds = 10
    pooling = "mean of non-overlapping 10-second window embeddings; zero-pad final window"

    def __init__(self, checkpoint: Path, device: str, source: Path | None = None):
        self.checkpoint = checkpoint.resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.checkpoint}")
        self.device = device
        self.source = source.resolve() if source else None

    def encode(self, batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def encode_tensor(self, batch, encoder=None):
        raise NotImplementedError(f"{self.name} adapter does not support fine-tuning")


class BeatsModel(AudioModel):
    name, sample_rate = "beats", 16000

    def __init__(self, checkpoint: Path, device: str, source: Path | None = None):
        source = source or Path(__file__).resolve().parent.parent / "models_src" / "BEATs"
        super().__init__(checkpoint, device, source)
        if self.source:
            sys.path.insert(0, str(self.source))
        try:
            import torch
            from BEATs import BEATs, BEATsConfig
        except ImportError as exc:
            raise RuntimeError("BEATs requires the vendored official source and torch/torchaudio") from exc
        self.torch = torch
        state = torch.load(str(self.checkpoint), map_location="cpu", weights_only=False)
        self.checkpoint_config = state["cfg"]
        self.model = BEATs(BEATsConfig(self.checkpoint_config))
        self.model.load_state_dict(state["model"])
        self.model.to(device).eval()

    def encode_tensor(self, batch, encoder=None):
        torch = self.torch
        values = batch if torch.is_tensor(batch) else torch.as_tensor(batch, dtype=torch.float32, device=self.device)
        values = values.to(self.device, dtype=torch.float32)
        mask = torch.zeros_like(values, dtype=torch.bool)
        model = encoder if encoder is not None else self.model
        features = model.extract_features(values, padding_mask=mask)[0]
        return features.mean(dim=1)

    def encode(self, batch: np.ndarray) -> np.ndarray:
        with self.torch.inference_mode():
            return self.encode_tensor(batch).cpu().numpy()


class ClapModel(AudioModel):
    name, sample_rate = "clap", 48000

    def __init__(self, checkpoint: Path, device: str, source: Path | None = None):
        super().__init__(checkpoint, device, source)
        try:
            import laion_clap
        except ImportError as exc:
            raise RuntimeError("CLAP requires laion-clap; install it in the audio environment") from exc
        self.model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
        self.model.load_ckpt(ckpt=str(self.checkpoint))

    def encode(self, batch: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.get_audio_embedding_from_data(x=batch, use_tensor=False), dtype=np.float32)


class PannsModel(AudioModel):
    name, sample_rate = "panns", 32000

    def __init__(self, checkpoint: Path, device: str, source: Path | None = None):
        super().__init__(checkpoint, device, source)
        try:
            module = importlib.import_module("panns_inference")
        except ImportError as exc:
            raise RuntimeError("PANNs requires panns-inference; install it in the audio environment") from exc
        self.model = module.AudioTagging(checkpoint_path=str(self.checkpoint), device=device)

    def encode(self, batch: np.ndarray) -> np.ndarray:
        _, embedding = self.model.inference(batch)
        return np.asarray(embedding, dtype=np.float32)


class SsastModel(AudioModel):
    """Frozen official SSAST encoder using the repository's ASTModel."""
    name, sample_rate = "ssast", 16000
    window_seconds = 10.24
    pooling = "official 128-bin Kaldi fbank; 1024 frames; average CLS and distillation tokens"
    norm_mean, norm_std = -4.2677393, 4.5689974

    def __init__(self, checkpoint: Path, device: str, source: Path | None = None):
        source = source or Path(__file__).resolve().parent.parent / "models_src" / "SSAST" / "src"
        super().__init__(checkpoint, device, source)
        model_dir = self.source / "models" if (self.source / "models").is_dir() else self.source
        sys.path.insert(0, str(model_dir.parent))
        try:
            import torch
            from models.ast_models import ASTModel
            import torchaudio
        except ImportError as exc:
            raise RuntimeError("SSAST requires its vendored official source, torch, torchaudio, and timm==0.4.5") from exc
        self.torch, self.torchaudio = torch, torchaudio
        # Official patch-based base model architecture. Loading is intentionally
        # delegated to ASTModel so official checkpoint key handling is retained.
        # self.model = ASTModel(label_dim=527, fshape=16, tshape=16, fstride=10,
        #                       tstride=10, input_fdim=128, input_tdim=1024,
        #                       model_size="base", pretrain_stage=False,
        #                       load_pretrained_mdl_path=str(self.checkpoint))
        # Official frame-based base model
        self.model = ASTModel(label_dim=527, fshape=128, tshape=2, fstride=128,
                              tstride=2, input_fdim=128, input_tdim=1024,
                              model_size="base", pretrain_stage=False,
                              load_pretrained_mdl_path=str(self.checkpoint))
        checkpoint_state = self.torch.load(str(self.checkpoint), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint_state, dict):
            raise ValueError("SSAST checkpoint must contain a state-dict mapping")
        visual_state = self.model.v.state_dict()
        missing = []
        shape_mismatches = []
        for key, value in visual_state.items():
            source_key = f"module.v.{key}"
            source_value = checkpoint_state.get(source_key)
            if source_value is None:
                missing.append(source_key)
            elif key != "pos_embed" and tuple(source_value.shape) != tuple(value.shape):
                shape_mismatches.append((source_key, tuple(source_value.shape), tuple(value.shape)))
        if missing or shape_mismatches:
            raise ValueError(
                "SSAST checkpoint does not cover the expected official encoder weights: "
                f"missing={missing[:8]}, shape_mismatches={shape_mismatches[:8]}"
            )
        self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def featurize(self, batch):
        torch = self.torch
        features = []
        for waveform in batch:
            audio = torch.as_tensor(waveform, dtype=torch.float32).detach().cpu()
            fbank = self.torchaudio.compliance.kaldi.fbank(
                audio.unsqueeze(0), htk_compat=True, sample_frequency=self.sample_rate,
                use_energy=False, window_type="hanning", num_mel_bins=128,
                dither=0.0, frame_shift=10)
            if fbank.shape[0] < 1024:
                fbank = torch.nn.functional.pad(fbank, (0, 0, 0, 1024 - fbank.shape[0]))
            else:
                fbank = fbank[:1024]
            features.append((fbank - self.norm_mean) / (self.norm_std * 2))
        return torch.stack(features).to(self.device)

    def encode_tensor(self, batch, encoder=None):
        if self.torch.is_tensor(batch) and batch.ndim == 3:
            x = batch
        elif self.torch.is_tensor(batch):
            x = self.featurize(batch.detach().cpu())
        else:
            x = self.featurize(batch)
        # Official ASTModel's visual transformer returns the token sequence;
        # use the pretrained CLS/distillation token representation before its
        # downstream classification head.
        model = encoder if encoder is not None else self.model
        v = model.v
        patches = v.patch_embed(x.transpose(1, 2).unsqueeze(1))
        batch_size = patches.shape[0]
        if model.cls_token_num == 2:
            cls = v.cls_token.expand(batch_size, -1, -1)
            dist = v.dist_token.expand(batch_size, -1, -1)
            tokens = self.torch.cat((cls, dist, patches), dim=1)
        else:
            tokens = self.torch.cat((v.cls_token.expand(batch_size, -1, -1), patches), dim=1)
        tokens = v.pos_drop(tokens + v.pos_embed)
        for block in v.blocks:
            tokens = block(tokens)
        tokens = v.norm(tokens)
        return tokens[:, :model.cls_token_num].mean(dim=1).float()

    def encode(self, batch: np.ndarray) -> np.ndarray:
        with self.torch.inference_mode():
            return self.encode_tensor(batch).cpu().numpy()


MODELS = {"beats": BeatsModel, "clap": ClapModel, "panns": PannsModel,
          "ssast": SsastModel}


def create_model(name: str, checkpoint: Path, device: str, source: Path | None = None) -> AudioModel:
    if name not in MODELS:
        raise ValueError(f"unsupported model: {name}")
    resolved = resolve_device(device)
    return MODELS[name](checkpoint, str(resolved), source)
