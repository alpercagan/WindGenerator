from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch
import torchaudio
import soundfile as sf


@dataclass(frozen=True)
class MelSpecConfig:
    sr: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 128
    f_min: float = 20.0
    f_max: float | None = None
    power: float = 2.0
    eps: float = 1e-5


def create_mel_transform(cfg: MelSpecConfig, device: torch.device) -> torchaudio.transforms.MelSpectrogram:
    # f_max defaults to sr/2 if None
    f_max = cfg.f_max if cfg.f_max is not None else cfg.sr / 2
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sr,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        f_min=cfg.f_min,
        f_max=f_max,
        n_mels=cfg.n_mels,
        power=cfg.power,
        center=True,
        pad_mode="reflect",
        norm=None,
        mel_scale="htk",
    ).to(device)


def load_wav_mono_resample(path: str | Path, target_sr: int) -> torch.Tensor:
    """
    Returns mono waveform as torch.float32 tensor of shape (T,).
    Uses soundfile for decoding (no torchcodec dependency).
    """
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (T, C)
    x = x.mean(axis=1)  # mono
    wav = torch.from_numpy(x)  # (T,)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    wav = wav.clamp(-1.0, 1.0)
    return wav


class LogMelExtractor:
    """
    Produces normalized log-mel in ~[-1, 1].

    Two modes:
    - normalization="global": uses mu/sigma from mel_stats.json (Path A)
    - normalization="per_clip": uses each clip's own mean/std (old behavior)
    """

    def __init__(
        self,
        config: MelSpecConfig,
        device: str = "cpu",
        normalization: str = "global",  # "global" or "per_clip"
        stats_path: str | None = None,
        clamp: float = 4.0,
    ):
        self.config = config
        self.device = torch.device(device)
        self.mel_transform = create_mel_transform(config, self.device)

        self.normalization = normalization
        self.clamp = float(clamp)

        self.global_mean: float | None = None
        self.global_std: float | None = None

        if self.normalization == "global":
            if not stats_path:
                raise ValueError("stats_path is required when normalization='global'")
            stats = json.loads(Path(stats_path).read_text())
            # robust medians
            self.global_mean = float(stats["logmel_mean_median"])
            self.global_std = float(stats["logmel_std_median"])

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: (T,) or (1,T)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device)

        mel = self.mel_transform(audio)  # (1, n_mels, frames)
        logmel = torch.log(mel + self.config.eps)

        if self.normalization == "per_clip":
            mean = logmel.mean()
            std = logmel.std().clamp_min(1e-5)
            z = (logmel - mean) / std
        else:
            # global
            assert self.global_mean is not None and self.global_std is not None
            z = (logmel - self.global_mean) / (self.global_std + 1e-8)

        x = z.clamp(-self.clamp, self.clamp) / self.clamp
        return x.detach().cpu()  # keep dataset on CPU