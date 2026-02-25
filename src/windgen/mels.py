from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import torch
import torchaudio

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class MelSpecConfig:
    sr: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 128
    f_min: float = 20.0
    f_max: float = None  # if None -> sr/2
    eps: float = 1e-5
    clamp_std_min: float = 1e-4
    clamp_after_std: float = 4.0  # clamp z-scores to [-4,4]


class LogMelExtractor(torch.nn.Module):
    def __init__(self, cfg: MelSpecConfig):
        super().__init__()
        f_max = cfg.f_max if cfg.f_max is not None else cfg.sr / 2
        self.cfg = cfg
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sr,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            f_min=cfg.f_min,
            f_max=f_max,
            n_mels=cfg.n_mels,
            power=2.0,           # power spectrogram
            normalized=False,
            center=True,
            pad_mode="reflect",
        )

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (1, N) float32 in [-1,1]
        returns: (1, n_mels, T) float32, log-mel then standardized
        """
        mel = self.mel(wav)  # (1, n_mels, T)
        mel = torch.log(mel + self.cfg.eps)

        # per-clip standardization for stable diffusion training
        mean = mel.mean()
        std = mel.std().clamp_min(self.cfg.clamp_std_min)
        mel = (mel - mean) / std
        mel = mel.clamp(-self.cfg.clamp_after_std, self.cfg.clamp_after_std)

        # optional rescale to [-1,1] by dividing by clamp value
        mel = mel / self.cfg.clamp_after_std
        return mel


def load_wav_mono_resample(path: Path, target_sr: int) -> torch.Tensor:
    """
    Robust WAV loader for macOS: uses soundfile (no torchcodec needed).
    Returns: (1, N) float32 in [-1, 1]
    """
    audio, sr = sf.read(str(path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)

    # If stereo, average to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    wav = torch.from_numpy(audio).unsqueeze(0)  # (1, N)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)

    return wav