from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# -------------------------
# Multi-Resolution STFT loss
# -------------------------

def _stft_mag(
    x: torch.Tensor,
    fft_size: int,
    hop: int,
    win: int,
) -> torch.Tensor:
    """
    x: (B, T)
    return: magnitude (B, F, frames)
    """
    window = torch.hann_window(win, device=x.device, dtype=x.dtype)
    X = torch.stft(
        x,
        n_fft=fft_size,
        hop_length=hop,
        win_length=win,
        window=window,
        center=True,
        return_complex=True,
    )
    return torch.sqrt(X.real.pow(2) + X.imag.pow(2) + 1e-9)


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, configs: List[Tuple[int, int, int]]):
        super().__init__()
        self.configs = configs

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for fft_size, hop, win in self.configs:
            Yh = _stft_mag(y_hat, fft_size, hop, win)
            Y = _stft_mag(y, fft_size, hop, win)

            # spectral convergence
            # Use explicit pow/sum/sqrt — torch.norm(p="fro") on 3D is broken in PyTorch 2.x
            sc = ((Y - Yh).pow(2).sum().sqrt() / Y.pow(2).sum().sqrt().clamp_min(1e-7)).clamp(max=10.0)
            # log-mag L1
            logmag = F.l1_loss(torch.log(Yh), torch.log(Y))
            loss = loss + (sc + logmag)
        return loss / len(self.configs)


# -------------------------
# Tiny vocoder (mel -> waveform)
# -------------------------

class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        pad = ((kernel_size - 1) // 2) * dilation
        self.c1 = weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation))
        self.c2 = weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.leaky_relu(x, 0.2)
        y = self.c1(y)
        y = F.leaky_relu(y, 0.2)
        y = self.c2(y)
        return x + y


class MRFBlock(nn.Module):
    """
    Multi-Receptive Field block (HiFi-GAN style).
    One ResBlock branch per kernel size, each branch applies dilations in sequence.
    forward() sums all branches and divides by len(kernel_sizes).
    """
    def __init__(
        self,
        channels: int,
        kernel_sizes: List[int] = [3, 7, 11],
        dilations: List[int] = [1, 3, 5],
    ):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(*[ResBlock(channels, k, d) for d in dilations])
            for k in kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = sum(branch(x) for branch in self.branches)
        return out / len(self.branches)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, scale: int):
        super().__init__()
        self.scale = scale
        self.conv = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size=2 * scale + 1, padding=scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        return self.conv(x)


class TinyVocoder(nn.Module):
    """
    Input: mel_norm (B, n_mels, frames) in ~[-1,1]
    Output: waveform (B, T) where T ≈ frames * hop_length (hop_length=256)

    Upsample factor total: 8 * 8 * 4 = 256
    """
    def __init__(self, n_mels: int = 128, base_ch: int = 256):
        super().__init__()
        self.pre = weight_norm(nn.Conv1d(n_mels, base_ch, kernel_size=7, padding=3))

        self.up1 = UpBlock(base_ch, base_ch // 2, scale=8)
        self.rb1 = MRFBlock(base_ch // 2, kernel_sizes=[3, 7, 11], dilations=[1, 3, 5])

        self.up2 = UpBlock(base_ch // 2, base_ch // 4, scale=8)
        self.rb2 = MRFBlock(base_ch // 4, kernel_sizes=[3, 7, 11], dilations=[1, 3, 5])

        self.up3 = UpBlock(base_ch // 4, base_ch // 8, scale=4)
        self.rb3 = MRFBlock(base_ch // 8, kernel_sizes=[3, 7, 11], dilations=[1, 3, 5])

        self.post = weight_norm(nn.Conv1d(base_ch // 8, 1, kernel_size=7, padding=3))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.pre(mel)
        x = F.leaky_relu(x, 0.2)

        x = self.up1(x)
        x = self.rb1(x)
        x = F.leaky_relu(x, 0.2)

        x = self.up2(x)
        x = self.rb2(x)
        x = F.leaky_relu(x, 0.2)

        x = self.up3(x)
        x = self.rb3(x)
        x = F.leaky_relu(x, 0.2)

        x = self.post(x)
        x = torch.tanh(x)
        return x.squeeze(1)  # (B, T)


# -------------------------
# Lite Multi-Period Discriminator
# -------------------------

class LiteMPD(nn.Module):
    """
    Lite Multi-Period Discriminator with periods [2, 3, 5, 7, 11].

    Each sub-discriminator reshapes (B, T) → (B, 1, T//p, p) and applies
    a stack of 2D convolutions. forward() returns a list of (score, feature_maps)
    per sub-discriminator, where feature_maps is a list of intermediate
    activations BEFORE each activation function (used for feature matching loss).
    """
    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.periods = periods
        self.sub_discs = nn.ModuleList([self._make_sub_disc() for _ in periods])

    def _make_sub_disc(self) -> nn.ModuleList:
        channel_pairs = [(1, 16), (16, 32), (32, 64), (64, 128), (128, 128)]
        layers = nn.ModuleList()
        for in_ch, out_ch in channel_pairs:
            layers.append(
                weight_norm(nn.Conv2d(in_ch, out_ch, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)))
            )
        layers.append(weight_norm(nn.Conv2d(128, 1, kernel_size=(3, 1), padding=(1, 0))))
        return layers

    def _forward_sub_disc(
        self,
        sub_disc: nn.ModuleList,
        x: torch.Tensor,
        p: int,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, T = x.shape
        if T % p != 0:
            x = F.pad(x, (0, p - T % p))
        x = x.view(B, 1, -1, p)  # (B, 1, T//p, p)

        feature_maps: List[torch.Tensor] = []
        for conv in sub_disc[:-1]:  # all conv layers except final
            x = conv(x)
            feature_maps.append(x)   # activation BEFORE leaky_relu
            x = F.leaky_relu(x, 0.1)

        score = sub_disc[-1](x)      # final conv, no activation
        return score, feature_maps

    def forward(
        self, audio: torch.Tensor
    ) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        audio: (B, T)
        returns: list of (score, feature_maps) per sub-discriminator
          score:        (B, 1, T', 1) — raw logits, flatten before loss
          feature_maps: list of 5 intermediate tensors (before activation)
        """
        return [
            self._forward_sub_disc(sub_disc, audio, p)
            for p, sub_disc in zip(self.periods, self.sub_discs)
        ]


# -------------------------
# Lite Multi-Scale Discriminator (MSD)
# -------------------------

class LiteMSD(nn.Module):
    """
    Lite Multi-Scale Discriminator for aperiodic texture audio (wind, rain, etc).

    Processes audio at multiple temporal resolutions via average pooling
    (scales=[1, 2, 4] → original, 2× downsampled, 4× downsampled).
    Each scale uses a Conv1d stack to capture spectral envelope and temporal
    amplitude modulation — the key characteristics of wind audio.

    forward() returns a list of (score, feature_maps) per scale,
    compatible with the same feature_matching_loss and discriminator_loss
    functions used by LiteMPD.
    """
    def __init__(self, scales: List[int] = [1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.sub_discs = nn.ModuleList([self._make_sub_disc() for _ in scales])
        self.pools = nn.ModuleList([
            nn.Identity() if s == 1 else nn.AvgPool1d(kernel_size=s, stride=s)
            for s in scales
        ])

    def _make_sub_disc(self) -> nn.ModuleList:
        # Large kernels to capture spectral envelope at each temporal scale
        layers = nn.ModuleList([
            weight_norm(nn.Conv1d(1,   16,  kernel_size=15, stride=1, padding=7)),
            weight_norm(nn.Conv1d(16,  32,  kernel_size=41, stride=4, padding=20, groups=4)),
            weight_norm(nn.Conv1d(32,  64,  kernel_size=41, stride=4, padding=20, groups=16)),
            weight_norm(nn.Conv1d(64,  128, kernel_size=41, stride=4, padding=20, groups=16)),
            weight_norm(nn.Conv1d(128, 128, kernel_size=5,  stride=1, padding=2)),
        ])
        layers.append(weight_norm(nn.Conv1d(128, 1, kernel_size=3, stride=1, padding=1)))
        return layers

    def _forward_sub_disc(
        self,
        pool: nn.Module,
        sub_disc: nn.ModuleList,
        audio: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = pool(audio.unsqueeze(1))  # (B, 1, T//scale)
        feature_maps: List[torch.Tensor] = []
        for conv in sub_disc[:-1]:
            x = conv(x)
            feature_maps.append(x)
            x = F.leaky_relu(x, 0.1)
        score = sub_disc[-1](x)
        return score, feature_maps

    def forward(
        self, audio: torch.Tensor
    ) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        audio: (B, T)
        returns: list of (score, feature_maps) per scale
        """
        return [
            self._forward_sub_disc(pool, sub_disc, audio)
            for pool, sub_disc in zip(self.pools, self.sub_discs)
        ]


class CombinedDisc(nn.Module):
    """
    Combined MPD + MSD discriminator.

    Returns concatenated list of (score, feature_maps) from both discriminators.
    Drop-in replacement for LiteMPD: same forward() interface, more sub-discriminators.
    """
    def __init__(
        self,
        periods: List[int] = [2, 3, 5, 7, 11],
        scales: List[int] = [1, 2, 4],
    ):
        super().__init__()
        self.mpd = LiteMPD(periods=periods)
        self.msd = LiteMSD(scales=scales)

    def forward(
        self, audio: torch.Tensor
    ) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        return self.mpd(audio) + self.msd(audio)
