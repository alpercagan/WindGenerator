#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import librosa

from diffusers import UNet2DModel, DDPMScheduler

from windgen.mels import MelSpecConfig


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="outputs/train_ddpm/final_model.pt")
    ap.add_argument("--mel_stats", type=str, default="outputs/mel_stats.json")
    ap.add_argument("--out_dir", type=str, default="outputs/samples_audio")
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--steps", type=int, default=1000, help="DDPM steps (1000 matches training).")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def build_model(sample_size_hw=(128, 440), device="cpu"):
    # Must match your training UNet
    H, W = sample_size_hw
    model = UNet2DModel(
        sample_size=(H, W),
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(16, 32, 64),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        mid_block_type="UNetMidBlock2D",
        norm_num_groups=8,
    ).to(device)
    return model


def mel_to_audio_griffinlim(
    mel_power: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    fmin: float,
    fmax: float,
    n_iter: int = 32,
) -> np.ndarray:
    """
    mel_power: (n_mels, T) power mel spectrogram (non-negative)
    Returns waveform (float32)
    """
    # librosa expects mel-power by default when converting to STFT magnitude
    stft_mag = librosa.feature.inverse.mel_to_stft(
        M=mel_power,
        sr=sr,
        n_fft=n_fft,
        power=2.0,
        fmin=fmin,
        fmax=fmax,
    )
    # Reconstruct waveform from magnitude with Griffin-Lim
    y = librosa.griffinlim(
        S=stft_mag,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
    )
    return y.astype(np.float32)


def main():
    args = parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt).resolve()
    stats_path = Path(args.mel_stats).resolve()
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")
    if not stats_path.exists():
        raise SystemExit(f"Missing mel stats: {stats_path}")

    stats = json.loads(stats_path.read_text())
    mu = float(stats["logmel_mean_median"])
    sig = float(stats["logmel_std_median"])
    mel_cfg = stats["mel_config"]

    # Must match training mel config
    cfg = MelSpecConfig(
        sr=int(mel_cfg["sr"]),
        n_fft=int(mel_cfg["n_fft"]),
        hop_length=int(mel_cfg["hop_length"]),
        win_length=int(mel_cfg["win_length"]),
        n_mels=int(mel_cfg["n_mels"]),
        f_min=float(mel_cfg["f_min"]),
        eps=float(mel_cfg["eps"]),
    )

    # These must match how you normalized in training:
    clamp_after_std = 4.0  # from your LogMelExtractor default
    H, W = 128, 440

    torch.manual_seed(args.seed)

    # Load model weights
    model = build_model((H, W), device=device)
    payload = torch.load(str(ckpt_path), map_location="cpu")
    if "model" in payload:
        model.load_state_dict(payload["model"])
    else:
        model.load_state_dict(payload)  # fallback
    model.eval()

    # Scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(args.steps)

    for i in range(args.num):
        with torch.no_grad():
            x = torch.randn((1, 1, H, W), device=device)
            for t in scheduler.timesteps:
                pred_noise = model(x, t).sample
                x = scheduler.step(pred_noise, t, x).prev_sample

        # x is normalized mel in roughly [-1,1]
        x_np = x.squeeze(0).squeeze(0).detach().cpu().numpy()  # (128, 440)

        # Undo training normalization:
        # training did: logmel -> z = (logmel-mean)/std -> clamp -> divide by clamp
        # so: z ≈ x * clamp_after_std
        z = x_np * clamp_after_std
        logmel = z * sig + mu

        # Convert logmel -> mel power
        mel_power = np.exp(logmel) - cfg.eps
        mel_power = np.clip(mel_power, 0.0, None)

        # Invert to audio
        y = mel_to_audio_griffinlim(
            mel_power=mel_power,
            sr=cfg.sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            fmin=cfg.f_min,
            fmax=cfg.sr / 2,
            n_iter=48,
        )

        # Normalize output audio a bit for listening (avoid clipping)
        peak = np.max(np.abs(y)) + 1e-12
        if peak > 0.98:
            y = y * (0.98 / peak)

        wav_path = out_dir / f"wind_sample_{i:02d}.wav"
        sf.write(str(wav_path), y, cfg.sr, subtype="PCM_16")
        print("Wrote:", wav_path)

    print("Done. Audio samples in:", out_dir)


if __name__ == "__main__":
    main()