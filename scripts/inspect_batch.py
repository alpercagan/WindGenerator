#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from windgen.dataset import WindMelDataset, DatasetConfig
from windgen.mels import MelSpecConfig


def save_mel_image(mel: torch.Tensor, out_path: Path, title: str = "") -> None:
    # mel: (1, n_mels, T)
    m = mel.squeeze(0).cpu().numpy()  # (n_mels, T)
    plt.figure()
    plt.imshow(m, aspect="auto", origin="lower")
    plt.title(title)
    plt.xlabel("time frames")
    plt.ylabel("mel bins")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    data_dir = Path(os.path.expanduser("~/Datasets/wind_clean")).resolve()
    out_dir = Path("outputs/inspect").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = WindMelDataset(
        DatasetConfig(data_dir=data_dir),
        MelSpecConfig(sr=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=128),
        target_frames=440,
    )

    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    batch = next(iter(dl))
    mel = batch["mel"]  # (B, 1, 128, T)

    print("Batch mel shape:", tuple(mel.shape))
    print("min/max:", float(mel.min()), float(mel.max()))
    print("mean/std:", float(mel.mean()), float(mel.std()))

    for i in range(min(4, mel.shape[0])):
        save_mel_image(mel[i], out_dir / f"mel_{i}.png", title=batch["clip_filename"][i])

    print(f"Saved images to: {out_dir}")


if __name__ == "__main__":
    main()