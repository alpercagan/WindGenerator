from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_mel_grid(mels: torch.Tensor, out_path: Path, title: str = "") -> None:
    """
    mels: (B, 1, 128, T) in roughly [-1, 1]
    Saves a vertical grid of up to 8 mels.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = mels.detach().cpu().numpy()
    b = min(m.shape[0], 8)

    plt.figure(figsize=(10, 2 * b))
    for i in range(b):
        ax = plt.subplot(b, 1, i + 1)
        img = m[i, 0]  # (128, T)
        ax.imshow(img, aspect="auto", origin="lower")
        ax.set_ylabel(f"{i}")
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()