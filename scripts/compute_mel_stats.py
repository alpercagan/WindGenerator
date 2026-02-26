#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="~/Datasets/wind_clean")
    ap.add_argument("--num_clips", type=int, default=400, help="How many clips to scan (400 is plenty).")
    ap.add_argument("--out", type=str, default="outputs/mel_stats.json")
    # Mel config (must match training)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop_length", type=int, default=256)
    ap.add_argument("--win_length", type=int, default=1024)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--f_min", type=float, default=20.0)
    ap.add_argument("--eps", type=float, default=1e-5)
    return ap.parse_args()


def load_wav(path: Path, target_sr: int) -> torch.Tensor:
    audio, sr = sf.read(str(path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    wav = torch.from_numpy(audio).unsqueeze(0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def main():
    args = parse_args()

    data_dir = Path(os.path.expanduser(args.data_dir)).resolve()
    meta = pd.read_csv(data_dir / "metadata.csv")
    clips_dir = data_dir / "clips"

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sr,
        n_fft=args.n_fft,
        win_length=args.win_length,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        f_min=args.f_min,
        f_max=args.sr / 2,
        power=2.0,
        center=True,
        pad_mode="reflect",
    )

    means = []
    stds = []

    n = min(args.num_clips, len(meta))
    # deterministic subset
    rows = meta.head(n)

    for _, r in tqdm(rows.iterrows(), total=n, desc="Scanning clips"):
        p = clips_dir / str(r["clip_filename"])
        if not p.exists():
            continue
        wav = load_wav(p, args.sr)
        with torch.no_grad():
            m = mel(wav)                       # (1, n_mels, T)
            logm = torch.log(m + args.eps)     # log-mel (NOT standardized)
            means.append(float(logm.mean().item()))
            stds.append(float(logm.std().item()))

    if not means:
        raise SystemExit("No clips scanned successfully.")

    stats = {
        "mel_config": {
            "sr": args.sr,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "win_length": args.win_length,
            "n_mels": args.n_mels,
            "f_min": args.f_min,
            "eps": args.eps,
        },
        # typical per-clip log-mel stats
        "logmel_mean_median": float(np.median(means)),
        "logmel_std_median": float(np.median(stds)),
        "logmel_mean_mean": float(np.mean(means)),
        "logmel_std_mean": float(np.mean(stds)),
        "num_scanned": int(len(means)),
    }

    out_path.write_text(json.dumps(stats, indent=2))
    print("Wrote:", out_path)
    print(stats)


if __name__ == "__main__":
    main()