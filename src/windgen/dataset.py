from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from windgen.mels import MelSpecConfig, LogMelExtractor, load_wav_mono_resample


@dataclass(frozen=True)
class DatasetConfig:
    data_dir: Path  # wind_clean
    clips_subdir: str = "clips"
    metadata_name: str = "metadata.csv"


class WindMelDataset(Dataset):
    def __init__(self, cfg: DatasetConfig, mel_cfg: MelSpecConfig):
        self.cfg = cfg
        self.data_dir = Path(os.path.expanduser(str(cfg.data_dir))).resolve()
        self.clips_dir = self.data_dir / cfg.clips_subdir
        self.meta_path = self.data_dir / cfg.metadata_name

        if not self.meta_path.exists():
            raise FileNotFoundError(f"Missing metadata.csv: {self.meta_path}")
        if not self.clips_dir.exists():
            raise FileNotFoundError(f"Missing clips dir: {self.clips_dir}")

        self.df = pd.read_csv(self.meta_path)
        if "clip_filename" not in self.df.columns:
            raise ValueError("metadata.csv must contain clip_filename column")

        self.extractor = LogMelExtractor(mel_cfg)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        clip_name = str(row["clip_filename"])
        clip_path = self.clips_dir / clip_name

        wav = load_wav_mono_resample(clip_path, target_sr=self.extractor.cfg.sr)
        mel = self.extractor(wav)  # (1, 128, T)

        return {
            "mel": mel,
            "clip_filename": clip_name,
            "source_path": str(row.get("source_path", "")),
        }