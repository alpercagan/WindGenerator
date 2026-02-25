#!/usr/bin/env python3
"""
prepare_dataset.py

Preprocess a folder of wind recordings into a clean dataset of fixed-length clips.

Pipeline:
1) Discover audio files (.wav/.mp3/.flac) recursively under input_dir
2) Load audio, resample to target SR, convert to mono
3) Trim leading/trailing dead air (librosa.effects.trim)
4) RMS-normalize to a target dBFS (with anti-clipping safeguard)
5) Chunk into fixed-length clips with overlap
6) Skip chunks that are too quiet
7) Save as 16-bit PCM WAV + write metadata.csv

Notes:
- MP3 decoding may require ffmpeg installed on your system.
- This script NEVER modifies your raw files. It only reads input_dir and writes output_dir.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm

# librosa is convenient for resampling + trimming + audio loading.
import librosa


AUDIO_EXTS = {".wav", ".mp3", ".flac"}


@dataclass
class Stats:
    files_scanned: int = 0
    files_loaded_ok: int = 0
    files_failed: int = 0
    clips_written: int = 0
    clips_skipped_too_quiet: int = 0
    clips_skipped_too_short: int = 0
    clips_skipped_after_trim: int = 0


def dbfs_from_rms(rms: float, eps: float = 1e-12) -> float:
    """Convert RMS (0..1) to dBFS."""
    return 20.0 * math.log10(max(rms, eps))


def rms_from_dbfs(db: float) -> float:
    """Convert dBFS to RMS (0..1)."""
    return 10.0 ** (db / 20.0)


def compute_rms(x: np.ndarray) -> float:
    """RMS of signal x."""
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def peak_abs(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)) + 1e-12)


def discover_audio_files(input_dir: Path) -> List[Path]:
    files: List[Path] = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    files.sort()
    return files


def load_audio_mono_resampled(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    """
    Load audio from file, return mono float32 in [-1, 1] and the original sr.
    We resample to target_sr inside librosa.load.
    """
    # librosa.load returns float32 mono by default if mono=True
    y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    y = np.asarray(y, dtype=np.float32)
    return y, sr


def trim_dead_air(y: np.ndarray, top_db: float) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Trim leading/trailing silence using librosa.effects.trim.
    top_db is relative to peak: higher => more trimming.
    """
    if y.size == 0:
        return y, (0, 0)
    yt, idx = librosa.effects.trim(y, top_db=top_db)
    return np.asarray(yt, dtype=np.float32), idx


def rms_normalize(y: np.ndarray, target_rms_dbfs: float, clip_peak: float = 0.99) -> Tuple[np.ndarray, float, float]:
    """
    RMS normalize to target dBFS.
    Anti-clipping: if peak exceeds clip_peak, scale down to fit.

    Returns:
      y_out, rms_dbfs_after, peak_after
    """
    if y.size == 0:
        return y, -120.0, 0.0

    current_rms = compute_rms(y)
    target_rms = rms_from_dbfs(target_rms_dbfs)

    if current_rms < 1e-10:
        # Basically silence: return unchanged (will likely be skipped later)
        return y, dbfs_from_rms(current_rms), peak_abs(y)

    gain = target_rms / current_rms
    y_out = y * gain

    pk = peak_abs(y_out)
    if pk > clip_peak:
        y_out = y_out * (clip_peak / pk)

    rms_db = dbfs_from_rms(compute_rms(y_out))
    pk2 = peak_abs(y_out)
    return np.asarray(y_out, dtype=np.float32), rms_db, pk2


def chunk_signal(
    y: np.ndarray,
    sr: int,
    clip_seconds: float,
    overlap: float,
) -> Iterable[Tuple[np.ndarray, float]]:
    """
    Yield (chunk, start_time_sec) for fixed-length chunks with overlap.
    """
    assert 0.0 <= overlap < 1.0, "overlap must be in [0, 1)"
    clip_len = int(round(clip_seconds * sr))
    hop = int(round(clip_len * (1.0 - overlap)))
    hop = max(hop, 1)

    if y.size < clip_len:
        return  # too short

    num = 1 + (y.size - clip_len) // hop
    for i in range(num):
        start = i * hop
        end = start + clip_len
        chunk = y[start:end]
        start_sec = start / sr
        yield np.asarray(chunk, dtype=np.float32), start_sec


def safe_makedirs(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_wav_16bit(path: Path, y: np.ndarray, sr: int) -> None:
    """
    Write mono float32 [-1,1] to 16-bit PCM WAV.
    """
    # soundfile will quantize if subtype='PCM_16'
    sf.write(str(path), y, sr, subtype="PCM_16")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare a wind dataset: clean + chunk audio into fixed-length WAV clips.")
    ap.add_argument("--input_dir", type=str, required=True, help="Folder with raw audio files (wav/mp3/flac).")
    ap.add_argument("--output_dir", type=str, required=True, help="Folder to write processed dataset.")
    ap.add_argument("--sr", type=int, default=22050, help="Target sample rate.")
    ap.add_argument("--clip_seconds", type=float, default=5.12, help="Chunk duration in seconds.")
    ap.add_argument("--overlap", type=float, default=0.5, help="Overlap fraction between chunks (0.. <1).")
    ap.add_argument("--trim_db", type=float, default=40.0, help="Trim threshold in dB below peak for librosa.effects.trim.")
    ap.add_argument("--target_rms_db", type=float, default=-20.0, help="Target RMS level in dBFS after normalization.")
    ap.add_argument("--min_rms_db", type=float, default=-45.0, help="Skip chunks quieter than this RMS dBFS.")
    ap.add_argument("--limit_files", type=int, default=0, help="Debug: process only first N files (0 = no limit).")
    ap.add_argument("--dry_run", action="store_true", help="Discover + analyze but do not write files.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    input_dir = Path(os.path.expanduser(args.input_dir)).resolve()
    output_dir = Path(os.path.expanduser(args.output_dir)).resolve()

    if not input_dir.exists():
        print(f"ERROR: input_dir does not exist: {input_dir}", file=sys.stderr)
        return 2

    clips_dir = output_dir / "clips"
    safe_makedirs(clips_dir)

    audio_files = discover_audio_files(input_dir)
    if args.limit_files and args.limit_files > 0:
        audio_files = audio_files[: args.limit_files]

    if len(audio_files) == 0:
        print(f"ERROR: No audio files found under {input_dir} with extensions: {sorted(AUDIO_EXTS)}", file=sys.stderr)
        return 3

    # Metadata setup
    metadata_path = output_dir / "metadata.csv"
    if not args.dry_run:
        safe_makedirs(output_dir)
        # write header fresh each run
        with open(metadata_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "clip_filename",
                "source_path",
                "out_sr",
                "clip_start_sec",
                "clip_duration_sec",
                "rms_dbfs",
                "peak",
                "trim_start_sample",
                "trim_end_sample",
            ])

    stats = Stats(files_scanned=len(audio_files))

    next_clip_idx = 1  # stable sequential naming
    clip_duration = float(args.clip_seconds)

    for src_path in tqdm(audio_files, desc="Processing files", unit="file"):
        try:
            y, sr_loaded = load_audio_mono_resampled(src_path, target_sr=int(args.sr))
            stats.files_loaded_ok += 1
        except Exception as e:
            stats.files_failed += 1
            tqdm.write(f"[LOAD FAIL] {src_path}: {type(e).__name__}: {e}")
            continue

        # Trim dead air
        y_trim, (t0, t1) = trim_dead_air(y, top_db=float(args.trim_db))
        if y_trim.size == 0:
            stats.clips_skipped_after_trim += 1
            continue

        # Normalize whole file (after trim) so chunk loudness is consistent-ish
        y_norm, _, _ = rms_normalize(y_trim, target_rms_dbfs=float(args.target_rms_db), clip_peak=0.99)

        # Chunk
        any_chunk = False
        for chunk, start_sec in chunk_signal(y_norm, sr=int(args.sr), clip_seconds=clip_duration, overlap=float(args.overlap)) or []:
            any_chunk = True

            # Skip too-quiet chunks
            rms_db = dbfs_from_rms(compute_rms(chunk))
            if rms_db < float(args.min_rms_db):
                stats.clips_skipped_too_quiet += 1
                continue

            pk = peak_abs(chunk)

            clip_name = f"wind_{next_clip_idx:06d}.wav"
            clip_path = clips_dir / clip_name

            if not args.dry_run:
                write_wav_16bit(clip_path, chunk, sr=int(args.sr))

                with open(metadata_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        clip_name,
                        str(src_path),
                        int(args.sr),
                        round(float(start_sec), 6),
                        round(float(clip_duration), 6),
                        round(float(rms_db), 6),
                        round(float(pk), 6),
                        int(t0),
                        int(t1),
                    ])

            stats.clips_written += 1
            next_clip_idx += 1

        if not any_chunk:
            stats.clips_skipped_too_short += 1

    # Summary
    print("\n=== Dataset Preparation Summary ===")
    print(f"Input dir:      {input_dir}")
    print(f"Output dir:     {output_dir}")
    print(f"Files scanned:  {stats.files_scanned}")
    print(f"Files loaded:   {stats.files_loaded_ok}")
    print(f"Files failed:   {stats.files_failed}")
    print(f"Clips written:  {stats.clips_written}" + (" (dry-run)" if args.dry_run else ""))
    print(f"Clips skipped (too quiet): {stats.clips_skipped_too_quiet}")
    print(f"Clips skipped (too short): {stats.clips_skipped_too_short}")
    print(f"Files w/ trim->empty:      {stats.clips_skipped_after_trim}")
    if not args.dry_run:
        print(f"Metadata:       {metadata_path}")
        print(f"Clips folder:   {clips_dir}")

    if stats.clips_written == 0:
        print("\nWARNING: 0 clips written. Most likely causes:")
        print("- Your files are shorter than clip_seconds")
        print("- trim_db trimmed everything (too aggressive)")
        print("- min_rms_db is too high (skipping all chunks)")
        print("- decoding failed (ffmpeg missing for mp3)")
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())