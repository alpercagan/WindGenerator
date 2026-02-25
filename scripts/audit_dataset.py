#!/usr/bin/env python3
"""
audit_dataset.py

Audit a prepared wind dataset.

Expected layout:
  <data_dir>/
    clips/
      wind_000001.wav ...
    metadata.csv

Outputs to:
  <repo_root>/outputs/audit/
    audit_summary.txt
    audit_report.csv
    audit_hist_rms.png
    audit_hist_peak.png
    audit_source_counts.csv
    audit_samples/  (optional)

Usage:
  python scripts/audit_dataset.py --data_dir ~/Datasets/wind_clean --num_listen 30

Optional deep check:
  python scripts/audit_dataset.py --data_dir ~/Datasets/wind_clean --recompute_audio_stats --limit_audio_scan 300
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm


def dbfs_from_rms(rms: float, eps: float = 1e-12) -> float:
    return 20.0 * math.log10(max(float(rms), eps))


def compute_rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def peak_abs(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)) + 1e-12)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit a prepared wind dataset (clips + metadata).")
    ap.add_argument("--data_dir", type=str, required=True, help="Path to wind_clean directory.")
    ap.add_argument("--num_listen", type=int, default=0, help="Copy N random clips to outputs/audit/audit_samples.")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for sampling.")
    ap.add_argument("--max_rows", type=int, default=0, help="Debug: limit number of metadata rows (0 = all).")
    ap.add_argument("--target_rms_db", type=float, default=-20.0, help="Target RMS dBFS used in preprocessing.")
    ap.add_argument("--quiet_db", type=float, default=-45.0, help="Threshold for 'too quiet' RMS dBFS.")
    ap.add_argument("--near_clip", type=float, default=0.98, help="Threshold for near-clipping peak.")
    ap.add_argument("--recompute_audio_stats", action="store_true",
                    help="Recompute rms/peak/duration from audio files (slower) and compare to metadata.")
    ap.add_argument("--limit_audio_scan", type=int, default=0,
                    help="Limit number of audio files to scan when recomputing (0 = all).")
    return ap.parse_args()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(12):
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return Path.cwd().resolve()


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    required = {"clip_filename", "source_path", "out_sr", "clip_start_sec", "clip_duration_sec", "rms_dbfs", "peak"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv missing required columns: {sorted(missing)}")
    return df


def summarize_numeric(series: pd.Series) -> Dict[str, float]:
    s = series.dropna().astype(float)
    if len(s) == 0:
        return {"count": 0}
    return {
        "count": int(s.count()),
        "min": float(s.min()),
        "p05": float(s.quantile(0.05)),
        "median": float(s.median()),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if s.count() > 1 else 0.0,
    }


def save_hist(values: np.ndarray, title: str, xlabel: str, out_path: Path, bins: int = 60) -> None:
    plt.figure()
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(os.path.expanduser(args.data_dir)).resolve()
    clips_dir = data_dir / "clips"
    metadata_path = data_dir / "metadata.csv"

    if not data_dir.exists():
        raise SystemExit(f"ERROR: data_dir does not exist: {data_dir}")
    if not clips_dir.exists():
        raise SystemExit(f"ERROR: missing clips/ folder: {clips_dir}")
    if not metadata_path.exists():
        raise SystemExit(f"ERROR: missing metadata.csv: {metadata_path}")

    repo_root = find_repo_root(Path.cwd())
    out_dir = repo_root / "outputs" / "audit"
    samples_dir = out_dir / "audit_samples"
    ensure_dir(out_dir)
    if args.num_listen > 0:
        ensure_dir(samples_dir)

    df = load_metadata(metadata_path)
    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    # File existence check
    clip_paths = [clips_dir / fn for fn in df["clip_filename"].astype(str).tolist()]
    exists_mask = np.array([p.exists() for p in clip_paths], dtype=bool)
    missing_files = int((~exists_mask).sum())
    present_files = int(exists_mask.sum())

    # Basic stats from metadata
    df["rms_dbfs"] = df["rms_dbfs"].astype(float)
    df["peak"] = df["peak"].astype(float)
    df["clip_duration_sec"] = df["clip_duration_sec"].astype(float)
    df["out_sr"] = df["out_sr"].astype(int)
    df["clip_start_sec"] = df["clip_start_sec"].astype(float)

    # How close to target RMS are we?
    df["rms_err_db"] = df["rms_dbfs"] - float(args.target_rms_db)

    rms_stats = summarize_numeric(df["rms_dbfs"])
    rms_err_stats = summarize_numeric(df["rms_err_db"])
    peak_stats = summarize_numeric(df["peak"])
    dur_stats = summarize_numeric(df["clip_duration_sec"])
    sr_stats = summarize_numeric(df["out_sr"])

    too_quiet = int((df["rms_dbfs"] < float(args.quiet_db)).sum())
    near_clip = int((df["peak"] >= float(args.near_clip)).sum())
    dup_filenames = int(df["clip_filename"].duplicated().sum())

    # Source counts (clips per raw file)
    source_counts = (
        df.groupby("source_path")["clip_filename"]
        .count()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"clip_filename": "num_clips"})
    )
    source_counts.to_csv(out_dir / "audit_source_counts.csv", index=False)

    # Plot histograms
    try:
        save_hist(df["rms_dbfs"].to_numpy(), "RMS dBFS distribution (metadata)", "rms_dbfs",
                  out_dir / "audit_hist_rms.png")
        save_hist(df["peak"].to_numpy(), "Peak abs distribution (metadata)", "peak_abs",
                  out_dir / "audit_hist_peak.png")
        save_hist(df["rms_err_db"].to_numpy(), f"RMS error from target ({args.target_rms_db} dBFS)", "rms_db - target_db",
                  out_dir / "audit_hist_rms_error.png")
    except Exception as e:
        print(f"WARNING: failed to save histograms: {type(e).__name__}: {e}")

    # Optional: recompute from audio WAVs and compare
    if args.recompute_audio_stats:
        print("Recomputing stats from audio files (this can take a bit)...")
        rows = []
        indices = [i for i, ok in enumerate(exists_mask) if ok]
        if args.limit_audio_scan and args.limit_audio_scan > 0:
            indices = indices[: args.limit_audio_scan]

        for i in tqdm(indices, desc="Audio scan", unit="clip"):
            p = clip_paths[i]
            meta_row = df.iloc[i]
            try:
                audio, sr = sf.read(str(p), always_2d=False)
                audio = np.asarray(audio, dtype=np.float32)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)

                rms_db_audio = dbfs_from_rms(compute_rms(audio))
                peak_audio = peak_abs(audio)
                dur_audio = float(len(audio)) / float(sr)

                rows.append({
                    "clip_filename": meta_row["clip_filename"],
                    "sr_audio": int(sr),
                    "duration_audio": dur_audio,
                    "rms_dbfs_audio": rms_db_audio,
                    "peak_audio": peak_audio,
                    "sr_meta": int(meta_row["out_sr"]),
                    "duration_meta": float(meta_row["clip_duration_sec"]),
                    "rms_dbfs_meta": float(meta_row["rms_dbfs"]),
                    "peak_meta": float(meta_row["peak"]),
                    "rms_abs_err": abs(rms_db_audio - float(meta_row["rms_dbfs"])),
                    "peak_abs_err": abs(peak_audio - float(meta_row["peak"])),
                    "dur_abs_err": abs(dur_audio - float(meta_row["clip_duration_sec"])),
                })
            except Exception:
                continue

        if rows:
            df_cmp = pd.DataFrame(rows)
            df_cmp.to_csv(out_dir / "audit_recomputed_compare.csv", index=False)

    # Copy random samples for listening
    if args.num_listen and args.num_listen > 0 and present_files > 0:
        present_indices = [i for i, ok in enumerate(exists_mask) if ok]
        n = min(args.num_listen, len(present_indices))
        chosen = random.sample(present_indices, n)
        for i in chosen:
            src = clip_paths[i]
            dst = samples_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)

    # Write a CSV report
    report_path = out_dir / "audit_report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["data_dir", str(data_dir)])
        w.writerow(["metadata_rows", int(len(df))])
        w.writerow(["clips_present", present_files])
        w.writerow(["clips_missing", missing_files])
        w.writerow(["duplicate_filenames", dup_filenames])
        w.writerow([f"count_rms_below_{args.quiet_db}dBFS", too_quiet])
        w.writerow([f"count_peak_ge_{args.near_clip}", near_clip])

        for k, v in rms_stats.items():
            w.writerow([f"rms_dbfs_{k}", v])
        for k, v in rms_err_stats.items():
            w.writerow([f"rms_err_db_{k}", v])
        for k, v in peak_stats.items():
            w.writerow([f"peak_{k}", v])
        for k, v in dur_stats.items():
            w.writerow([f"duration_sec_{k}", v])
        for k, v in sr_stats.items():
            w.writerow([f"out_sr_{k}", v])

    # Human summary
    summary_path = out_dir / "audit_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Wind Dataset Audit Summary ===\n")
        f.write(f"Data dir: {data_dir}\n")
        f.write(f"Metadata rows: {len(df)}\n")
        f.write(f"Clips present: {present_files}\n")
        f.write(f"Clips missing: {missing_files}\n")
        f.write(f"Duplicate filenames: {dup_filenames}\n\n")

        f.write("RMS dBFS stats (metadata):\n")
        for k, v in rms_stats.items():
            f.write(f"  {k:>8}: {v}\n")
        f.write("\nRMS error from target (rms_dbfs - target_rms_db):\n")
        f.write(f"  target_rms_db: {args.target_rms_db}\n")
        for k, v in rms_err_stats.items():
            f.write(f"  {k:>8}: {v}\n")

        f.write("\nPeak stats (metadata):\n")
        for k, v in peak_stats.items():
            f.write(f"  {k:>8}: {v}\n")

        f.write("\nDuration stats (metadata):\n")
        for k, v in dur_stats.items():
            f.write(f"  {k:>8}: {v}\n")

        f.write("\nOut SR stats (metadata):\n")
        for k, v in sr_stats.items():
            f.write(f"  {k:>8}: {v}\n")

        f.write("\nHeuristics:\n")
        f.write(f"  Too quiet (< {args.quiet_db} dBFS): {too_quiet}\n")
        f.write(f"  Near clipping (peak >= {args.near_clip}): {near_clip}\n\n")

        f.write("Top sources by number of clips:\n")
        topn = source_counts.head(10)
        for _, r in topn.iterrows():
            f.write(f"  {r['num_clips']:>6}  {r['source_path']}\n")

        f.write("\nOutputs:\n")
        f.write(f"  {report_path}\n")
        f.write(f"  {summary_path}\n")
        f.write(f"  {out_dir / 'audit_hist_rms.png'}\n")
        f.write(f"  {out_dir / 'audit_hist_peak.png'}\n")
        f.write(f"  {out_dir / 'audit_hist_rms_error.png'}\n")
        f.write(f"  {out_dir / 'audit_source_counts.csv'}\n")
        if args.num_listen > 0:
            f.write(f"  {samples_dir}  (listening samples)\n")
        if args.recompute_audio_stats:
            f.write(f"  {out_dir / 'audit_recomputed_compare.csv'}  (audio-vs-metadata comparison)\n")

    print("\nAudit complete.")
    print(f"- Summary: {summary_path}")
    print(f"- Report:  {report_path}")
    print(f"- Plots:   {out_dir / 'audit_hist_rms.png'}, {out_dir / 'audit_hist_peak.png'}, {out_dir / 'audit_hist_rms_error.png'}")
    print(f"- Source counts: {out_dir / 'audit_source_counts.csv'}")
    if args.num_listen > 0:
        print(f"- Listening samples: {samples_dir}")
    if args.recompute_audio_stats:
        print(f"- Audio comparison: {out_dir / 'audit_recomputed_compare.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())