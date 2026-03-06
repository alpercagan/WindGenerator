# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Dataset Pipeline (run in order)
```bash
# 1. Preprocess raw audio → standardized 5.12s clips
python scripts/prepare_dataset.py --input_dir /path/to/raw/audio --output_dir ~/Datasets/wind_clean

# 2. Compute global mel statistics (required before training)
python scripts/compute_mel_stats.py --data_dir ~/Datasets/wind_clean --out outputs/mel_stats.json

# 3. Train diffusion model
python scripts/train_diffusion.py

# 4. Generate wind audio samples
python scripts/sample_audio.py --ckpt outputs/train_ddpm/final_model.pt --num 5 --steps 1000
```

### Debugging
```bash
python scripts/inspect_batch.py   # Check dataset batch shapes and mel stats
python scripts/audit_dataset.py   # Data quality analysis
```

### Lint / Type Check
```bash
flake8 src/windgen
mypy src/windgen
```

## Architecture

**Goal:** Generate synthetic wind audio via spectrogram diffusion. Wind lacks strict periodic structure, so log-mel spectrograms are treated as 2D images and generated with a DDPM model.

### Pipeline
1. **Prepare** raw audio → 22,050 Hz mono WAV clips (5.12s, RMS-normalized to -20 dBFS, 50% overlap)
2. **Compute stats** → `outputs/mel_stats.json` (global μ/σ across dataset clips)
3. **Train** a small 2D UNet with DDPM on normalized log-mel spectrograms (shape: 1×128×440)
4. **Sample** by running the DDPM reverse process → denormalize → Griffin-Lim vocoding → WAV

### Key Design Decisions
- **Global normalization** (switched from per-clip in commit `1c8bc9f`): `LogMelExtractor` clamps to ±4σ then rescales to ~[-1, 1] using stats from `mel_stats.json`. Consistency requires recomputing stats if the dataset changes.
- **Fixed shape:** All spectrograms are padded/cropped to exactly 440 frames (128 mel bins × 440 frames).
- **UNet config:** Small model — 16/32/64 channel blocks, no attention, gradient checkpointing enabled for MPS/CPU efficiency. Effective batch size 8 via gradient accumulation (batch=1, accumulate=8).
- **Vocoder:** Griffin-Lim (32 iterations) was the initial inference path and produced recognizable wind texture, but the output was too metallic/artificial. The plan has shifted to training `TinyVocoder` (defined in `vocoder_tiny.py`) on the same dataset so the vocoder learns real wind audio characteristics. Griffin-Lim is deprecated for final output.
- **Device:** Auto-detects Apple MPS, falls back to CPU. No explicit CUDA path.

### Source Layout (`src/windgen/`)
| File | Role |
|---|---|
| `mels.py` | `MelSpecConfig` dataclass + `LogMelExtractor` (normalization logic) |
| `dataset.py` | `WindMelDataset` — loads WAVs, applies global normalization, pads to 440 frames |
| `vocoder_tiny.py` | `TinyVocoder` (mel→waveform) + `MultiResolutionSTFTLoss` |
| `viz.py` | `save_mel_grid()` — saves spectrogram PNG grids |

### Data Paths (conventions)
- Raw audio → `~/Datasets/wind_clean/clips/*.wav` + `metadata.csv`
- Mel stats → `outputs/mel_stats.json`
- Checkpoints → `outputs/train_ddpm/ckpt_step_XXXXXX.pt`
- Generated audio → `outputs/samples_audio/`


## Current Status & Next Steps

### What is done
- Dataset cleaned and preprocessed (`prepare_dataset.py`)
- Mel stats computed (`outputs/mel_stats.json`)
- Diffusion model trained (`outputs/train_ddpm/final_model.pt`)
- Griffin-Lim sampling confirmed wind-like spectral structure (but metallic output)
- `TinyVocoder` architecture exists in `vocoder_tiny.py` but is untrained

### What is NOT done yet
- `TinyVocoder` has never been trained
- No `train_vocoder.py` script exists yet
- No end-to-end pipeline connecting diffusion output → vocoder → WAV
- Code consistency across pipeline has not been formally verified

### Immediate Priority
Before writing any new code: audit the full pipeline for data format consistency.
The vocoder must receive mel spectrograms in the exact same format the diffusion model outputs.
Any mismatch in normalization, shape, or scale will silently produce bad audio.

### Known Risk
The biggest failure mode is a normalization mismatch:
- Diffusion model outputs normalized mels (clamped ±4σ, rescaled ~[-1,1])
- Vocoder must denormalize before reconstructing audio
- This boundary must be explicitly validated before training starts