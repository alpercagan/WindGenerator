#!/usr/bin/env python3
"""End-to-end wind audio generation: diffusion mel → vocoder → 10s WAV."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from diffusers import UNet2DModel, DDPMScheduler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from windgen.vocoder_tiny import TinyVocoder  # noqa: E402

SR = 22050
MEL_H, MEL_W = 128, 440
# 440 frames × 256 hop_length = 112640 samples ≈ 5.12 s per vocoder segment
SEGMENT_SAMPLES = MEL_W * 256  # 112640
CROSSFADE_SAMPLES = 11025      # 0.5 s at 22050 Hz


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate wind audio via diffusion + GAN vocoder")
    ap.add_argument("--diffusion_ckpt", type=str, default="outputs/train_ddpm/final_model.pt",
                    help="Path to diffusion final_model.pt")
    ap.add_argument("--vocoder_ckpt", type=str, required=True,
                    help="Path to GAN vocoder checkpoint (latest_checkpoint.pt or similar)")
    ap.add_argument("--mel_stats", type=str, default="outputs/mel_stats.json",
                    help="Path to mel_stats.json")
    ap.add_argument("--output_dir", type=str, default="outputs/generated",
                    help="Directory to write generated WAV files")
    ap.add_argument("--num_clips", type=int, default=3,
                    help="Number of ~10 s clips to generate")
    ap.add_argument("--ddpm_steps", type=int, default=50,
                    help="DDPM reverse-diffusion steps (fewer = faster, lower quality)")
    ap.add_argument("--device", type=str, default="auto",
                    help="Device: auto (cuda → cpu), cuda, or cpu")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _build_unet(device: torch.device) -> UNet2DModel:
    """Construct UNet2DModel matching the training configuration (~11M params)."""
    return UNet2DModel(
        sample_size=(MEL_H, MEL_W),
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        mid_block_type="UNetMidBlock2D",
        norm_num_groups=8,
    ).to(device)


def load_diffusion_model(ckpt_path: Path, device: torch.device) -> UNet2DModel:
    model = _build_unet(device)
    payload = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = payload["model"] if "model" in payload else payload
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_vocoder(ckpt_path: Path, device: torch.device) -> TinyVocoder:
    gen = TinyVocoder(n_mels=MEL_H, base_ch=256).to(device)
    payload = torch.load(str(ckpt_path), map_location="cpu")
    # GAN checkpoint saves generator under "gen"; fall back to bare state_dict
    state_dict = payload["gen"] if "gen" in payload else payload
    gen.load_state_dict(state_dict)
    gen.eval()
    return gen


# ---------------------------------------------------------------------------
# Per-clip inference
# ---------------------------------------------------------------------------

def run_diffusion(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    device: torch.device,
) -> torch.Tensor:
    """
    Run DDPM reverse process.

    Returns: (1, 128, 440) normalized mel in ~[-1, 1].
    """
    x = torch.randn((1, 1, MEL_H, MEL_W), device=device)
    with torch.no_grad():
        for t in scheduler.timesteps:
            pred_noise = model(x, t).sample
            x = scheduler.step(pred_noise, t, x).prev_sample

    x = x.squeeze(1)          # (1, 1, 128, 440) → (1, 128, 440)
    x = x.clamp(-1.0, 1.0)
    return x


def run_vocoder(gen: TinyVocoder, mel: torch.Tensor) -> np.ndarray:
    """
    Run TinyVocoder on a normalized mel spectrogram.

    Args:
        mel: (1, 128, 440) in ~[-1, 1]

    Returns:
        waveform as float32 numpy array of shape (SEGMENT_SAMPLES,)
    """
    with torch.no_grad():
        wav = gen(mel)  # (1, T)
    return wav.squeeze(0).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Crossfade helpers
# ---------------------------------------------------------------------------

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9)


def crossfade_with_self(segment: np.ndarray, crossfade_len: int) -> np.ndarray:
    """
    Overlap-add a segment with a copy of itself, matching RMS before joining.

    Layout:
      copy1: [0 .. n)               fade-out applied to last crossfade_len samples
      copy2: [n - crossfade_len ..) fade-in  applied to first crossfade_len samples

    Output length = 2 * n - crossfade_len  (≈ 9.7 s for n=112640, fade=11025)

    Args:
        segment:       float32 array, shape (SEGMENT_SAMPLES,)
        crossfade_len: number of samples for the linear crossfade overlap

    Returns:
        float32 array of length 2 * len(segment) - crossfade_len
    """
    n = len(segment)
    if crossfade_len >= n:
        raise ValueError(f"crossfade_len ({crossfade_len}) must be < segment length ({n})")

    copy1 = segment.copy()
    copy2 = segment.copy()

    # Normalize second copy's RMS to match the first
    rms1 = _rms(copy1)
    rms2 = _rms(copy2)
    copy2 = copy2 * (rms1 / rms2)

    # Linear fade envelopes
    fade_out = np.linspace(1.0, 0.0, crossfade_len, dtype=np.float32)
    fade_in  = np.linspace(0.0, 1.0, crossfade_len, dtype=np.float32)

    # Apply fades in-place at the join region
    copy1[-crossfade_len:] *= fade_out
    copy2[:crossfade_len]  *= fade_in

    # Allocate output and overlap-add
    out_len = n + n - crossfade_len
    out = np.zeros(out_len, dtype=np.float32)
    out[:n] += copy1
    out[n - crossfade_len : n - crossfade_len + n] += copy2

    return out


def normalize_peak(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) + 1e-12
    if peak > target_peak:
        audio = audio * (target_peak / peak)
    return audio


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    print(f"Device: {device}")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    diff_ckpt  = Path(args.diffusion_ckpt).resolve()
    voc_ckpt   = Path(args.vocoder_ckpt).resolve()
    stats_path = Path(args.mel_stats).resolve()

    for path, label in [
        (diff_ckpt,  "--diffusion_ckpt"),
        (voc_ckpt,   "--vocoder_ckpt"),
        (stats_path, "--mel_stats"),
    ]:
        if not path.exists():
            raise SystemExit(f"File not found ({label}): {path}")

    stats = json.loads(stats_path.read_text())
    print(
        f"Mel stats: mean_median={stats['logmel_mean_median']:.4f}, "
        f"std_median={stats['logmel_std_median']:.4f}"
    )

    print(f"Loading diffusion model from {diff_ckpt} ...")
    diff_model = load_diffusion_model(diff_ckpt, device)

    print(f"Loading vocoder from {voc_ckpt} ...")
    vocoder = load_vocoder(voc_ckpt, device)

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(args.ddpm_steps)
    print(f"DDPM scheduler: {args.ddpm_steps} inference steps")

    expected_segment_len = 2 * SEGMENT_SAMPLES - CROSSFADE_SAMPLES
    print(
        f"\nPipeline per clip:"
        f"\n  Segment:  {SEGMENT_SAMPLES} samples ({SEGMENT_SAMPLES / SR:.3f} s)"
        f"\n  Crossfade: {CROSSFADE_SAMPLES} samples ({CROSSFADE_SAMPLES / SR:.3f} s)"
        f"\n  Output:   {expected_segment_len} samples ({expected_segment_len / SR:.3f} s)"
        f"\n  Output dir: {out_dir}"
    )

    for i in range(args.num_clips):
        clip_num = i + 1
        print(f"\n[Clip {clip_num}/{args.num_clips}]")

        # Step 1 & 2: diffusion → mel (1, 128, 440)
        print("  Step 1/4  Diffusion sampling ...")
        mel = run_diffusion(diff_model, scheduler, device)
        print(f"            mel shape={tuple(mel.shape)}  "
              f"range=[{mel.min():.3f}, {mel.max():.3f}]")

        # Step 3: vocoder → waveform segment
        print("  Step 2/4  Vocoder synthesis ...")
        segment = run_vocoder(vocoder, mel)
        print(f"            segment={len(segment)} samples ({len(segment) / SR:.3f} s)  "
              f"range=[{segment.min():.3f}, {segment.max():.3f}]  "
              f"rms={_rms(segment):.4f}")

        # Steps 4 & 5: RMS match + crossfade overlap-add
        print(f"  Step 3/4  Crossfade overlap-add ({CROSSFADE_SAMPLES} samples) ...")
        clip = crossfade_with_self(segment, CROSSFADE_SAMPLES)
        clip = normalize_peak(clip)
        print(f"            output={len(clip)} samples ({len(clip) / SR:.3f} s)  "
              f"peak={float(np.max(np.abs(clip))):.4f}")

        # Step 6: save
        wav_path = out_dir / f"generated_clip_{clip_num:03d}.wav"
        print(f"  Step 4/4  Writing {wav_path}")
        sf.write(str(wav_path), clip, SR, subtype="PCM_16")
        print(f"  Saved:    {wav_path}")

    print(f"\nDone. {args.num_clips} clip(s) written to {out_dir}/")


if __name__ == "__main__":
    main()
