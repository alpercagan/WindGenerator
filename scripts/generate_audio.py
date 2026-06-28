#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from diffusers import UNet2DModel, DDPMScheduler

SR = 22050
MEL_H, MEL_W = 128, 440
SEGMENT_SAMPLES = MEL_W * 256  # 112640 ≈ 5.12 s
CROSSFADE_SAMPLES = 11025      # 0.5 s at 22050 Hz


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate wind audio via diffusion + Griffin-Lim")
    ap.add_argument("--diffusion_ckpt", type=str, default="outputs/train_ddpm/final_model.pt",
                    help="Path to diffusion final_model.pt")
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

def _infer_unet_cfg(state_dict: dict) -> tuple:
    """Infer (block_out_channels, layers_per_block) from checkpoint weights."""
    n_levels = max(int(k.split(".")[1]) for k in state_dict if k.startswith("down_blocks.")) + 1
    c0 = state_dict["conv_in.weight"].shape[0]
    block_out_channels = tuple(c0 * (2 ** i) for i in range(n_levels))
    layers_per_block = max(
        int(k.split(".")[3]) for k in state_dict if k.startswith("down_blocks.0.resnets.")
    ) + 1
    return block_out_channels, layers_per_block


def _build_unet(
    block_out_channels: tuple,
    layers_per_block: int,
    device: torch.device,
) -> UNet2DModel:
    n = len(block_out_channels)
    return UNet2DModel(
        sample_size=(MEL_H, MEL_W),
        in_channels=1,
        out_channels=1,
        layers_per_block=layers_per_block,
        block_out_channels=block_out_channels,
        down_block_types=("DownBlock2D",) * n,
        up_block_types=("UpBlock2D",) * n,
        mid_block_type="UNetMidBlock2D",
        norm_num_groups=8,
    ).to(device)


def load_diffusion_model(ckpt_path: Path, device: torch.device) -> UNet2DModel:
    payload = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = payload["model"] if "model" in payload else payload
    block_out_channels, layers_per_block = _infer_unet_cfg(state_dict)
    print(f"  Detected arch: block_out_channels={block_out_channels}, layers_per_block={layers_per_block}")
    model = _build_unet(block_out_channels, layers_per_block, device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Per-clip inference
# ---------------------------------------------------------------------------

def run_diffusion(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    device: torch.device,
) -> torch.Tensor:
    """Run DDPM reverse process. Returns (1, 128, 440) normalized mel in ~[-1, 1]."""
    x = torch.randn((1, 1, MEL_H, MEL_W), device=device)
    with torch.no_grad():
        for t in scheduler.timesteps:
            pred_noise = model(x, t).sample
            x = scheduler.step(pred_noise, t, x).prev_sample

    x = x.squeeze(1)   # (1, 1, 128, 440) → (1, 128, 440)
    x = x.clamp(-1.0, 1.0)
    return x


def mel_to_audio(mel: torch.Tensor, mel_mean: float, mel_std: float) -> np.ndarray:
    eps = 1e-5
    # Reverse [-1, 1] scaling → z-score → linear amplitude
    x = mel * (4.0 * mel_std) + mel_mean
    x = torch.exp(x) - eps
    x = x.clamp(min=0.0).cpu()

    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=513,
        n_mels=128,
        sample_rate=22050,
        f_min=20.0,
        f_max=11025.0,
    )
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        power=2.0,
        n_iter=64,
    )

    # x is already denormalized linear mel power spectrogram (128, T)
    x = inverse_mel(x)   # (513, T)
    wav = griffin_lim(x) # (T,)
    return wav.squeeze(0).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Crossfade helpers
# ---------------------------------------------------------------------------

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9)


def crossfade_with_self(segment: np.ndarray, crossfade_len: int) -> np.ndarray:
    n = len(segment)
    if crossfade_len >= n:
        raise ValueError(f"crossfade_len ({crossfade_len}) must be < segment length ({n})")

    copy1 = segment.copy()
    copy2 = segment.copy()

    rms1 = _rms(copy1)
    rms2 = _rms(copy2)
    copy2 = copy2 * (rms1 / rms2)

    fade_out = np.linspace(1.0, 0.0, crossfade_len, dtype=np.float32)
    fade_in  = np.linspace(0.0, 1.0, crossfade_len, dtype=np.float32)

    copy1[-crossfade_len:] *= fade_out
    copy2[:crossfade_len]  *= fade_in

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
    stats_path = Path(args.mel_stats).resolve()

    for path, label in [
        (diff_ckpt,  "--diffusion_ckpt"),
        (stats_path, "--mel_stats"),
    ]:
        if not path.exists():
            raise SystemExit(f"File not found ({label}): {path}")

    stats = json.loads(stats_path.read_text())
    mel_mean = stats["logmel_mean_median"]
    mel_std  = stats["logmel_std_median"]
    print(f"Mel stats: mean={mel_mean:.4f}, std={mel_std:.4f}")

    print(f"Loading diffusion model from {diff_ckpt} ...")
    diff_model = load_diffusion_model(diff_ckpt, device)

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(args.ddpm_steps)
    print(f"DDPM scheduler: {args.ddpm_steps} inference steps")

    expected_segment_len = 2 * SEGMENT_SAMPLES - CROSSFADE_SAMPLES
    print(
        f"\nPipeline per clip:"
        f"\n  Segment:   {SEGMENT_SAMPLES} samples ({SEGMENT_SAMPLES / SR:.3f} s)"
        f"\n  Crossfade: {CROSSFADE_SAMPLES} samples ({CROSSFADE_SAMPLES / SR:.3f} s)"
        f"\n  Output:    {expected_segment_len} samples ({expected_segment_len / SR:.3f} s)"
        f"\n  Output dir: {out_dir}"
    )

    for i in range(args.num_clips):
        clip_num = i + 1
        print(f"\n[Clip {clip_num}/{args.num_clips}]")

        print("  Step 1/3  Diffusion sampling ...")
        mel = run_diffusion(diff_model, scheduler, device)
        print(f"            mel shape={tuple(mel.shape)}  "
              f"range=[{mel.min():.3f}, {mel.max():.3f}]")

        print("  Step 2/3  Griffin-Lim synthesis ...")
        segment = mel_to_audio(mel, mel_mean, mel_std)
        print(f"            segment={len(segment)} samples ({len(segment) / SR:.3f} s)  "
              f"range=[{segment.min():.3f}, {segment.max():.3f}]  "
              f"rms={_rms(segment):.4f}")

        print(f"  Step 3/3  Crossfade + normalize, writing WAV ...")
        clip = crossfade_with_self(segment, CROSSFADE_SAMPLES)
        clip = normalize_peak(clip)
        wav_path = out_dir / f"generated_clip_{clip_num:03d}.wav"
        sf.write(str(wav_path), clip, SR, subtype="PCM_16")
        print(f"  Saved:    {wav_path}")

    print(f"\nDone. {args.num_clips} clip(s) written to {out_dir}/")


if __name__ == "__main__":
    main()
