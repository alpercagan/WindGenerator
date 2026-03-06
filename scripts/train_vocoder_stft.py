from __future__ import annotations

import argparse
import collections
import glob
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import warnings
warnings.filterwarnings("ignore", message="An output with one or more elements was resized")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from windgen.mels import LogMelExtractor, MelSpecConfig, load_wav_mono_resample
from windgen.vocoder_tiny import MultiResolutionSTFTLoss, TinyVocoder

TARGET_SAMPLES = 112640   # 440 frames × 256 hop_length
TARGET_FRAMES  = 440


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VocoderDataset(Dataset):
    """Returns (mel, audio) pairs.

    mel:   (128, 440)      normalized log-mel in ~[-1, 1]
    audio: (112640,)       peak-normalized waveform in [-1, 1]
    """

    def __init__(self, clips_dir: str, mel_stats_path: str):
        self.clips: List[Path] = sorted(Path(clips_dir).rglob("*.wav"))
        if not self.clips:
            raise FileNotFoundError(f"No WAV files found in {clips_dir}")
        self.mel_cfg = MelSpecConfig()
        self.extractor = LogMelExtractor(
            config=self.mel_cfg,
            device="cpu",
            normalization="global",
            stats_path=mel_stats_path,
        )

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.clips[idx]
        audio = load_wav_mono_resample(str(path), target_sr=22050)  # (T,)

        # Peak normalize to [-1, 1]
        audio = audio / (audio.abs().max() + 1e-7)

        # Pad or crop to TARGET_SAMPLES
        if audio.shape[0] < TARGET_SAMPLES:
            audio = F.pad(audio, (0, TARGET_SAMPLES - audio.shape[0]))
        else:
            audio = audio[:TARGET_SAMPLES]

        # Compute normalized log-mel -> (1, 128, frames)
        mel = self.extractor(audio)

        # Pad or crop frames to TARGET_FRAMES
        T = mel.shape[-1]
        if T < TARGET_FRAMES:
            mel = F.pad(mel, (0, TARGET_FRAMES - T))
        else:
            mel = mel[..., :TARGET_FRAMES]

        return mel.squeeze(0), audio  # (128, 440), (112640)


# ---------------------------------------------------------------------------
# Resume helper
# ---------------------------------------------------------------------------

def find_latest_checkpoint(out_dir: Path) -> Tuple[str | None, int]:
    ckpts = glob.glob(str(out_dir / "ckpt_step_*.pt"))
    if not ckpts:
        return None, 0

    def step_num(p: str) -> int:
        m = re.search(r"ckpt_step_(\d+)\.pt", p)
        return int(m.group(1)) if m else 0

    ckpts.sort(key=step_num, reverse=True)
    for ckpt_path in ckpts:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "gen" not in ckpt:
                print(f"Skipping {ckpt_path}: no 'gen' key (incompatible checkpoint)")
                continue
            return ckpt_path, step_num(ckpt_path)
        except Exception as e:
            print(f"Skipping {ckpt_path}: failed to load ({e})")
            continue
    return None, 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinyVocoder with STFT loss only")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.expanduser("~/Datasets/wind_clean/clips"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/train_vocoder_stft",
    )
    parser.add_argument(
        "--mel_stats",
        type=str,
        default="outputs/mel_stats.json",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset & DataLoader
    dataset = VocoderDataset(clips_dir=args.data_dir, mel_stats_path=args.mel_stats)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    # Model + loss
    gen = TinyVocoder(n_mels=128, base_ch=256).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        gen = torch.nn.DataParallel(gen)

    loss_fn = MultiResolutionSTFTLoss(configs=[
        (512,  50,  240),
        (1024, 120, 600),
        (2048, 240, 1200),
    ])

    opt_G = torch.optim.AdamW(gen.parameters(), lr=2e-4, betas=(0.8, 0.99))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt_G, gamma=0.99999)

    start_step = 0
    if args.resume:
        ckpt_path, start_step = find_latest_checkpoint(out_dir)
        if ckpt_path is None:
            print("No compatible checkpoint found, starting from scratch.")
            start_step = 0
        else:
            print(f"Resuming from {ckpt_path} (step {start_step})")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state = gen.module if hasattr(gen, "module") else gen
            state.load_state_dict(ckpt["gen"])
            opt_G.load_state_dict(ckpt["opt_G"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])

    # -----------------------------------------------------------------------
    # Startup assertions
    # -----------------------------------------------------------------------
    print("\n--- MelSpecConfig ---")
    for k, v in dataset.mel_cfg.__dict__.items():
        print(f"  {k}: {v}")
    print()

    _mel_batch, _audio_batch = next(iter(loader))
    _mel_dev = _mel_batch.to(device)

    print(f"mel batch shape:   {_mel_batch.shape}")
    assert _mel_batch.shape == (args.batch_size, 128, 440), \
        f"Expected mel batch ({args.batch_size}, 128, 440), got {_mel_batch.shape}"

    print(f"audio batch shape: {_audio_batch.shape}")
    assert _audio_batch.shape == (args.batch_size, TARGET_SAMPLES), \
        f"Expected audio batch ({args.batch_size}, {TARGET_SAMPLES}), got {_audio_batch.shape}"

    with torch.no_grad():
        _test_out = gen(_mel_dev)
    print(f"vocoder output shape: {_test_out.shape}")
    assert _test_out.shape == (args.batch_size, TARGET_SAMPLES), \
        f"Expected vocoder output ({args.batch_size}, {TARGET_SAMPLES}), got {_test_out.shape}"

    mel_min = _mel_batch.min().item()
    mel_max = _mel_batch.max().item()
    print(f"mel value range: min={mel_min:.2f} max={mel_max:.2f}")
    assert mel_min >= -1.5 and mel_max <= 1.5, \
        f"Mel values out of expected range [-1.5, 1.5]: min={mel_min:.2f} max={mel_max:.2f}"

    print("\nAll assertions passed. Starting training...\n")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    global_step = start_step
    loader_iter = iter(loader)
    loss_history: collections.deque = collections.deque(maxlen=500)

    while global_step < args.max_steps:
        try:
            mel, audio = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            mel, audio = next(loader_iter)

        mel   = mel.to(device)    # (B, 128, 440)
        audio = audio.to(device)  # (B, 112640)

        gen.train()

        y_hat = gen(mel)
        loss = loss_fn(y_hat, audio)

        opt_G.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 5.0)
        opt_G.step()
        scheduler.step()

        global_step += 1
        loss_history.append(loss.item())

        if global_step % 50 == 0:
            rolling_avg = sum(loss_history) / len(loss_history)
            current_lr = scheduler.get_last_lr()[0]
            flag = " WARNING" if loss.item() != loss.item() or loss.item() > 1e4 else ""
            print(
                f"Step {global_step:06d}/{args.max_steps} | "
                f"stft={loss.item():.4f}  avg500={rolling_avg:.4f}  "
                f"lr={current_lr:.2e}{flag}"
            )

        # Preview audio every 500 steps
        if global_step % 500 == 0:
            gen.eval()
            with torch.no_grad():
                preview_wav = gen(mel[0:1])  # (1, T)
            preview_path = out_dir / f"preview_step_{global_step:06d}.wav"
            wav_np = preview_wav.squeeze(0).detach().cpu().float().numpy()
            sf.write(str(preview_path), wav_np, samplerate=22050)
            print(f"Saved preview: {preview_path}")

        # Checkpoint every 2000 steps
        if global_step % 2000 == 0:
            ckpt_path = out_dir / f"ckpt_step_{global_step:06d}.pt"
            _state = gen.module if hasattr(gen, "module") else gen
            torch.save({
                "gen":           _state.state_dict(),
                "opt_G":         opt_G.state_dict(),
                "scheduler":     scheduler.state_dict(),
                "step":          global_step,
                "mel_cfg":       dataset.mel_cfg.__dict__,
                "target_frames": TARGET_FRAMES,
            }, str(ckpt_path))
            print(f"Saved checkpoint: {ckpt_path}")

    # Final checkpoint (generator only)
    final_path = out_dir / "final_model.pt"
    _state = gen.module if hasattr(gen, "module") else gen
    torch.save({
        "gen":           _state.state_dict(),
        "step":          global_step,
        "mel_cfg":       dataset.mel_cfg.__dict__,
        "target_frames": TARGET_FRAMES,
    }, str(final_path))
    print(f"\nTraining complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
