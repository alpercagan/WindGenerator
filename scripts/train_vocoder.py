from __future__ import annotations

import argparse
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

# MPS async safety
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from windgen.mels import LogMelExtractor, MelSpecConfig, load_wav_mono_resample
from windgen.vocoder_tiny import LiteMPD, MultiResolutionSTFTLoss, TinyVocoder

TARGET_SAMPLES = 112640   # 440 frames × 256 hop_length
TARGET_FRAMES  = 440
MEL_STATS_PATH = "outputs/mel_stats.json"
OUT_DIR        = Path("outputs/train_vocoder")
GAN_WARMUP     = 5000     # steps before enabling adversarial + feature-matching losses


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VocoderDataset(Dataset):
    """Returns (mel, audio) pairs.

    mel:   (128, 440)      normalized log-mel in ~[-1, 1]
    audio: (112640,)       peak-normalized waveform in [-1, 1]
    """

    def __init__(self, clips_dir: str, mel_stats_path: str = MEL_STATS_PATH):
        self.clips: List[Path] = sorted(Path(clips_dir).glob("*.wav"))
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
    parser = argparse.ArgumentParser(description="Train TinyVocoder on wind audio")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.expanduser("~/Datasets/wind_clean/clips"),
    )
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--max_steps", type=int, default=100000)
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Dataset & DataLoader
    dataset = VocoderDataset(clips_dir=args.data_dir)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True,
                        pin_memory=False, persistent_workers=False)

    # Models
    gen  = TinyVocoder(n_mels=128, base_ch=256).to(device)
    disc = LiteMPD().to(device)

    loss_fn = MultiResolutionSTFTLoss(configs=[
        (512,  50,  240),
        (1024, 120, 600),
        (2048, 240, 1200),
    ])

    opt_G = torch.optim.AdamW(gen.parameters(),  lr=2e-4, betas=(0.8, 0.99))
    opt_D = torch.optim.AdamW(disc.parameters(), lr=2e-4, betas=(0.8, 0.99))

    start_step = 0
    if args.resume:
        ckpt_path, start_step = find_latest_checkpoint(OUT_DIR)
        if ckpt_path is None:
            print("No compatible checkpoint found, starting from scratch.")
            start_step = 0
        else:
            print(f"Resuming from {ckpt_path} (step {start_step})")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            gen.load_state_dict(ckpt["gen"])
            disc.load_state_dict(ckpt["disc"])
            opt_G.load_state_dict(ckpt["opt_G"])
            opt_D.load_state_dict(ckpt["opt_D"])

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
    assert _mel_batch.shape == (4, 128, 440), \
        f"Expected mel batch (4, 128, 440), got {_mel_batch.shape}"

    print(f"audio batch shape: {_audio_batch.shape}")
    assert _audio_batch.shape == (4, TARGET_SAMPLES), \
        f"Expected audio batch (4, {TARGET_SAMPLES}), got {_audio_batch.shape}"

    with torch.no_grad():
        _test_out = gen(_mel_dev)
    print(f"vocoder output shape: {_test_out.shape}")
    assert _test_out.shape == (4, TARGET_SAMPLES), \
        f"Expected vocoder output (4, {TARGET_SAMPLES}), got {_test_out.shape}"

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

    while global_step < args.max_steps:
        try:
            mel, audio = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            mel, audio = next(loader_iter)

        mel   = mel.to(device)    # (B, 128, 440)
        audio = audio.to(device)  # (B, 112640)

        gen.train()
        disc.train()

        # === DISCRIMINATOR UPDATE (always runs) ===
        y_hat_d = gen(mel).detach()
        real_out = disc(audio)
        fake_out = disc(y_hat_d)
        d_loss = 0.0
        for (score_real, _), (score_fake, _) in zip(real_out, fake_out):
            d_loss += (score_real - 1).pow(2).mean() + score_fake.pow(2).mean()
        d_loss = d_loss / len(real_out)
        opt_D.zero_grad()
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(disc.parameters(), 5.0)
        opt_D.step()
        if device.type == "mps" and global_step % 10 == 0:
            torch.mps.synchronize()
            torch.mps.empty_cache()

        # === GENERATOR UPDATE ===
        y_hat = gen(mel)
        stft_loss = loss_fn(y_hat, audio)

        if global_step >= GAN_WARMUP:
            fake_out = disc(y_hat)
            real_out = disc(audio.detach())

            adv_loss = 0.0
            fm_loss  = 0.0
            for (score_fake, feats_fake), (_, feats_real) in zip(fake_out, real_out):
                adv_loss = adv_loss + (score_fake - 1).pow(2).mean()
                fm_loss  = fm_loss + sum(
                    F.l1_loss(ff, fr.detach()).clamp(max=10.0)
                    for ff, fr in zip(feats_fake, feats_real)
                ) / len(feats_fake)
            adv_loss = adv_loss / len(fake_out)
            fm_loss  = fm_loss  / len(fake_out)
            g_loss = stft_loss + 1.0 * adv_loss + 2.0 * fm_loss
        else:
            adv_loss = torch.tensor(0.0, device=device)
            fm_loss  = torch.tensor(0.0, device=device)
            g_loss = stft_loss

        opt_G.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 5.0)
        opt_G.step()

        global_step += 1

        if global_step % 50 == 0:
            d_val   = d_loss.item()
            s_val   = stft_loss.item()
            adv_log = adv_loss.item() if global_step >= GAN_WARMUP else 0.0
            fm_log  = fm_loss.item()  if global_step >= GAN_WARMUP else 0.0
            if any(v != v or v > 1e4 for v in (d_val, s_val)):
                print(f"WARNING  Step {global_step:05d}: stft={s_val:.4f} d={d_val:.4f} -- possible instability")
            print(
                f"Step {global_step:05d}/{args.max_steps} | "
                f"stft={s_val:.4f} adv={adv_log:.4f} fm={fm_log:.4f} d={d_val:.4f}"
            )

        # Preview audio every 500 steps
        if global_step % 500 == 0:
            gen.eval()
            with torch.no_grad():
                preview_wav = gen(mel[0:1])  # (1, 112640)
            preview_path = OUT_DIR / f"preview_step_{global_step:06d}.wav"
            wav_np = preview_wav.squeeze(0).detach().cpu().float().numpy()
            sf.write(str(preview_path), wav_np, samplerate=22050)
            print(f"Saved preview: {preview_path}")

        # Checkpoint every 1000 steps
        if global_step % 1000 == 0:
            ckpt_path = OUT_DIR / f"ckpt_step_{global_step:06d}.pt"
            torch.save({
                "gen":           gen.state_dict(),
                "disc":          disc.state_dict(),
                "opt_G":         opt_G.state_dict(),
                "opt_D":         opt_D.state_dict(),
                "step":          global_step,
                "mel_cfg":       dataset.mel_cfg.__dict__,
                "target_frames": TARGET_FRAMES,
            }, str(ckpt_path))
            print(f"Saved checkpoint: {ckpt_path}")

    # Final checkpoint (generator only, no optimizer state)
    final_path = OUT_DIR / "final_model.pt"
    torch.save({
        "gen":           gen.state_dict(),
        "step":          global_step,
        "mel_cfg":       dataset.mel_cfg.__dict__,
        "target_frames": TARGET_FRAMES,
    }, str(final_path))
    print(f"\nTraining complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
