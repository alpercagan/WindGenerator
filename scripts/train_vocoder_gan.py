from __future__ import annotations

import argparse
import collections
import glob
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import shutil

import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import warnings
warnings.filterwarnings("ignore", message="An output with one or more elements was resized")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from windgen.mels import LogMelExtractor, MelSpecConfig, load_wav_mono_resample
from windgen.vocoder_tiny import CombinedDisc, MultiResolutionSTFTLoss, TinyVocoder

TARGET_SAMPLES = 112640   # 440 frames × 256 hop_length
TARGET_FRAMES  = 440
GAN_WARMUP     = 1000     # steps of STFT-only before adversarial loss kicks in


# ---------------------------------------------------------------------------
# Dataset (same as train_vocoder_stft.py)
# ---------------------------------------------------------------------------

class VocoderDataset(Dataset):
    """Returns (mel, audio) pairs.

    mel:   (128, 440)  normalized log-mel in ~[-1, 1]
    audio: (112640,)   peak-normalized waveform in [-1, 1]
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
        audio = load_wav_mono_resample(str(path), target_sr=22050)

        audio = audio / (audio.abs().max() + 1e-7)

        if audio.shape[0] < TARGET_SAMPLES:
            audio = F.pad(audio, (0, TARGET_SAMPLES - audio.shape[0]))
        else:
            audio = audio[:TARGET_SAMPLES]

        mel = self.extractor(audio)

        T = mel.shape[-1]
        if T < TARGET_FRAMES:
            mel = F.pad(mel, (0, TARGET_FRAMES - T))
        else:
            mel = mel[..., :TARGET_FRAMES]

        return mel.squeeze(0), audio  # (128, 440), (112640)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def discriminator_loss(
    real_results: List[Tuple[torch.Tensor, List[torch.Tensor]]],
    fake_results: List[Tuple[torch.Tensor, List[torch.Tensor]]],
) -> torch.Tensor:
    """LSGAN discriminator loss averaged over sub-discriminators."""
    loss = torch.tensor(0.0, device=real_results[0][0].device)
    for (real_score, _), (fake_score, _) in zip(real_results, fake_results):
        loss = loss + (1.0 - real_score).pow(2).mean() + fake_score.pow(2).mean()
    return loss / len(real_results)


def generator_adv_loss(
    fake_results: List[Tuple[torch.Tensor, List[torch.Tensor]]],
) -> torch.Tensor:
    """LSGAN generator adversarial loss."""
    loss = torch.tensor(0.0, device=fake_results[0][0].device)
    for (fake_score, _) in fake_results:
        loss = loss + (1.0 - fake_score).pow(2).mean()
    return loss / len(fake_results)


def feature_matching_loss(
    real_results: List[Tuple[torch.Tensor, List[torch.Tensor]]],
    fake_results: List[Tuple[torch.Tensor, List[torch.Tensor]]],
) -> torch.Tensor:
    """L1 feature matching loss across all sub-discriminator intermediate layers."""
    loss = torch.tensor(0.0, device=real_results[0][0].device)
    n_maps = 0
    for (_, real_fmaps), (_, fake_fmaps) in zip(real_results, fake_results):
        for rf, ff in zip(real_fmaps, fake_fmaps):
            loss = loss + F.l1_loss(ff, rf.detach())
            n_maps += 1
    return loss / max(n_maps, 1)


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def find_latest_gan_checkpoint(out_dir: Path) -> Tuple[str | None, int]:
    """Find latest GAN checkpoint (must contain both 'gen' and 'disc' keys)."""
    latest_fixed = out_dir / "ckpt_step_latest.pt"
    if latest_fixed.exists():
        try:
            ckpt = torch.load(str(latest_fixed), map_location="cpu", weights_only=False)
            if "gen" in ckpt and "disc" in ckpt:
                step = int(ckpt.get("step", 0))
                return str(latest_fixed), step
            else:
                print(f"Skipping {latest_fixed}: missing 'gen'/'disc' keys (incompatible)")
        except Exception as e:
            print(f"Skipping {latest_fixed}: failed to load ({e})")

    ckpts = glob.glob(str(out_dir / "ckpt_step_*.pt"))
    ckpts = [p for p in ckpts if not p.endswith("ckpt_step_latest.pt")]
    if not ckpts:
        return None, 0

    def step_num(p: str) -> int:
        m = re.search(r"ckpt_step_(\d+)\.pt", p)
        return int(m.group(1)) if m else 0

    ckpts.sort(key=step_num, reverse=True)
    for ckpt_path in ckpts:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "gen" not in ckpt or "disc" not in ckpt:
                print(f"Skipping {ckpt_path}: missing 'gen'/'disc' keys (incompatible)")
                continue
            return ckpt_path, step_num(ckpt_path)
        except Exception as e:
            print(f"Skipping {ckpt_path}: failed to load ({e})")
            continue
    return None, 0


def load_stft_checkpoint_gen(stft_ckpt_path: str, gen: TinyVocoder, device: torch.device) -> None:
    """Load generator weights from a Phase 1 STFT checkpoint."""
    ckpt = torch.load(stft_ckpt_path, map_location=device, weights_only=False)
    if "gen" not in ckpt:
        raise ValueError(f"STFT checkpoint {stft_ckpt_path} has no 'gen' key")
    state = gen.module if hasattr(gen, "module") else gen
    state.load_state_dict(ckpt["gen"])
    print(f"Loaded generator from STFT checkpoint: {stft_ckpt_path} (step {ckpt.get('step', '?')})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GAN fine-tuning of TinyVocoder (Phase 2)")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.expanduser("~/Datasets/wind_clean/clips"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/train_vocoder_gan",
    )
    parser.add_argument(
        "--mel_stats",
        type=str,
        default="outputs/mel_stats.json",
    )
    parser.add_argument(
        "--stft_ckpt",
        type=str,
        default="outputs/train_vocoder_stft/ckpt_step_latest.pt",
        help="Phase 1 STFT checkpoint to initialize generator from",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from latest GAN checkpoint")
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--drive_dir",
        type=str,
        default=None,
        help="Google Drive directory for checkpoint persistence (Colab)",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Restore from Google Drive before resume logic reads out_dir
    if args.drive_dir is not None:
        drive_ckpt = Path(args.drive_dir) / "latest_checkpoint.pt"
        if drive_ckpt.exists():
            dest = out_dir / "ckpt_step_latest.pt"
            try:
                shutil.copy2(str(drive_ckpt), str(dest))
                print(f"Restored checkpoint from Drive: {drive_ckpt} -> {dest}")
            except Exception as e:
                print(f"WARNING: Could not restore from Drive: {e}")

    # Dataset & DataLoader
    dataset = VocoderDataset(clips_dir=args.data_dir, mel_stats_path=args.mel_stats)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    # Models
    gen  = TinyVocoder(n_mels=128, base_ch=256).to(device)
    disc = CombinedDisc(periods=[2, 3, 5, 7, 11], scales=[1, 2, 4]).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        gen  = torch.nn.DataParallel(gen)
        disc = torch.nn.DataParallel(disc)

    # STFT loss
    loss_fn = MultiResolutionSTFTLoss(configs=[
        (512,  50,  240),
        (1024, 120, 600),
        (2048, 240, 1200),
    ])

    # Optimizers + schedulers
    opt_G = torch.optim.AdamW(gen.parameters(),  lr=1e-4, betas=(0.8, 0.99))
    opt_D = torch.optim.AdamW(disc.parameters(), lr=1e-4, betas=(0.8, 0.99))
    scheduler_G = torch.optim.lr_scheduler.ExponentialLR(opt_G, gamma=0.99999)
    scheduler_D = torch.optim.lr_scheduler.ExponentialLR(opt_D, gamma=0.99999)

    start_step = 0

    if args.resume:
        ckpt_path, start_step = find_latest_gan_checkpoint(out_dir)
        if ckpt_path is None:
            print("No compatible GAN checkpoint found. Will initialize generator from STFT checkpoint.")
        else:
            print(f"Resuming from GAN checkpoint {ckpt_path} (step {start_step})")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            _gen_state  = gen.module  if hasattr(gen,  "module") else gen
            _disc_state = disc.module if hasattr(disc, "module") else disc
            _gen_state.load_state_dict(ckpt["gen"])
            _disc_state.load_state_dict(ckpt["disc"])
            opt_G.load_state_dict(ckpt["opt_G"])
            opt_D.load_state_dict(ckpt["opt_D"])
            if "scheduler_G" in ckpt:
                scheduler_G.load_state_dict(ckpt["scheduler_G"])
            if "scheduler_D" in ckpt:
                scheduler_D.load_state_dict(ckpt["scheduler_D"])

    # Load generator from STFT Phase 1 checkpoint if starting fresh
    if start_step == 0:
        stft_ckpt = Path(args.stft_ckpt)
        if stft_ckpt.exists():
            load_stft_checkpoint_gen(str(stft_ckpt), gen, device)
        else:
            print(f"WARNING: STFT checkpoint not found at {stft_ckpt}. Starting from random weights.")

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

    print(f"\nGAN_WARMUP: {GAN_WARMUP} steps (STFT-only before adversarial loss)")
    print("All assertions passed. Starting training...\n")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    global_step = start_step
    loader_iter = iter(loader)

    stft_history: collections.deque = collections.deque(maxlen=500)
    adv_history:  collections.deque = collections.deque(maxlen=500)
    fm_history:   collections.deque = collections.deque(maxlen=500)
    d_history:    collections.deque = collections.deque(maxlen=500)

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

        # -------------------------------------------------------------------
        # 1. Discriminator update (every step)
        # -------------------------------------------------------------------
        with torch.no_grad():
            y_hat_d = gen(mel)

        real_results_d = disc(audio)
        fake_results_d = disc(y_hat_d)

        d_loss = discriminator_loss(real_results_d, fake_results_d)

        opt_D.zero_grad()
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(disc.parameters(), 5.0)
        opt_D.step()
        scheduler_D.step()

        d_loss_val = d_loss.item()

        # -------------------------------------------------------------------
        # 2. Generator update
        # -------------------------------------------------------------------
        y_hat = gen(mel)
        stft_loss = loss_fn(y_hat, audio)

        if global_step >= GAN_WARMUP:
            fake_results_g = disc(y_hat)
            with torch.no_grad():
                real_results_g = disc(audio)

            adv_loss = generator_adv_loss(fake_results_g)
            fm_loss  = feature_matching_loss(real_results_g, fake_results_g)
            g_loss   = 2.0 * stft_loss + 1.0 * adv_loss + 4.0 * fm_loss
        else:
            adv_loss = torch.tensor(0.0, device=device)
            fm_loss  = torch.tensor(0.0, device=device)
            g_loss   = stft_loss

        opt_G.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 5.0)
        opt_G.step()
        scheduler_G.step()

        global_step += 1

        stft_history.append(stft_loss.item())
        adv_history.append(adv_loss.item())
        fm_history.append(fm_loss.item())
        d_history.append(d_loss_val)

        # -------------------------------------------------------------------
        # Logging every 50 steps
        # -------------------------------------------------------------------
        if global_step % 50 == 0:
            avg_stft = sum(stft_history) / len(stft_history)
            avg_adv  = sum(adv_history)  / len(adv_history)
            avg_fm   = sum(fm_history)   / len(fm_history)
            avg_d    = sum(d_history)    / len(d_history)
            lr_g = scheduler_G.get_last_lr()[0]
            flag = " WARNING" if (g_loss.item() != g_loss.item() or g_loss.item() > 1e4) else ""
            warmup_tag = " [warmup]" if global_step < GAN_WARMUP else ""
            print(
                f"Step {global_step:06d}/{args.max_steps}{warmup_tag} | "
                f"stft={avg_stft:.4f}  adv={avg_adv:.4f}  "
                f"fm={avg_fm:.4f}  d={avg_d:.4f}  "
                f"lr_g={lr_g:.2e}{flag}"
            )

        # -------------------------------------------------------------------
        # Preview audio every 500 steps
        # -------------------------------------------------------------------
        if global_step % 500 == 0:
            gen.eval()
            with torch.no_grad():
                preview_wav = gen(mel[0:1])  # (1, T)
            preview_path = out_dir / f"preview_step_{global_step:06d}.wav"
            wav_np = preview_wav.squeeze(0).detach().cpu().float().numpy()
            sf.write(str(preview_path), wav_np, samplerate=22050)
            print(f"Saved preview: {preview_path}")

        # -------------------------------------------------------------------
        # Checkpoint every 500 steps
        # -------------------------------------------------------------------
        if global_step % 500 == 0:
            ckpt_path = out_dir / f"ckpt_step_{global_step:06d}.pt"
            _gen_state  = gen.module  if hasattr(gen,  "module") else gen
            _disc_state = disc.module if hasattr(disc, "module") else disc
            torch.save({
                "gen":           _gen_state.state_dict(),
                "disc":          _disc_state.state_dict(),
                "opt_G":         opt_G.state_dict(),
                "opt_D":         opt_D.state_dict(),
                "scheduler_G":   scheduler_G.state_dict(),
                "scheduler_D":   scheduler_D.state_dict(),
                "step":          global_step,
                "mel_cfg":       dataset.mel_cfg.__dict__,
                "target_frames": TARGET_FRAMES,
            }, str(ckpt_path))
            print(f"Saved checkpoint: {ckpt_path}")

            # Kaggle backup
            kaggle_out = Path("/kaggle/working/latest_checkpoint.pt")
            try:
                shutil.copy2(str(ckpt_path), str(kaggle_out))
                print(f"Copied latest checkpoint to {kaggle_out}")
            except Exception as e:
                print(f"WARNING: Kaggle backup failed at step {global_step}: {e}")

            # Google Drive backup
            if args.drive_dir is not None:
                drive_out = Path(args.drive_dir) / "latest_checkpoint.pt"
                try:
                    Path(args.drive_dir).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(ckpt_path), str(drive_out))
                    print(f"Backed up to Drive: {drive_out}")
                except Exception as e:
                    print(f"WARNING: Drive backup failed at step {global_step}: {e}")

    # Final checkpoint (generator weights only for inference)
    final_path = out_dir / "final_model.pt"
    _gen_state = gen.module if hasattr(gen, "module") else gen
    torch.save({
        "gen":           _gen_state.state_dict(),
        "step":          global_step,
        "mel_cfg":       dataset.mel_cfg.__dict__,
        "target_frames": TARGET_FRAMES,
    }, str(final_path))
    print(f"\nTraining complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
