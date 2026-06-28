# WindGenerator

Can a diffusion model trained on mel spectrograms learn the structure of wind well enough to generate new wind sounds that actually sound like wind? That is the question this project tests.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alperarslan19/WindGenerator/blob/main/notebooks/wind_generator_demo.ipynb)


**[Generated audio samples →](https://github.com/alperarslan19/WindGenerator/releases/tag/v0.1-audio-samples)**

**[Model weights on Hugging Face →](https://huggingface.co/alpercagann/wind-generator)**


---



## Why wind?

Wind is an unusual signal to model. It has no pitch, no rhythm, no harmonics, and no meaning. Physically it is shaped noise: broadband turbulent energy filtered by whatever it moves through. A model cannot cheat by memorizing a melody or a phrase. To produce wind it has to capture how energy is spread across frequencies and how that spread changes over time.

That makes it a clean test of a single question: can a generative model pick up the statistics of a natural texture from data alone, with no hand-written signal-processing rules?

The approach is to treat the sound as an image. Each clip becomes a mel spectrogram, a 2D array of frequency against time, so the task turns into learning a distribution over images. A standard image model then applies without changes.


The mel spectrogram

A mel spectrogram is a 2D picture of sound. The horizontal axis is time, the vertical axis is frequency. The frequency axis uses the mel scale, which gives more resolution to low frequencies and less to high ones, closer to how we hear.

Each 5.12-second clip at 22,050 Hz becomes a (128, 440) array: 128 frequency bins, 440 time frames. To the model this is just a grayscale image, 128 tall and 440 wide.

How a clip becomes this image:


STFT. The waveform is cut into overlapping windows and each window is Fourier-transformed. Window length is 1024 samples (~46 ms); the window steps forward 256 samples (~12 ms) each time, so consecutive frames overlap by 75%.
Power, phase dropped. I keep the magnitude (squared) and discard the phase. This matters later: throwing away phase is exactly why reconstruction needs Griffin-Lim.
Mel projection. The linear frequency bins are collapsed onto 128 mel bins.
Log. Raw energy spans several orders of magnitude (a quiet bin near 0.00001, a loud one near 100). Taking the log compresses that range into something gradient descent can work with, and it also matches how loudness is perceived.
Normalize. See below.


Steps 1–3 are done by torchaudio's MelSpectrogram. Steps 4–5 are mine.

---


## Dataset
 
1,966 wind clips, each 5.12 seconds, 22,050 Hz, mono.
 
**[Wind Sounds Dataset on Kaggle →](https://www.kaggle.com/datasets/alperaanarslan/wind-sounds-dataset)**
 
The clips were cut from longer recordings and filtered to drop silence, non-wind content, and clips with very low RMS energy. `outputs/audit/` holds the quality check: RMS and peak histograms, reconstruction-error stats, and 30 sample clips spread across the quality range.
 
| Parameter | Value | Why |
|---|---|---|
| Sample rate | 22,050 Hz | Standard for audio ML; covers up to 11,025 Hz |
| FFT size | 1,024 | ~46 ms window; reasonable time/frequency trade-off |
| Hop length | 256 | ~12 ms between frames; 440 frames per clip |
| Mel bins | 128 | Enough resolution without blowing up dimensionality |
| Frequency range | 20 Hz – 11,025 Hz | Covers the audible wind range |
 
### Normalization
 
After taking the log, every spectrogram is normalized with statistics computed once over the dataset:
 
```
x_normalized = clip((x - mean) / std, -4, 4) / 4
```
 
This puts values roughly in `[-1, 1]`, which is the range diffusion models expect.
 
The important choice is that `mean` and `std` are **global** (one pair for the whole dataset), not per-clip. Per-clip normalization caused a problem early on: the diffusion model learned one normalization, the audio reconstruction assumed another, and the two did not line up, so the output was incoherent. With one shared mean/std, every spectrogram lives in the same numerical space, whether it comes from the dataset or from the model. The stats are computed from a sample of clips (median of their per-clip mean and std), saved to `outputs/mel_stats.json`, and read by every part of the pipeline.
 
---
 
## The diffusion model
 
### Why diffusion
 
A diffusion model generates by learning to reverse a noising process. During training, Gaussian noise is added to real data at a random level, and the model learns to predict that noise. At generation time it starts from pure noise and removes a little at a time until something that looks like real data is left.
 
This fits wind well. Wind has no single correct shape to memorize; it is a stochastic texture. A diffusion model does not try to produce one right answer, it learns the distribution of possibilities and draws from it. The training setup also gives free variety: the same clip is seen at a different noise level and with different noise every time it comes up, so even with under 2,000 clips the model does not simply memorize them.
 
A latent diffusion model (compress first, then diffuse in the smaller space, as in AudioLDM or Stable Audio) would be more efficient, but it needs two training stages. For a first experiment, running diffusion directly on the spectrogram is a more direct test of the idea.
 
### Architecture
 
A `UNet2DModel` from the Hugging Face Diffusers library, with the spectrogram as a single-channel image. The architecture comes from the library; I picked the configuration and trained the weights from scratch (no pretrained checkpoint).
 
| Parameter | Value |
|---|---|
| Input / output shape | `(1, 128, 440)` |
| Channel widths | `(32, 64, 128)` |
| Levels | 3 |
| Blocks per level | 1 |
| Total parameters | ~2.5M |
 
The down and up paths are convolution-only; the bottleneck keeps one self-attention layer. For a stationary texture like wind there is little long-range structure to coordinate, so I kept the model small and shallow on purpose.
 
The model is intentionally small. A bigger model would in principle fit the distribution better, but a single T4 GPU in a Colab session (12-hour limit) sets a practical ceiling. At ~2.5M parameters it trains at about 180 ms/step, so 74,000 steps take roughly 4 hours. I tried the larger `(64, 128, 256)` widths; they ran about 10× slower without a matching gain in quality for this dataset size, so I stayed with `(32, 64, 128)`.
 
### Training
 
At each step a random timestep `t` is drawn, the spectrogram is noised to that level with `add_noise` (which applies `sqrt(alpha_bar)*x0 + sqrt(1-alpha_bar)*noise`), and the model predicts the noise:
 
```
loss = MSE(model(x_t, t), noise)
```
 
- **Scheduler:** DDPMScheduler, 1,000 timesteps, linear noise schedule.
- **Optimizer:** AdamW, lr = 2e-4.
- **Mixed precision:** `autocast` plus `GradScaler`, about 2× faster than fp32. Gradients are clipped to a max norm of 1.0.
- **Length:** 74,000 steps. The target was 100k; the Colab session ended first.
Training ran across several interrupted Colab sessions. Each session wipes the working directory, so I saved numbered checkpoints to Google Drive on a fixed interval and resumed from the latest one at the start of each session.
 
### Generating
 
To generate, the model starts from a `(1, 128, 440)` noise tensor and runs the DDPM reverse process (50 steps by default, set with `--ddpm_steps`). The output is a normalized mel spectrogram in `[-1, 1]`, which is then denormalized with the same global stats and turned back into audio.
 
---
 
## Getting audio back: Griffin-Lim
 
The model produces a mel spectrogram, which is a magnitude-only representation. Turning it back into a waveform means recovering the phase that was thrown away when the spectrogram was computed.
 
**Griffin-Lim** does this. It starts from the target magnitude and a random phase guess, then iterates back and forth between valid waveforms and the target magnitude until the phase settles. The pipeline first maps the 128 mel bins back to 513 STFT bins (`InverseMelScale`), then runs `GriffinLim` for 64 iterations.
 
This is the weakest part of the pipeline. Griffin-Lim settles on a phase that minimizes error on average, which leaves a slightly metallic, "phasey" sound. That artifact comes from phase reconstruction, not from the spectrogram itself: the fact that recognizable wind comes out at all is the sign that the model learned the spectral structure. If the model's output were structurally wrong, Griffin-Lim would just produce noise.
 
### Next step
 
The cleaner fix is a neural vocoder: a network that maps mel spectrograms straight to waveforms and learns phase on its own. Off-the-shelf vocoders (HiFi-GAN, WaveGlow) are trained on speech and carry speech-specific assumptions, so a vocoder trained on this wind data should do better. That is the main direction for future work.
 
---
 
## Results
 
Samples are in the [v0.1 release](https://github.com/alperarslan19/WindGenerator/releases/tag/v0.1-audio-samples).
 
The outputs have the things that make wind sound like wind: broadband texture, energy concentrated at low frequencies, and slow variation over time. The metallic edge is from Griffin-Lim, not from what the model learned. Some clips hold together better than others, which is expected given that sampling is stochastic and training stopped at 74k rather than 100k steps.
 
---
 
## Repository structure
 
```
WindGenerator/
├── src/windgen/
│   ├── mels.py              — log-mel extraction, global normalization
│   ├── dataset.py           — dataset, on-the-fly mel computation
│   └── config.py            — mel and dataset config
├── scripts/
│   ├── prepare_dataset.py   — segmentation and silence filtering
│   ├── compute_mel_stats.py — global normalization statistics
│   ├── audit_dataset.py     — dataset quality analysis
│   ├── train_diffusion.py   — DDPM training
│   └── generate_audio.py    — end-to-end generation
├── notebooks/
│   └── wind_generator_demo.ipynb
└── outputs/
    ├── mel_stats.json        — global normalization statistics
    ├── audit/                — quality analysis and samples
    └── inspect/              — spectrogram visualizations
```
 
---
 
## Usage
 
**Generate wind audio (needs a trained checkpoint):**
```bash
python scripts/generate_audio.py \
    --diffusion_ckpt path/to/checkpoint.pt \
    --mel_stats outputs/mel_stats.json \
    --output_dir outputs/generated \
    --num_clips 5 \
    --ddpm_steps 50
```
 
**Train from scratch:**
```bash
# 1. Prepare the dataset
python scripts/prepare_dataset.py --input_dir /path/to/raw_audio
 
# 2. Compute global mel statistics
python scripts/compute_mel_stats.py
 
# 3. Train
python scripts/train_diffusion.py \
    --data_dir /path/to/clips \
    --mel_stats outputs/mel_stats.json \
    --output_dir outputs/train_ddpm \
    --max_steps 100000
```
 
**Requirements:** Python 3.10+, PyTorch 2.0+, torchaudio, diffusers, soundfile
 
```bash
pip install -e .
```
 
---
 
## References
 
- Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (NeurIPS 2020)
- Kong et al., [HiFi-GAN](https://arxiv.org/abs/2010.05646) (NeurIPS 2020)
- Griffin & Lim, [Signal Estimation from Modified Short-Time Fourier Transform](https://ieeexplore.ieee.org/document/1164317) (IEEE TASSP 1984)
