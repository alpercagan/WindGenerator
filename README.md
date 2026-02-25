# WindGenerator: Audio Synthesis via Spectrogram Diffusion

## Project Overview
This project explores the generation of synthetic wind audio using **Spectrogram Diffusion**. 

### Why Spectrogram Diffusion?
Wind is fundamentally "shaped noise"—it lacks the strict periodic structure of melodic instruments or human speech. Spectrograms are exceptionally well-suited for representing these types of stochastic textures. 

### Implementation Strategy
The model treats the spectrogram as a 2D image, leveraging standard diffusion libraries (e.g., `diffusers`) for generation. A pre-trained Vocoder, such as **HiFi-GAN**, is then utilized to reconstruct the final audio from the generated spectrogram.

### Why this over Latent Models?
While Latent Models are highly efficient, they require a two-stage training process (VAE compression followed by Diffusion training). Spectrogram Diffusion represents the "sweet spot" for high-fidelity sound generation in this context, effectively capturing the swirling, time-varying frequency energy shifts characteristic of wind.