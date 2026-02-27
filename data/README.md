# Data Specifications (v1)

To ensure model stability and GPU efficiency, all raw audio is pre-processed into a standardized format before training.

### Audio Standards
* **Format:** WAV (16-bit PCM)
* **Sample Rate:** 22,050 Hz (Resampled during ingestion)
* **Channels:** Mono
* **Duration:** 5.12 seconds  
    * *Note: 5.12s is used to ensure the resulting spectrogram frames align with "Power of 2" dimensions (e.g., 512 frames), optimizing GPU tensor operations.*
* **Loudness Normalization:** Integrated Peak Normalization at -1.0 dB to prevent the model from confounding low-amplitude wind with stochastic silence.

### Feature Representation
* **Type:** Log-Mel Spectrogram
* **Frequency Resolution:** 128 Mel bins (Vertical axis)
* **Temporal Resolution:** Optimized to create near-square matrices (e.g., 128 x 512) for compatibility with standard Diffusion U-Net architectures.

### Pre-processing Pipeline
All raw data passes through `scripts/prepare_dataset.py` which performs:
1.  **Discovery:** Recursive search for `.wav`, `.mp3`, and `.flac`.
2.  **Cleaning:** Silence trimming (Leading/Trailing) and RMS normalization.
3.  **Segmentation:** Sliding window approach with 50% overlap to maximize training samples from long-form recordings.

### Mel Config
sr=22050
n_fft=1024
hop_length=256
win_length=1024
n_mels=128
and diffusion uses fixed mel shape (1, 128, 440) → 440 frames