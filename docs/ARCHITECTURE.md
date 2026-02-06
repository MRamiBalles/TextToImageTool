# Architecture Manifest: SOTA Local Image Engine (2025)

## 1. Engineering Decisions & Trade-offs

### 1.1 Why Flux.1 (Dev/Schnell) Architecture?
We chose **Flux.1** over Traditional Latent Diffusion (SDXL) or standard DiT (SD3) for the following reasons:
- **Rectified Flow Matching**: Flux uses ODE-based flow matching ($v = x_t - x_0$), allowing for linear interpolation. This enables high-fidelity generation in just 4-8 steps (Schnell) or 20 steps (Dev), compared to 30-50 for SDXL.
- **MM-DiT (Multimodal Diffusion Transformer)**: Unlike UNet which injects text via simple Cross-Attention, Flux uses `DoubleStreamBlock`s where image and text modalities are processed by separate transformer streams that exchange information via attention. This preserves the semantic fidelity of T5-XXL tokens much better than concatenation.
- **Rotary Positional Embeddings (RoPE)**: Provides superior generalization to different aspect ratios compared to fixed sinusoidal embeddings.

### 1.2 Memory Strategy: The "Sequential Offloading" Pipeline
Targeting **12GB VRAM** (RTX 3060) with a 12B parameter model requires strict memory discipline. We implemented a sequential pipeline:
1.  **Text Encoding**: T5-XXL (~11B) + CLIP are loaded, prompt is encoded, then **immediately offloaded** to CPU/RAM.
    - **DistillT5 Optimization**: Fallback support for T5-Base (~250MB) implemented for constrained environments (CVPR 2025 "Scaling Down Text Encoders").
2.  **Diffusion**: The 12B DiT is loaded. With **FP8 (e4m3fn)** quantization, weights take ~12GB. Activations are kept minimal using **Flash Attention**.
    - **Diffusers Integration**: We wrap `diffusers.FluxTransformer2DModel` to ensure weight compatibility and robust tensor math, while retaining our custom memory orchestration.
3.  **Decoding**: VAE is loaded last. **Tiled Decoding** is enforced to prevent OOM during the pixel upsampling phase ($128 \times 128 \to 1024 \times 1024$).

### 1.3 Precision Arithmetic
- **Weights**: stored in `torch.float8_e4m3fn` (where hardware supports) to maximize model size fitting in VRAM.
- **Compute**: cast to `torch.bfloat16` for matrix multiplications. FP16 is avoided for the Transformer backbone to prevent numerical instability (NaNs) in the attention layers.

## 2. Tensor Anatomy

### 2.1 Latent Space
- **VAE Output**: `[B, C, H/8, W/8]` (e.g. 16 channels for Flux).
- **Packed Latents (DiT Input)**: `[B, Sequence_Length, Hidden_Dim]`.
  - Flux does **not** process grids. It "patchifies" the image (flattening 2x2 blocks) into a 1D sequence.
  - Final Input: `[B, 4096, D]`.
  - **RoPE Handling**: Since the sequence is 1D, RoPE re-calculates the 2D grid positions `(x, y)` based on the sequence index during the forward pass.

### 2.2 Text Embeddings
- **Pooled (CLIP)**: `[B, 768]`. Provides global style/content context.
- **Sequence (T5)**: `[B, 256, 4096]`. Provides detailed semantic modification instructions.

## 3. Optimization Techniques (Implemented)
- **Dynamic Thresholding**: Clamps latents to `percentile(99.5)` during sampling to prevent color burn at high CFG scales.
- **Guidance Interval (V1)**: CFG is strictly limited to the middle 70-80% of inference steps.
- **Simulation Mode**: `--dummy` flag enables full pipeline verification without weight loading (CI/CD standard).

## 4. SOTA Roadmap (V2.1 - Planned)

### 4.1 Time-Varying Guidance (TV-CFG)
Current static guidance (Interval 10-80%) will be upgraded to **Dynamic Guidance Scheduling**.
- **Theory**: Generation has three stages: Direction Shift, Mode Separation, and Concentration.
- **Implementation**: A time-dependent guidance curve (Triangle/Cosine) that is low at $t=1$, peaks at $t=0.5$, and decays at $t=0$.

### 4.2 Dynamic Activation Quantization (Salient Channels)
Standard Int8/FP8 quantization degrades DiT quality due to "Salient Channels" with extreme variance.
- **Solution (PTQ4DiT)**: Implement **Group-wise** or **Token-wise** dynamic scaling for activations.
- **Action**: Add pre-processing hooks to smooth outliers before quantization ("SmoothQuant").

### 4.3 Cache Optimization
- **TeaCache / Skip-Cache**: Leverage temporal redundancy between steps. If embedding changes < $\epsilon$, reuse previous DiT block outputs to speed up inference by 1.5x.
