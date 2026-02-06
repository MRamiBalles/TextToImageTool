# Implementation Plan - Local SOTA Image Generation Engine

## Goal Description
Build a state-of-the-art Image Generation Engine from scratch (Python/PyTorch) optimized for local execution on consumer hardware (e.g., RTX 3060+). The system will use a Modular Architecture based on Diffusion Transformers (DiT) utilizing pre-trained weights (likely Flux or SD3 compatible) but with a custom inference implementation to ensure maximum control and efficiency.

## User Review Required
> [!IMPORTANT]
> **Foundation Model Weights**: We need to decide which specific set of pre-trained weights to target for the "from scratch" implementation architecture (e.g., Flux.1-schnell/dev, SD3 Medium, or a custom mix). The plan currently assumes a generic DiT structure adaptable to these.

> [!WARNING]
> **VRAM Constraints**: The target is 8GB-12GB VRAM. Aggressive offloading (CPU <-> GPU) and Quantization (FP8/INT8) are critical path features, not optional.

## Proposed Changes

### Infrastructure Layer (Phase 1)
#### [NEW] `engine/vae.py`
- Implementation of the AutoencoderKL wrapper.
- Compatible with StabilityAI's VAE weights.
- Features: Tiled decoding to save VRAM.

#### [NEW] `engine/noise.py`
- Implementation of Flow Matching (preferred for DiT/Flux) or DDPM/DDIM schedulers.
- Positional Embeddings (Sinusoidal/RoPE).

### Core Diffusion Layer (Phase 2)
#### [NEW] `engine/dit.py`
- Main Diffusion Transformer Class.
- Support for `patch_size` configuration.
- Implementation of DiT Blocks with Flash Attention 2.0.

#### [NEW] `engine/text_encoder.py`
- CLIP ViT-L wrapper.
- T5-XXL wrapper with immediate CPU offload.

### Optimization Layer (Phase 3)
#### [NEW] `engine/sampling.py`
- Euler Ancestral and DPM++ solvers.
- Implementation of CFG and Dynamic Thresholding.

## Verification Plan

### Automated Tests
- **VAE Reconstruction**: `test_vae_reconstruction.py` - Assert MSE < 0.05 on standard test image.
- **Overfitting Test**: `test_one_batch_convergence.py` - Verify loss goes to near zero on a single image + prompt pair to prove architectural learning capability.

### Manual Verification
- **Generation Quality**: Generate 'A red cube on a blue cube' to verify T5 spatial understanding.
- **Memory Profiling**: Ensure peak VRAM usage stays < 8GB (or 12GB depending on config) during generation.
