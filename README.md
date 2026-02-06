# TextToImageTool (Local SOTA Engine)

A high-performance, modular Image Generation Engine built from scratch (Python/PyTorch) targeting **Flux.1** architecture on consumer hardware (RTX 3060 12GB).

## Key Features
- **Architecture**: Diffusion Transformer (DiT) with Rectified Flow Matching.
- **Memory Efficient**: strict **Sequential Offloading** (T5 -> DiT -> VAE) and **FP8 Quantization** support.
- **Advanced Sampling**: Euler Flow Solver with **Dynamic Thresholding** and **Guidance Interval**.
- **Dual Text Encoder**: CLIP ViT-L + T5-XXL integration.

## Installation

1.  **Requirements**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Requires PyTorch 2.1+ with CUDA 12.1+ for FP8 support.*

2.  **Environment**:
    Ensure you have at least 12GB VRAM and 32GB System RAM (for offloading).

## Running Verification
To verify the pipeline logic and memory limits without downloading heavy weights (running in simulation mode):

```bash
run_verification.bat
```

Or manually:
```bash
python scripts/verify_pipeline.py
```

## Project Structure
- `src/engine/`: Core modules (VAE, DiT, Noise, Sampling, Memory).
- `docs/ARCHITECTURE.md`: Detailed engineering manifest and tensor anatomy.
- `tests/`: Unit tests for critical components.

## License
MIT / Apache 2.0 (Codebase). 
*Note: Model weights (Flux.1) are subject to Black Forest Labs licenses.*
