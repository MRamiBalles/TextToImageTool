# Task Checklist: Generative Image Engine (Local & SOTA)

- [ ] **Phase 1: Infrastructure & Base Components** <!-- id: 0 -->
    - [ ] Project Structure & Environment Setup (PyTorch, Dependencies) <!-- id: 1 -->
    - [ ] **VAE Module (Autoencoder)** <!-- id: 2 -->
        - [ ] Implement Encoder & Decoder classes <!-- id: 3 -->
        - [ ] Unit Test: Reconstruction consistency `Dec(Enc(Img))` <!-- id: 4 -->
        - [ ] Integration: Load pre-trained VAE weights (e.g., from SDXL/SD3) <!-- id: 5 -->
    - [ ] **Noise Management** <!-- id: 6 -->
        - [ ] Implement Sinusoidal/Rotary Positional Embeddings (Time Embeddings) <!-- id: 7 -->
        - [ ] Implement Noise Schedulers (Linear, Cosine) <!-- id: 8 -->
        - [ ] Visualize Forward Diffusion process <!-- id: 9 -->

- [ ] **Phase 2: The Diffusion Engine (DiT Core)** <!-- id: 10 -->
    - [ ] **DiT Architecture Implementation** <!-- id: 11 -->
        - [ ] Patchify/Embeddings mechanism <!-- id: 12 -->
        - [ ] Transformer Blocks (Attention + MLP) with Flash Attention <!-- id: 13 -->
        - [ ] Modulation (AdaLN / AdaLN-Zero) for Time/Label injection <!-- id: 14 -->
    - [ ] **Text Encoder Integration (The Semantic Brain)** <!-- id: 15 -->
        - [ ] Implement Dual Encoder capability (CLIP ViT-L primary) <!-- id: 16 -->
        - [ ] Implement optional T5 offloading mechanism <!-- id: 17 -->
        - [ ] Cross-Attention mechanism implementation <!-- id: 18 -->
    - [ ] **Verification Loop** <!-- id: 19 -->
        - [ ] Dummy Training Loop (Single Batch Overfitting) <!-- id: 20 -->

- [ ] **Phase 3: Samplers & Optimization** <!-- id: 21 -->
    - [ ] **Samplers** <!-- id: 22 -->
        - [ ] Implement Euler Ancestral <!-- id: 23 -->
        - [ ] Implement DPM++ <!-- id: 24 -->
    - [ ] **Guiding & Thresholding** <!-- id: 25 -->
        - [ ] Classifier-Free Guidance (CFG) implementation <!-- id: 26 -->
        - [ ] Dynamic Thresholding (Mimetic/CFG fix) <!-- id: 27 -->
    - [ ] **Efficiency** <!-- id: 28 -->
        - [ ] Implement Model Offloading (CPU <-> GPU) <!-- id: 29 -->
        - [ ] Basic Quantization Support (FP16/BF16 loading) <!-- id: 30 -->

- [ ] **Phase 4: Advanced Features** <!-- id: 31 -->
    - [ ] Image-to-Image Pipeline <!-- id: 32 -->
    - [ ] Inpainting Support (Mask handling) <!-- id: 33 -->
    - [ ] Safety Checker Integration <!-- id: 34 -->

- [ ] **Documentation & Final Polish** <!-- id: 35 -->
    - [ ] Architecture Manifest (Why DiT? Why these specific optimizations?) <!-- id: 36 -->
    - [ ] Tensor Shape Documentation <!-- id: 37 -->
    - [ ] Full Integration Tests <!-- id: 38 -->
