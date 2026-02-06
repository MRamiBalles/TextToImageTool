import torch
import sys
import os
import argparse

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.memory_manager import MemoryManager, apply_dynamic_quant
from src.engine.text_encoder import DualTextEncoder
from src.engine.dit import FluxDiT
from src.engine.vae import VAEWrapper
from src.engine.noise import RectifiedFlowScheduler, RotaryPositionalEmbeddings
from src.engine.sampling import EulerFlowSampler

def run_inference(prompt: str, height: int = 1024, width: int = 1024, steps: int = 4, guidance: float = 0.0, seed: int = 42, use_distill: bool = False, dummy: bool = False):
    print("=== Starting Flux.1 Inference ===")
    print(f"Prompt: {prompt}")
    print(f"Config: {height}x{width}, Steps={steps}, DistillT5={use_distill}, DummyMode={dummy}")
    
    # 0. Setup Cache on D: to avoid System Drive
    cache_dir = "d:/TextToImageTool/cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    print(f"HF_HOME set to: {cache_dir}")
    
    torch.manual_seed(seed)
    
    # 1. Initialize Memory Manager
    mm = MemoryManager(device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {mm.device}, Storage Dtype: {mm.storage_dtype}")

    # 2. Text Encoding
    print("\n[Stage 1] Text Encoding (Sequential Offload)...")
    encoder = DualTextEncoder(mm, use_distill_t5=use_distill, cache_dir=cache_dir, dummy=dummy)
    pooled_prompt, prompt_embeds = encoder.encode(prompt)
    
    print(f"Pooled Shape: {pooled_prompt.shape}")      # [1, 768]
    print(f"Sequence Shape: {prompt_embeds.shape}")    # [1, 256, 4096] (or 512 for distil)

    # 3. Latent Generation (Noise)
    print("\n[Stage 2] Preparing Latents...")
    # For inference we start with random noise
    latent_H = height // 8
    latent_W = width // 8
    # Flux Latents: [B, 16, H/8/2, W/8/2] -> Packed to [B, L, 64]
    # For this script we assume Standard DiT shapes for the wrapper 
    # until we integrate the specific flux1-schnell.safetensors loader.
    # Placeholder: 4 channel noise
    latents = torch.randn(1, 4, latent_H, latent_W).to(mm.device, dtype=mm.compute_dtype)

    # 4. DiT Sampling Loop
    print("\n[Stage 3] Loading DiT & Sampling...")
    # NOTE: Here we would load the actual 'flux1-schnell.safetensors'
    # For now we instantiate the class.
    # In a real run, you would do: dit = FluxDiT.from_pretrained(..., cache_dir=cache_dir)
    dit = FluxDiT(cache_dir=cache_dir, dummy=dummy) 
    
    # V2.1 SOTA: Apply Group-wise Dynamic Quantization
    apply_dynamic_quant(dit.model, group_size=128)
    
    mm.load_model_to_gpu(dit, "FluxDiT")

    sampler = RectifiedFlowScheduler()
    sigmas = sampler.get_sigmas(steps=steps)
    solver = EulerFlowSampler(num_inference_steps=steps)
    
    print(f"Starting {steps} steps of Euler Flow...")
    for i in range(steps):
        t = sigmas[i]
        t_next = sigmas[i+1]
        print(f"Step {i+1}/{steps} (t={t:.2f} -> {t_next:.2f})")
        
        # Forward pass (Mocked here, real one needs weights)
        with torch.no_grad():
            # In real Flux, we pass packed latents
            # output = dit(latents, prompt_embeds, t, pooled_prompt)
            output_cond = torch.randn_like(latents) 
            output_uncond = torch.randn_like(latents) # Only needed if guidance > 1.0 (Schnell usually 0.0)
            
        latents = solver.step(
            model_output_cond=output_cond, 
            model_output_uncond=output_uncond,
            timestep=t, 
            sample=latents, 
            next_timestep=t_next,
            step_index=i,
            guidance_scale=guidance # Schnell uses 0.0 usually, Dev uses 3.5
        )
        
    mm.offload_model(dit, "FluxDiT")

    # 5. VAE Decoding
    print("\n[Stage 4] Decoding Image...")
    vae = VAEWrapper(use_tiling=True, device=mm.device) # Should load weights
    # For smoke test we skip actual decode if weights missing
    print("Inference Complete. (Image decoding skipped in smoke test without weights)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A futuristic city")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--distill", action="store_true", help="Use DistillT5 to save space")
    parser.add_argument("--dummy", action="store_true", help="Use Dummy/Random weights for smoke testing")
    args = parser.parse_args()
    
    run_inference(args.prompt, steps=args.steps, use_distill=args.distill, dummy=args.dummy)
