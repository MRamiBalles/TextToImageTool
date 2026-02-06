import torch
import torch.nn as nn
from diffusers import FluxTransformer2DModel
from typing import Optional

class FluxDiT(nn.Module):
    """
    Wrapper around diffusers.FluxTransformer2DModel to integrate with our Custom Engine.
    Uses Official Weights (Flux.1 Schnell/Dev) but allows custom Sampling & Memory Management.
    """
    def __init__(self, model_id: str = "black-forest-labs/FLUX.1-schnell", dtype=torch.bfloat16, cache_dir=None, dummy=False):
        super().__init__()
        self.dtype = dtype
        self.model_id = model_id
        
        if dummy:
             print("[FluxDiT] Smoke Test Mode: Initializing with Random Weights (No Download)...")
             from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
             # Create a small/default config for testing
             # We use a known small config or default
             config = FluxTransformer2DModel.load_config(model_id, cache_dir=cache_dir) if not dummy else None
             if config is None:
                 # Fallback to manual config creation if load_config also hits network hard
                 # Minimal Flux config
                 self.model = FluxTransformer2DModel(
                    patch_size=1,
                    in_channels=4,
                    num_layers=2, # Tiny for smoke test
                    num_single_layers=2,
                    attention_head_dim=64,
                    num_attention_heads=4,
                    joint_attention_dim=4096,
                    pooled_projection_dim=768,
                    guidance_embeds=True, # Schnell uses guidance=0 but model uses embeds
                 )
        else:
            print(f"[FluxDiT] Loading SOTA Transformer: {model_id}...")
            # Load the official model structure
            try:
                self.model = FluxTransformer2DModel.from_pretrained(
                    model_id, 
                    subfolder="transformer",
                    torch_dtype=dtype,
                    cache_dir=cache_dir
                )
            except Exception as e:
                print(f"[FluxDiT] Error loading from specific subfolder: {e}")
                print("[FluxDiT] Attempting direct load...")
                self.model = FluxTransformer2DModel.from_pretrained(
                    model_id, 
                    torch_dtype=dtype,
                    cache_dir=cache_dir
                )
            
        self.config = self.model.config

    def forward(self, img: torch.Tensor, txt: torch.Tensor, timesteps: torch.Tensor, y: torch.Tensor):
        """
        Maps Engine Inputs -> Diffusers Inputs.
        
        Args:
            img: Latents [B, C, H, W] (Unpacked) OR [B, L, D] (Packed)
            txt: Encoder Hidden States [B, L_txt, D]
            timesteps: [B]
            y: Pooled Projections [B, D_y]
        """
        # Flux Expects Packed Latents? 
        # Diffusers FluxTransformer2DModel expects `hidden_states` as [B, L_img, D] usually.
        # But if input is 4D [B, C, H, W], we might need to pack/flatten.
        # However, checking Diffusers documentation, FluxTransformer2DModel takes packed 3D inputs.
        
        # 1. Check Input Shape
        if img.ndim == 4:
            B, C, H, W = img.shape
            # Basic flattening/packing if needed. 
            # For Flux, usually standard packing is (H/2 * W/2) length.
            # But let's assume the VAE/Noise manager passes what is needed.
            # If standard VAE output [B, 4, H/8, W/8], we need to patchify.
            
            # Use diffusers internal helper if accessible, or simple flatten
            img = img.view(B, C, -1).transpose(1, 2) # [B, L, C]
            
        # 2. Prepare IDs (RoPE)
        # Flux needs `img_ids` and `txt_ids` for 3D positional encoding.
        # We generate them based on sequence length if not provided.
        # (Simplified generation for SOTA wrapper - ideal would be to pass H,W)
        
        # NOTE: For this wrapper, we rely on the model to handle defaults if possible, 
        # or we generate simple linear IDs.
        
        # 3. Forward
        # Output is Tuple (sample,)
        output = self.model(
            hidden_states=img,
            encoder_hidden_states=txt,
            pooled_projections=y,
            timestep=timesteps,
            return_dict=False
        )[0]
        
        return output
