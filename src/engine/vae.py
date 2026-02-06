import torch
from diffusers import AutoencoderKL
from typing import Optional, Union, Tuple

class VAEWrapper:
    """
    Wrapper for the Variational Autoencoder (VAE) to handle image compression/decompression.
    Optimized for local execution with support for tiled processing and offloading.
    """
    def __init__(
        self, 
        model_id: str = "stabilityai/sdxl-vae", # Default to SDXL VAE (compatible with most modern usage)
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16,
        use_tiling: bool = True
    ):
        self.device = device
        self.dtype = dtype
        
        print(f"[VAE] Loading AutoencoderKL from {model_id}...")
        self.vae = AutoencoderKL.from_pretrained(model_id).to(self.device, dtype=self.dtype)
        
        if use_tiling:
            self.enable_tiling()
            
    def enable_tiling(self):
        """Enables tiled decoding to save VRAM on high-res images."""
        self.vae.enable_tiling()
        print("[VAE] Tiling enabled for VRAM optimization.")

    def disable_tiling(self):
        self.vae.disable_tiling()

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encodes an image to latent space.
        Args:
            image: [B, C, H, W] tensor in range [-1, 1]
        Returns:
            latents: [B, 4, H/8, W/8] (for SDXL VAE)
        """
        # Ensure input is correct dtype/device
        image = image.to(device=self.device, dtype=self.dtype)
        
        # Encode to distribution
        with torch.no_grad():
            dist = self.vae.encode(image).latent_dist
            latents = dist.sample()
            
            # Scale latents (magic number for SDXL/SD1.5 VAEs)
            # Note: SDXL scale factor is 0.13025, SD1.5 is 0.18215. 
            # We use the config value if available, or default to SDXL.
            scaling_factor = self.vae.config.scaling_factor if hasattr(self.vae.config, "scaling_factor") else 0.13025
            latents = latents * scaling_factor
            
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decodes latents back to image space.
        Args:
            latents: [B, 4, H/8, W/8]
        Returns:
            image: [B, C, H, W] in range [-1, 1]
        """
        latents = latents.to(device=self.device, dtype=self.dtype)
        
        with torch.no_grad():
            # Unscale latents
            scaling_factor = self.vae.config.scaling_factor if hasattr(self.vae.config, "scaling_factor") else 0.13025
            latents = latents / scaling_factor
            
            image = self.vae.decode(latents).sample
            
        return image

    def get_latent_shape(self, height: int, width: int) -> Tuple[int, int]:
        """Returns the height and width of the latent representation."""
        # SD VAE downsamples by factor of 8
        return height // 8, width // 8
