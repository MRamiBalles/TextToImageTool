import torch
import torch.nn as nn
import math

class RotaryPositionalEmbeddings(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) for 2D images (Flux/DiT style).
    Applies rotation to pairs of dimensions to encode relative positions.
    """
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, D] or [B, H, W, D] input tensor
            h, w: Spatial dimensions
        Returns:
            cos, sin: Positional embeddings to apply
        """
        # Create grid
        # For Flux/DiT, we usually flatten H, W to N.
        # This implementation creates the 2D freqs.
        
        dim = self.dim // 2 # split into two 1D ROPEs (height and width)
        
        # 1. Height frequencies
        inv_freq = 1.0 / (self.max_period ** (torch.arange(0, dim, 2).float() / dim)).to(x.device)
        
        # Grid creation
        h_seq = torch.arange(h, device=x.device).float()
        w_seq = torch.arange(w, device=x.device).float()
        
        # Outer product to get grid args
        freqs_h = torch.einsum("i,j->ij", h_seq, inv_freq) # [H, D/4]
        freqs_w = torch.einsum("i,j->ij", w_seq, inv_freq) # [W, D/4]
        
        # Concat to get full embeddings for each position (y, x)
        # We need to broadcast them to [H, W, D/2]
        # This part depends heavily on the specific "packed" format of Flux.
        # For standard 2D, we concat freqs_h and freqs_w.
        
        # Simplified standard RoPE logic for now:
        # Just return the freqs assuming the caller handles the broadcast/apply
        # Or returns pre-computed cos/sin tables used for 'apply_rotary_emb'
        
        return freqs_h, freqs_w # Placeholder for specific Flux logic adjustment

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies RoPE rotation to x using cos/sin frequencies.
    Rotates half-dimensions.
    """
    # Standard RoPE rotation:
    # x = [x1, x2], new_x1 = x1cos - x2sin, new_x2 = x1sin + x2cos
    # Handle shapes [B, Seq, Head, Dim] or similar
    
    # Placeholder specific implementation awaiting DiT context
    return x * cos + rotate_half(x) * sin

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


class RectifiedFlowScheduler:
    """
    Rectified Flow Scheduler (Flux.1 style).
    Interpolates linearly between noise (t=1) and data (t=0).
    v = x_1 - x_0
    """
    def __init__(self, num_inference_steps: int = 20, shift: float = 1.0):
        self.num_inference_steps = num_inference_steps
        self.shift = shift 
        # Flux uses a "shift" parameter to bias sampling towards noise or data.
        
    def get_sigmas(self, steps: int):
        """
        Generates sigmas (timesteps) for the schedule.
        Flux typically goes from 1.0 down to 0.0.
        """
        # Linear schedule
        # Simple implementation:
        sigmas = torch.linspace(1.0, 0.0, steps + 1)
        
        # Apply shift if needed (Flux Dev uses specific shifting logic for beta limit)
        # sigma_shifted = shift * sigma / (1 + (shift - 1) * sigma)
        
        if self.shift != 1.0:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
            
        return sigmas

    def step(self, model_output: torch.Tensor, timestep: float, sample: torch.Tensor, next_timestep: float) -> torch.Tensor:
        """
        Euler Step for Rectified Flow.
        sample: x_t
        model_output: v (velocity prediction)
        
        x_{t-1} = x_t + (t_{next} - t_{prev}) * v
        """
        dt = next_timestep - timestep
        prev_sample = sample + dt * model_output
        return prev_sample

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Forward process for training/testing.
        x_t = (1 - t) * x_0 + t * x_1
        where x_1 is noise.
        """
        # Broadcast timesteps
        t = timesteps.view(-1, *([1]*(original_samples.ndim - 1)))
        
        # Rectified Flow: Linear Interpolation
        # Note: Flux notation might be t=0 is data, t=1 is noise.
        # x_t = (1 - t) * x_0 + t * noise
        noisy_samples = (1 - t) * original_samples + t * noise
        return noisy_samples
