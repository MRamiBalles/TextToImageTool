import torch
from typing import Optional, Tuple

class EulerFlowSampler:
    """
    Euler Solver for Rectified Flow ODEs (Flux.1) with Optimization.
    x_{next} = x_{curr} + (t_{next} - t_{curr}) * v_pred
    """
    def __init__(self, num_inference_steps: int = 20):
        self.num_inference_steps = num_inference_steps
        
    def step(
        self, 
        model_output_cond: torch.Tensor,
        model_output_uncond: torch.Tensor,
        timestep: float,
        sample: torch.Tensor,
        next_timestep: float,
        step_index: int,
        guidance_scale: float = 3.5,
        target_device: torch.device = None
    ) -> torch.Tensor:
        """
        Performs one Euler step with Guidance Interval Optimization.
        If current step is outside guidance interval (10%-80%), we skip CFG formula
        assuming the caller provided appropriate model outputs (or we prioritize cond).
        """
        # Guidance Interval check
        apply_guidance = SamplerUtils.should_apply_cfg(step_index, self.num_inference_steps)
        
        if apply_guidance and guidance_scale > 1.0:
            # Apply standard CFG
            v_pred = SamplerUtils.apply_cfg(model_output_cond, model_output_uncond, guidance_scale)
            # Apply Dynamic Thresholding (Mimetic)
            v_pred = SamplerUtils.apply_dynamic_thresholding(v_pred)
        else:
            # Outside interval or scale=1: Use conditional output directly (Standard Flux behavior)
            # Logic: At very high noise (beginning), unconditional is random, conditional is strong structure.
            # At very low noise (end), structure is settled, texture shouldn't diverge.
            v_pred = model_output_cond
            
        dt = next_timestep - timestep
        prev_sample = sample + dt * v_pred
        return prev_sample

class SamplerUtils:
    @staticmethod
    def apply_cfg(
        noise_pred_cond: torch.Tensor, 
        noise_pred_uncond: torch.Tensor, 
        guidance_scale: float
    ) -> torch.Tensor:
        """Standard Classifier-Free Guidance."""
        return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

    @staticmethod
    def apply_dynamic_thresholding(
        x: torch.Tensor, 
        quantile: float = 0.995, 
        max_val: float = 1.0
    ) -> torch.Tensor:
        """
        Dynamic Thresholding (Imagen/SOTA style).
        Prevents saturation when using high CFG scales.
        """
        dtype = x.dtype
        x = x.float() # Compute in float32
        
        # Calculate quantile
        # Flatten batch/spatial for quantile calc
        s = torch.quantile(x.abs().reshape(x.shape[0], -1), quantile, dim=1)
        s = torch.maximum(s, torch.ones_like(s) * max_val)
        s = s.view(-1, *([1]*(x.ndim-1))) # Broadcast
        
        # Clip
        x = torch.clamp(x, -s, s)
        
        # Renormalize to [-max_val, max_val]
        x = x / s * max_val
        
        return x.to(dtype)

    @staticmethod
    def should_apply_cfg(step_index: int, total_steps: int, start_ratio: float = 0.1, end_ratio: float = 0.8) -> bool:
        """
        Guidance Interval Logic.
        Returns True if we should apply CFG at this step.
        """
        ratio = step_index / total_steps
        return start_ratio <= ratio <= end_ratio
