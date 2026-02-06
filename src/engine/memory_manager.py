import torch
import gc
import os
from contextlib import contextmanager

class MemoryManager:
    """
    Manages VRAM usage by implementing Sequential Offloading and FP8/BF16 casting.
    Critical for running 12B+ parameter models (Flux.1) on consumer GPUs (12GB VRAM).
    """
    def __init__(self, device: str = "cuda", offload_device: str = "cpu"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.offload_device = offload_device
        self.check_fp8_support()

    def check_fp8_support(self):
        """Checks for System FP8 support and sets the storage dtype."""
        self.supports_fp8 = hasattr(torch, 'float8_e4m3fn')
        if self.supports_fp8:
            print("[MemoryManager] Native FP8 (e4m3fn) support detected.")
            self.storage_dtype = torch.float8_e4m3fn
            self.compute_dtype = torch.bfloat16
        else:
            print("[MemoryManager] FP8 not supported. Fallback to FP16/BF16.")
            self.storage_dtype = torch.float16
            self.compute_dtype = torch.float16

    def load_model_to_gpu(self, model: torch.nn.Module, model_name: str = "Model"):
        """
        Moves a model to GPU for inference. 
        """
        print(f"[MemoryManager] Loading {model_name} to {self.device}...")
        
        # In a real SOTA FP8 implementation, we'd keep weights in FP8 and only cast activations.
        model.to(self.device)
        
        # Verify memory
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / 1024**3
            print(f"[MemoryManager] VRAM used: {mem_used:.2f} GB")

    def offload_model(self, model: torch.nn.Module, model_name: str = "Model"):
        """Moves model back to CPU to free up VRAM for the next pipeline stage."""
        print(f"[MemoryManager] Offloading {model_name} to {self.offload_device}...")
        model.to(self.offload_device)
        self.cleanup()

    def cleanup(self):
        """Aggressive garbage collection and cache clearing."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    @contextmanager
    def execution_scope(self, model: torch.nn.Module, model_name: str = "Model"):
        """
        Context manager for temporary execution.
        Usage:
            with memory_manager.execution_scope(unet, "UNet"):
                output = unet(input)
        """
        try:
            self.load_model_to_gpu(model, model_name)
            yield model
        finally:
            self.offload_model(model, model_name)

    def quantize_to_fp8(self, model: torch.nn.Module):
        """
        Helper to convert a model's weights to FP8 for storage.
        NOTE: This requires the model to support mixed precision or be castable.
        """
        if not self.supports_fp8:
            print("[MemoryManager] FP8 not supported, skipping quantization.")
            return model
            
        print("[MemoryManager] Quantizing model weights to FP8 (e4m3fn)...")
        for name, module in model.named_parameters():
             # Basic casting (naive). Real implementations use torchAO or bitsandbytes Linear8bitLt
             module.data = module.data.to(self.storage_dtype)
        
        return model

class DynamicQuantLinear(torch.nn.Module):
    """
    V2.1 SOTA Feature: Group-wise Dynamic Activation Quantization.
    Protects 'Salient Channels' in Flux DiT by calculating scales on-the-fly per token/group.
    Wraps a standard nn.Linear layer.
    """
    def __init__(self, original_linear: torch.nn.Linear, group_size: int = 128):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.group_size = group_size
        
        # Keep weights in original precision (BF16) or FP8, but we need high precision for the kernel
        self.weight = original_linear.weight
        if original_linear.bias is not None:
            self.bias = original_linear.bias
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]
        Performs group-wise quantization simulation (or actual kernel if available).
        For this Python implementation, we simulate the numerical protection ensuring 
        outliers don't saturate the entire tensor.
        """
        dtype = x.dtype
        x_float = x.float() # Calc in FP32
        
        # 1. Grouping
        # [B, L, D] -> [B, L, Groups, GroupDim]
        B, L, D = x.shape
        # Pad if not divisible
        if D % self.group_size != 0:
            pad = self.group_size - (D % self.group_size)
            x_float = torch.nn.functional.pad(x_float, (0, pad))
            D_padded = D + pad
        else:
            D_padded = D
            
        num_groups = D_padded // self.group_size
        x_grouped = x_float.view(B, L, num_groups, self.group_size)
        
        # 2. Calculate Scale per Group (Dynamic)
        # s = max(|x|) / 127
        mx = x_grouped.abs().max(dim=-1, keepdim=True)[0]
        mx = torch.clamp(mx, min=1e-5)
        scales = mx / 127.0
        
        # 3. Quantize (Fake Quant) to simulate W8A8 effects but protected
        x_quant = torch.round(x_grouped / scales)
        x_quant = torch.clamp(x_quant, -128, 127)
        
        # 4. Dequantize
        x_dequant = x_quant * scales
        x_dequant = x_dequant.view(B, L, D_padded)
        
        # Remove padding
        if D_padded != D:
            x_dequant = x_dequant[..., :D]
            
        # 5. Linear projection
        # Ideally, we would use an I8 kernel here. For now we use the protected FP activation against weight.
        out = torch.nn.functional.linear(x_dequant.to(dtype), self.weight, self.bias)
        return out

def apply_dynamic_quant(model: torch.nn.Module, group_size: int = 128):
    """
    Recursively replaces nn.Linear with DynamicQuantLinear in a model.
    """
    print(f"[MemoryManager] Applying Dynamic Activation Quantization (GroupSize={group_size})...")
    count = 0
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # Wrap it
            setattr(model, name, DynamicQuantLinear(module, group_size=group_size))
            count += 1
        else:
            # Recurse
            apply_dynamic_quant(module, group_size)
    if count > 0:
        print(f"[MemoryManager] Wrapped {count} layers in {model.__class__.__name__}")

