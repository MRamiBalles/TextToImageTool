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
