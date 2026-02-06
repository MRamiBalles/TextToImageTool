import torch
from transformers import T5EncoderModel, AutoTokenizer, CLIPTextModel, CLIPTokenizer
from typing import Tuple, Optional
from src.engine.memory_manager import MemoryManager

class DualTextEncoder:
    """
    Handles the two text encoders for Flux.1:
    1. CLIP ViT-L/14 (Pooled embedding for global context)
    2. T5-XXL (Token embeddings for fine-grained details)
    
    Implements STRICT sequential offloading to fit in 12GB VRAM.
    Supports DistillT5 fallback for storage constrained environments.
    """
    def __init__(self, memory_manager: MemoryManager, use_distill_t5: bool = False, cache_dir: Optional[str] = None):
        self.mm = memory_manager
        self.device = self.mm.device
        self.cache_dir = cache_dir
        
        # Paths or IDs
        self.clip_id = "openai/clip-vit-large-patch14" 
        # Use T5-XXL by default, or DistillT5/Base if requested to save ~10GB Disk/VRAM
        self.t5_id = "google/t5-v1_1-xxl" if not use_distill_t5 else "google/t5-v1_1-base" 
        # Note: True DistillT5 weights might require specific repo IDs like 'LifuWang/DistillT5' 
        # but t5-v1_1-base is a safe 250MB fallback for "Smoke Tests".
        
        if use_distill_t5:
            print(f"[DualTextEncoder] Optimization Active: Using lightweight T5 ({self.t5_id})")
        
        # Tokenizers (Always in CPU/RAM, lightweight)
        print("[DualTextEncoder] Loading Tokenizers...")
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(self.clip_id, cache_dir=self.cache_dir)
        # Use AutoTokenizer with use_fast=False to avoid tiktoken conversion issues
        self.t5_tokenizer = AutoTokenizer.from_pretrained(self.t5_id, use_fast=False, cache_dir=self.cache_dir)
        
    def encode(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the prompt using both models sequentially.
        Returns:
            pooled_prompt_embeds: [B, D_clip] (Vector)
            prompt_embeds: [B, L, D_t5] (Sequence)
        """
        print(f"[DualTextEncoder] Encoding prompt: '{prompt}'")
        
        # 1. CLIP Encoding
        # Load CLIP
        print("[DualTextEncoder] Loading CLIP...")
        try:
            # Fix for Transformers 5.1.0 / Python 3.14: Explicitly load CLIPTextConfig
            # because from_pretrained might be passing the full CLIPConfig (vision+text)
            # which lacks 'hidden_size' at the root level.
            from transformers import CLIPTextConfig
            clip_config = CLIPTextConfig.from_pretrained(self.clip_id, cache_dir=self.cache_dir)
            clip_model = CLIPTextModel.from_pretrained(self.clip_id, config=clip_config, cache_dir=self.cache_dir)
        except Exception as e:
            print(f"Warning: Manual Config Load failed ({e}). Trying default...")
            clip_model = CLIPTextModel.from_pretrained(self.clip_id, cache_dir=self.cache_dir)
            
        self.mm.load_model_to_gpu(clip_model, "CLIP")
        
        # Tokenize
        text_inputs = self.clip_tokenizer(
            prompt, 
            padding="max_length", 
            max_length=77, 
            truncation=True, 
            return_tensors="pt"
        )
        
        # Inference
        with torch.no_grad():
            outputs = clip_model(text_inputs.input_ids.to(self.device))
            pooled_prompt_embeds = outputs.pooler_output # [B, 768] (or 1024 for L/14? Correct is 768 for ViT-L/14 usually, but Flux might use specialized projection)
        
        # Offload CLIP
        self.mm.offload_model(clip_model, "CLIP")
        del clip_model
        
        # 2. T5 Encoding
        # Load T5 (This is the heavy step)
        print(f"[DualTextEncoder] Loading {self.t5_id} (This may take time)...")
        # Optimization: In a real deploy, we would load quantized weights directly.
        # Here we rely on the memory manager or load_in_8bit if supported by libs.
        try:
             # Try loading with fp8 if available in transformers + bitsandbytes
             t5_model = T5EncoderModel.from_pretrained(
                 self.t5_id, 
                 torch_dtype=self.mm.storage_dtype,
                 low_cpu_mem_usage=True,
                 cache_dir=self.cache_dir
             )
        except Exception as e:
            print(f"Warning: Failed to load T5 with optimization. Fallback. Error: {e}")
            t5_model = T5EncoderModel.from_pretrained(self.t5_id, cache_dir=self.cache_dir)
            
        self.mm.load_model_to_gpu(t5_model, "T5-XXL")
        
        # Tokenize T5
        text_inputs_t5 = self.t5_tokenizer(
            prompt,
            padding="max_length",
            max_length=256, # Flux default 256 or 512
            truncation=True,
            return_tensors="pt"
        )
        
        # Inference
        with torch.no_grad():
            outputs_t5 = t5_model(text_inputs_t5.input_ids.to(self.device))
            prompt_embeds = outputs_t5.last_hidden_state
            
        # Offload T5
        self.mm.offload_model(t5_model, "T5-XXL")
        del t5_model
        self.mm.cleanup()
        
        return pooled_prompt_embeds, prompt_embeds
