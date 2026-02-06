import os
import sys

# Cache redirection
os.environ["HF_HOME"] = "D:/TextToImageTool/cache"

try:
    print("Testing Transformers Import...")
    import transformers
    print(f"Transformers Version: {transformers.__version__}")

    print("Testing CLIPTextConfig Import...")
    from transformers import CLIPTextConfig, CLIPTextModel
    
    model_id = "openai/clip-vit-large-patch14"
    print(f"Loading Config for {model_id}...")
    
    # Try manual config load
    config = CLIPTextConfig.from_pretrained(model_id)
    print("Config loaded successfully.")
    print(f"Hidden Size: {config.hidden_size}")
    
    print("Loading Model with explicit config...")
    model = CLIPTextModel.from_pretrained(model_id, config=config)
    print("Model loaded successfully.")
    
except Exception as e:
    print(f"\nCRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
