import os
import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def get_available_memory():
    """Get available system and GPU memory in GB"""
    # System RAM
    ram_gb = psutil.virtual_memory().available / (1024 ** 3)
    
    # GPU memory if available
    gpu_gb = 0
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        # Get free memory - this is approximate
        gpu_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        gpu_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        gpu_gb = gpu_gb - gpu_reserved
    
    return ram_gb, gpu_gb

def optimize_model_loading(model_path, model_type):
    """
    Configure model loading parameters based on available resources and model type
    
    Args:
        model_path: Path to the model directory
        model_type: One of "llama", "deepseek", "gemma", or "other"
        
    Returns:
        Dictionary of kwargs for model loading
    """
    ram_gb, gpu_gb = get_available_memory()
    print(f"Available RAM: {ram_gb:.2f} GB, Available GPU: {gpu_gb:.2f} GB")
    
    # Base configuration
    config = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "trust_remote_code": True,
    }
    
    # Model-specific optimizations
    model_size_gb = estimate_model_size(model_path)
    print(f"Estimated model size: {model_size_gb:.2f} GB")
    
    # Decide on quantization based on model size vs available resources
    if torch.cuda.is_available():
        if model_size_gb > gpu_gb * 0.8:  # If model would use more than 80% of GPU
            print("Model is large relative to GPU memory. Using 4-bit quantization.")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            config["quantization_config"] = quantization_config
        elif model_size_gb > gpu_gb * 0.5:  # If model would use more than 50% of GPU
            print("Model will use significant GPU memory. Using 8-bit quantization.")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            config["quantization_config"] = quantization_config
    else:
        # CPU-only machine
        if model_size_gb > ram_gb * 0.5:
            print("No GPU available and model is large. Using 8-bit quantization.")
            config["load_in_8bit"] = True
    
    # Model-specific adjustments
    if model_type == "llama":
        # LLaMA models work well with specific settings
        config["rope_scaling"] = {"type": "dynamic", "factor": 2.0}
    elif model_type == "gemma":
        # Gemma might need specific flash attention settings
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            config["use_flash_attention_2"] = True
    
    return config

def estimate_model_size(model_path):
    """Estimate model size in GB based on files in directory"""
    total_size = 0
    
    try:
        # Check if it's a directory
        if os.path.isdir(model_path):
            for root, _, files in os.walk(model_path):
                for file in files:
                    # Only count model weight files
                    if file.endswith(('.bin', '.pt', '.safetensors', '.gguf')):
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)
        # Single file (like .gguf models)
        elif os.path.isfile(model_path) and model_path.endswith(('.bin', '.pt', '.safetensors', '.gguf')):
            total_size = os.path.getsize(model_path)
    except Exception as e:
        print(f"Error estimating model size: {e}")
        # Return a conservative estimate
        return 20  # Assume 20GB if we can't determine
        
    # Convert to GB
    return total_size / (1024 ** 3)

def load_local_model(model_path, model_type="other"):
    """
    Load a local model with optimized settings based on available resources
    
    Args:
        model_path: Path to the model directory
        model_type: One of "llama", "deepseek", "gemma", or "other"
        
    Returns:
        Loaded model and tokenizer
    """
    print(f"Loading {model_type} model from {model_path}...")
    
    # Get optimized loading config
    config = optimize_model_loading(model_path, model_type)
    
    # Special handling for tokenizer
    tokenizer_kwargs = {"use_fast": True}
    if model_type == "llama" or model_type == "gemma":
        tokenizer_kwargs["use_fast"] = False  # Some models work better with slow tokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        **tokenizer_kwargs
    )
    
    # Set tokenizer padding settings
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **config
    )
    
    return model, tokenizer

# Example usage function
def demo_model_loading():
    """Demonstration of how to use the model loading utilities"""
    model_configs = [
        {"path": "/path/to/llama3.1", "type": "llama"},
        {"path": "/path/to/deepseek-r1-8b", "type": "deepseek"},
        {"path": "/path/to/gemma3-12b", "type": "gemma"}
    ]
    
    loaded_models = {}
    
    for config in model_configs:
        try:
            model, tokenizer = load_local_model(config["path"], config["type"])
            model_name = os.path.basename(config["path"])
            loaded_models[model_name] = (model, tokenizer)
            
            # Test the model with a simple generation
            inputs = tokenizer("Hello, I am", return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs, 
                max_new_tokens=20,
                temperature=0.7,
                top_p=0.9
            )
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Model {model_name} output: {result}")
            
        except Exception as e:
            print(f"Failed to load model {config['path']}: {e}")
    
    return loaded_models

if __name__ == "__main__":
    print("Local Model Loading Utility")
    print("---------------------------")
    print("This script provides utilities for efficiently loading large language models.")
    print("To use, import the functions in your own scripts:")
    print("from local_model_helper import load_local_model")
    
    # Uncomment to test loading models
    # loaded_models = demo_model_loading()