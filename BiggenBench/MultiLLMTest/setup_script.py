import os
import argparse
import subprocess
from pathlib import Path

def install_dependencies():
    """Install necessary Python packages"""
    requirements = [
        "langchain",
        "langchain-core",
        "langchain-community", 
        "langchain-openai", 
        "langchain-anthropic",
        "transformers",
        "accelerate",  # For GPU optimization
        "torch",       # PyTorch
        "sentencepiece",  # For tokenization with some models
        "python-dotenv",  # For loading API keys
        "bitsandbytes",   # For quantization support
        "safetensors",    # For model loading
    ]
    
    print("Installing dependencies...")
    subprocess.check_call(["pip", "install"] + requirements)
    print("Dependencies installed successfully!")

def create_env_file():
    """Create a .env file for API keys if it doesn't exist"""
    if not os.path.exists(".env"):
        print("Creating .env file for API keys...")
        with open(".env", "w") as f:
            f.write("# API Keys for external LLM services\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
            f.write("ANTHROPIC_API_KEY=your_anthropic_api_key_here\n")
        print(".env file created. Please edit it to add your actual API keys.")
    else:
        print(".env file already exists.")

def check_model_paths(args):
    """Check if the specified model paths exist"""
    models = {
        "llama3.1": args.llama_path,
        "deepseek-r1:8b": args.deepseek_path,
        "gemma3:12b": args.gemma_path
    }
    
    missing_models = []
    for model_name, path in models.items():
        if not path or not os.path.exists(path):
            missing_models.append(model_name)
    
    if missing_models:
        print(f"Warning: The following model paths are missing or invalid: {', '.join(missing_models)}")
        print("Please download these models or provide correct paths.")
    else:
        print("All model paths exist.")
    
    return models

def update_model_paths(models):
    """Update the benchmark script with the correct model paths"""
    script_path = "biggen_bench_test.py"
    
    if os.path.exists(script_path):
        with open(script_path, "r") as f:
            content = f.read()
        
        # Update model paths in the script
        for model_name, path in models.items():
            if path:
                # Escape backslashes for string literals in Python
                path_str = str(Path(path)).replace("\\", "\\\\")
                # Replace placeholder with actual path
                content = content.replace(f"path/to/{model_name.lower()}", path_str)
        
        with open(script_path, "w") as f:
            f.write(content)
        
        print(f"Updated model paths in {script_path}")
    else:
        print(f"Warning: Could not find {script_path} to update model paths.")

def main():
    parser = argparse.ArgumentParser(description="Setup for BigGen Bench testing")
    parser.add_argument("--llama-path", type=str, help="Path to llama3.1 model")
    parser.add_argument("--deepseek-path", type=str, help="Path to deepseek-r1:8b model")
    parser.add_argument("--gemma-path", type=str, help="Path to gemma3:12b model")
    parser.add_argument("--skip-install", action="store_true", help="Skip installing dependencies")
    
    args = parser.parse_args()
    
    if not args.skip_install:
        install_dependencies()
    
    create_env_file()
    models = check_model_paths(args)
    update_model_paths(models)
    
    print("\nSetup complete! You can now run the benchmark with:")
    print("python biggen_bench_test.py")

if __name__ == "__main__":
    main()
