#!/bin/bash

# Full BigGen Bench testing pipeline script
# This script automates the entire process of downloading models,
# running benchmarks, and visualizing results

# Set up colored output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create directories
MODELS_DIR="models"
RESULTS_DIR="results"
VISUALIZATIONS_DIR="visualizations"

mkdir -p "$MODELS_DIR" "$RESULTS_DIR" "$VISUALIZATIONS_DIR"

# Function to display section headers
section() {
    echo -e "\n${BLUE}======== $1 ========${NC}\n"
}

# Function to check if command was successful
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $1 completed successfully${NC}"
    else
        echo -e "${RED}❌ $1 failed${NC}"
        exit 1
    fi
}

# Parse command line arguments
HF_TOKEN=""
RUN_LLAMA=false
RUN_DEEPSEEK=false
RUN_GEMMA=false
RUN_API=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --llama)
            RUN_LLAMA=true
            shift
            ;;
        --deepseek)
            RUN_DEEPSEEK=true
            shift
            ;;
        --gemma)
            RUN_GEMMA=true
            shift
            ;;
        --api)
            RUN_API=true
            shift
            ;;
        --all)
            RUN_LLAMA=true
            RUN_DEEPSEEK=true
            RUN_GEMMA=true
            RUN_API=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--token HF_TOKEN] [--llama] [--deepseek] [--gemma] [--api] [--all]"
            exit 1
            ;;
    esac
done

# Check if at least one model or API is selected
if ! $RUN_LLAMA && ! $RUN_DEEPSEEK && ! $RUN_GEMMA && ! $RUN_API; then
    echo -e "${YELLOW}No models or APIs selected. Using --deepseek by default.${NC}"
    RUN_DEEPSEEK=true
fi

# Step 1: Check for benchmark data
section "Checking Benchmark Data"

if [ ! -f "biggen_bench_instruction_idx0.json" ]; then
    echo -e "${YELLOW}Benchmark data file not found. Creating a symlink to the uploaded file...${NC}"
    # Try to find the uploaded file
    UPLOADED_FILE=$(find . -maxdepth 1 -name "biggen_bench_instruction_idx0.json" | head -n 1)
    
    if [ -z "$UPLOADED_FILE" ]; then
        echo -e "${RED}Could not find benchmark data file.${NC}"
        exit 1
    fi
    
    ln -sf "$UPLOADED_FILE" "biggen_bench_instruction_idx0.json"
    check_status "Creating symlink to benchmark data"
else
    echo -e "${GREEN}Benchmark data file found.${NC}"
fi

# Step 2: Install dependencies
section "Installing Dependencies"

echo "Installing Python dependencies..."
pip install -q transformers langchain langchain-core langchain-community langchain-openai langchain-anthropic torch accelerate huggingface_hub tqdm numpy matplotlib seaborn pandas python-dotenv safetensors sentencepiece bitsandbytes
check_status "Dependencies installation"

# Step 3: Download models (if requested)
if $RUN_LLAMA || $RUN_DEEPSEEK || $RUN_GEMMA; then
    section "Downloading Models"
    
    DOWNLOAD_CMD="python model_download.py --output-dir $MODELS_DIR"
    
    if $RUN_LLAMA; then
        DOWNLOAD_CMD="$DOWNLOAD_CMD --llama"
    fi
    
    if $RUN_DEEPSEEK; then
        DOWNLOAD_CMD="$DOWNLOAD_CMD --deepseek"
    fi
    
    if $RUN_GEMMA; then
        DOWNLOAD_CMD="$DOWNLOAD_CMD --gemma"
    fi
    
    if [ ! -z "$HF_TOKEN" ]; then
        DOWNLOAD_CMD="$DOWNLOAD_CMD --token $HF_TOKEN"
    fi
    
    echo "Running: $DOWNLOAD_CMD"
    eval $DOWNLOAD_CMD
    check_status "Model download"
fi

# Step 4: Run benchmark
section "Running Benchmark"

BENCHMARK_CMD="python biggen_bench_test.py --output-dir $RESULTS_DIR"

if $RUN_LLAMA && [ -d "$MODELS_DIR/llama3.1" ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --llama-path $MODELS_DIR/llama3.1"
fi

if $RUN_DEEPSEEK && [ -d "$MODELS_DIR/deepseek-r1-8b" ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --deepseek-path $MODELS_DIR/deepseek-r1-8b"
fi

if $RUN_GEMMA && [ -d "$MODELS_DIR/gemma3-12b" ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --gemma-path $MODELS_DIR/gemma3-12b"
fi

if $RUN_API; then
    BENCHMARK_CMD="$BENCHMARK_CMD --use-openai --use-anthropic"
fi

echo "Running: $BENCHMARK_CMD"
eval $BENCHMARK_CMD
check_status "Benchmark execution"

# Step 5: Visualize results
section "Visualizing Results"

echo "Generating visualizations..."
python visualize_results.py --results-dir "$RESULTS_DIR" --output-dir "$VISUALIZATIONS_DIR"
check_status "Visualization generation"

# Final message
section "Benchmark Pipeline Completed"

echo -e "${GREEN}The benchmark pipeline has completed successfully!${NC}"
echo -e "Results are available in the following locations:"
echo -e "- Raw benchmark results: ${YELLOW}$RESULTS_DIR${NC}"
echo -e "- Visualizations: ${YELLOW}$VISUALIZATIONS_DIR${NC}"
echo -e "- Report: ${YELLOW}$VISUALIZATIONS_DIR/benchmark_report.html${NC}"

echo -e "\nTo view the HTML report, open the following file in your browser:"
echo -e "${BLUE}$(realpath "$VISUALIZATIONS_DIR/benchmark_report.html")${NC}"
