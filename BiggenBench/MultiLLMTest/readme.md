# BigGen Bench Testing Framework

This framework allows you to test multiple language models (both local and API-based) against the BigGen Bench instruction dataset.

## Overview

biggen_bench_testing/
├── biggen_bench_test.py          # Main testing script
├── local_model_helper.py         # Utilities for local model loading
├── model_download.py             # Script to download models from HF
├── visualize_results.py          # Script to visualize benchmark results
├── run_pipeline.sh               # Full pipeline automation script
├── .env                          # API keys (you must create this)
├── biggen_bench_instruction_idx0.json  # Benchmark data
├── results/                      # Directory for benchmark results
├── models/                       # Directory for downloaded models
└── visualizations/               # Directory for visualization outputs

The testing process follows these steps:

1. **Extract test inputs** from the BigGen dataset (excluding reference answers and rubrics)
2. **Get responses** from different LLM agents
3. **Evaluate the responses** using reference answers and scoring rubrics
4. **Calculate total scores** by agent

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- LangChain
- Access to local LLM models or API keys for OpenAI/Anthropic

## Setup

1. Clone this repository
2. Install the required dependencies:

```bash
python setup.py --llama-path "/path/to/llama3.1" --deepseek-path "/path/to/deepseek-r1-8b" --gemma-path "/path/to/gemma3-12b"
```

3. Edit the `.env` file to add your API keys:

```
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Working with Local Models

### Model Loading

The script supports loading local models using Hugging Face Transformers. Here's how various model types should be loaded:

#### LLaMA 3.1

LLaMA 3.1 models are typically downloaded from Meta or converted from GGUF format. The path should point to the directory containing:
- `config.json`
- `tokenizer.model`
- `model.safetensors` (or sharded files)

#### DeepSeek R1:8B

DeepSeek models follow the standard Hugging Face structure. The path should point to a directory with:
- `config.json`
- `tokenizer_config.json`
- Model weights (safetensors or bin files)

#### Gemma 3:12B

Gemma models require access from Google and are typically downloaded through Hugging Face. The path should contain the model files similar to other Hugging Face models.

### Running with Custom Models

If you want to use different local models, modify the `setup_agents` function in `biggen_bench_test.py`:

```python
def setup_agents():
    agents = {}
    
    # Change model paths as needed
    llama_path = "/path/to/your/model1"
    deepseek_path = "/path/to/your/model2"
    gemma_path = "/path/to/your/model3"
    
    # Setup local models (modify names as appropriate)
    agents["your_model1"] = setup_local_model("your_model1", llama_path)
    agents["your_model2"] = setup_local_model("your_model2", deepseek_path)
    # etc.
    
    return agents
```

## Running the Benchmark

```bash
python biggen_bench_test.py
```

This will:
1. Load the models
2. Generate responses for each test case
3. Evaluate the responses
4. Calculate and save the final scores

## Output Files

The script generates three JSON files with timestamps:

1. `biggen_bench_instruction_idx0_result_YYYYMMDDHHMMSS.json` - Contains all test inputs and responses from each agent
2. `biggen_bench_instruction_idx0_evaluation_YYYYMMDDHHMMSS.json` - Contains evaluations for each response
3. `biggen_bench_instruction_idx0_score_YYYYMMDDHHMMSS.json` - Contains summary scores for each agent

## Customizing the Evaluation

By default, Claude is used to evaluate responses. You can modify the `evaluate_responses` function in `biggen_bench_test.py` to use a different evaluator or approach.

## Troubleshooting

### Memory Issues

If you encounter GPU memory errors:
- Try using model quantization (4-bit or 8-bit)
- Set `device_map="auto"` to distribute model across multiple GPUs
- For very large models, consider using offloading to CPU

### API Rate Limits

The script includes retry logic for API calls, but you may still hit rate limits. To mitigate:
- Add longer sleep times between retries
- Process fewer test cases at a time

## License

This project is intended for research purposes only. Use of the various models are subject to their respective licenses.
