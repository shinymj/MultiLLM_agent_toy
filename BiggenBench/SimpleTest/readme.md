# AI Agent Benchmark Tool

This tool allows you to benchmark different AI agents (like Llama 3.1 and Gemma 3) against instruction following tasks and evaluate their performance using Claude 3.7.

## Requirements

- Python 3.8+
- Ollama installed locally with the required models
- Anthropic API key for Claude 3.7 (in .env file or passed as parameter)
- Required Python packages:
  - langchain
  - langchain_community
  - langchain_anthropic
  - python-dotenv

## Installation

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the same directory with your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-api123...
```

2. Make sure Ollama is installed and the required models are downloaded:

```bash
ollama pull llama3.1
ollama pull gemma3:12b
```

## Usage

Run the script with the following command:

```bash
python benchmark_agents.py
```

The script will automatically use the API key from your `.env` file.

### Optional arguments:

- `--benchmark`: Path to the benchmark JSON file (default: "biggen_bench_instruction_idx0.json")
- `--agent1`: First agent model name for Ollama (default: "llama3.1") 
- `--agent2`: Second agent model name for Ollama (default: "gemma3:12b")
- `--anthropic_api_key`: Manually specify the Anthropic API key (overrides the .env file)

### Example:

```bash
# Using API key from .env file
python benchmark_agents.py --agent1 llama3.1 --agent2 gemma3:12b

# Or manually specifying API key
python benchmark_agents.py --anthropic_api_key sk-ant-api123... --agent1 llama3.1 --agent2 gemma3:12b
```

## Output

The script generates two JSON files:
1. `biggen_bench_instruction_idx0_result_YYYYMMDDHHMMSS.json` - Contains the responses from both agents
2. `biggen_bench_instruction_idx0_evaluation_YYYYMMDDHHMMSS.json` - Contains Claude 3.7's evaluation of each response

## Customizing Models

You can use different Ollama models by changing the `--agent1` and `--agent2` parameters. The script can be extended to support other model providers by modifying the `get_agent_responses` function.