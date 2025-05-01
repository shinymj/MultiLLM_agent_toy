import json
import os
import time
import argparse
import datetime
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline, Ollama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Transformers imports (사용하지 않을 경우 제거 가능)
from transformers import pipeline

# Load environment variables for API keys
load_dotenv()

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Test LLMs on BigGen Bench dataset")
    parser.add_argument("--data-path", type=str, default="biggen_bench_instruction_idx0.json",
                        help="Path to the benchmark data file")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--use-ollama", action="store_true",
                        help="Use Ollama local models")
    parser.add_argument("--use-openai", action="store_true",
                        help="Use OpenAI API (requires API key)")
    parser.add_argument("--use-anthropic", action="store_true",
                        help="Use Anthropic API (requires API key)")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Start index for test cases (for partial runs)")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="End index for test cases (for partial runs)")
    return parser.parse_args()

# Setup file paths with timestamp
def setup_file_paths(args):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define file paths
    result_path = os.path.join(args.output_dir, f"biggen_bench_instruction_idx0_result_{timestamp}.json")
    evaluation_path = os.path.join(args.output_dir, f"biggen_bench_instruction_idx0_evaluation_{timestamp}.json")
    score_path = os.path.join(args.output_dir, f"biggen_bench_instruction_idx0_score_{timestamp}.json")
    
    return result_path, evaluation_path, score_path

# Ollama 모델을 설정하는 함수
def setup_ollama_model(model_name):
    """Ollama 모델을 LangChain과 통합하기 위한 설정"""
    print(f"Setting up Ollama model: {model_name}...")
    try:
        # Ollama LLM 설정
        llm = Ollama(
            model=model_name,
            temperature=0.1,
            repeat_penalty=1.15,
            top_p=0.95,
            num_predict=2048
        )
        print(f"Ollama model {model_name} setup successfully")
        return llm
    except Exception as e:
        print(f"Error setting up Ollama model {model_name}: {e}")
        return None

# Setup LLM agents
def setup_agents(args):
    agents = {}
    
    # Ollama 모델 설정 (로컬)
    if args.use_ollama:
        # Ollama 모델 리스트
        ollama_models = ["llama3.1", "deepseek-r1", "gemma3"]
        
        for model_name in ollama_models:
            llm = setup_ollama_model(model_name)
            if llm:
                agents[model_name] = llm
    
    # API 모델 설정 (OpenAI)
    if args.use_openai:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("Warning: OPENAI_API_KEY not found in .env file. Skipping GPT-4o setup.")
        else:
            print("Setting up GPT-4o via OpenAI API...")
            agents["gpt-4o"] = ChatOpenAI(
                model_name="gpt-4o", 
                temperature=0.1,
                openai_api_key=openai_api_key
            )
            print("GPT-4o setup complete")
    
    # API 모델 설정 (Anthropic)
    if args.use_anthropic:
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            print("Warning: ANTHROPIC_API_KEY not found in .env file. Skipping Claude setup.")
        else:
            print("Setting up Claude 3.5 Sonnet via Anthropic API...")
            agents["claude-sonnet-3-5"] = ChatAnthropic(
                model_name="claude-3-5-sonnet-20240620", 
                temperature=0.1,
                anthropic_api_key=anthropic_api_key
            )
            print("Claude 3.5 Sonnet setup complete")
    
    if not agents:
        raise ValueError("No models specified. Please provide at least one model option (--use-ollama, --use-openai, or --use-anthropic).")
    
    return agents

# 이하 나머지 함수는 원래 코드와 동일하게 유지
# extract_test_inputs, get_agent_responses, combine_results, evaluate_responses, calculate_scores, print_summary, main 함수 등

# Main 함수 예시
def main():
    # Parse arguments
    args = parse_args()
    
    # Setup file paths
    result_path, evaluation_path, score_path = setup_file_paths(args)
    
    # Load the benchmark data
    try:
        with open(args.data_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} test cases from {args.data_path}")
    except Exception as e:
        print(f"Error loading benchmark data: {e}")
        return
    
    # Extract test inputs
    test_inputs = extract_test_inputs(data, args.start_idx, args.end_idx)
    print(f"Extracted {len(test_inputs)} test inputs (from index {args.start_idx} to {args.end_idx if args.end_idx else len(data)})")
    
    # Setup agents
    try:
        agents = setup_agents(args)
        print(f"Setup {len(agents)} agents: {', '.join(agents.keys())}")
    except Exception as e:
        print(f"Error setting up agents: {e}")
        return
    
    # Get responses from agents
    agent_responses = get_agent_responses(agents, test_inputs)
    
    # 나머지 코드는 이전과 동일...

if __name__ == "__main__":
    main()
