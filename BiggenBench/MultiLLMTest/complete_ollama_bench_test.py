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
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

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

# Extract test inputs from the benchmark data
def extract_test_inputs(data: List[Dict[str, Any]], start_idx=0, end_idx=None) -> List[Dict[str, Any]]:
    test_inputs = []
    
    # Apply indices for partial runs
    if end_idx is None:
        end_idx = len(data)
    
    data_subset = data[start_idx:end_idx]
    
    for instance in data_subset:
        test_input = {
            "id": instance["id"],
            "capability": instance["capability"],
            "task": instance["task"],
            "instance_idx": instance["instance_idx"],
            "system_prompt": instance["system_prompt"],
            "input": instance["input"]
        }
        test_inputs.append(test_input)
    
    return test_inputs

# Get responses from agents
def get_agent_responses(agents: Dict[str, Any], test_inputs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    responses = {agent_name: [] for agent_name in agents.keys()}
    
    for test_idx, test_input in enumerate(test_inputs):
        print(f"Processing test input {test_idx+1}/{len(test_inputs)}: {test_input['id']}")
        
        for agent_name, agent in agents.items():
            print(f"  Getting response from {agent_name}...")
            
            # 에이전트 유형에 따라 다른 프롬프트 템플릿 사용
            if agent_name.startswith("llama") or agent_name.startswith("deepseek") or agent_name.startswith("gemma"):
                # Ollama 모델은 특별한 프롬프트 포맷이 필요할 수 있음
                prompt = ChatPromptTemplate.from_messages([
                    ("system", test_input["system_prompt"]),
                    ("human", test_input["input"])
                ])
            else:
                # API 모델 (OpenAI, Anthropic)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", test_input["system_prompt"]),
                    ("human", test_input["input"])
                ])
            
            # Create chain
            chain = prompt | agent | StrOutputParser()
            
            # Get response with retry logic
            max_retries = 3
            response = None
            
            for attempt in range(max_retries):
                try:
                    response = chain.invoke({})
                    break
                except Exception as e:
                    print(f"    Error with {agent_name} (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        delay = 2 ** attempt * 5  # Exponential backoff
                        print(f"    Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        response = f"ERROR: Failed to get response after {max_retries} attempts"
            
            # Add response to results
            response_item = {
                "id": test_input["id"],
                "agent": agent_name,
                "system_prompt": test_input["system_prompt"],
                "input": test_input["input"],
                "response": response
            }
            responses[agent_name].append(response_item)
            
            # Add small delay to avoid rate limits for API models
            if agent_name in ["gpt-4o", "claude-sonnet-3-5"]:
                time.sleep(1)
    
    return responses

# Combine test inputs with agent responses
def combine_results(test_inputs: List[Dict[str, Any]], agent_responses: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    results = []
    
    for i, test_input in enumerate(test_inputs):
        result = {
            "id": test_input["id"],
            "capability": test_input["capability"],
            "task": test_input["task"],
            "instance_idx": test_input["instance_idx"],
            "system_prompt": test_input["system_prompt"],
            "input": test_input["input"],
            "agent_responses": {}
        }
        
        for agent_name in agent_responses:
            # Check if we have enough responses (in case of partial failures)
            if i < len(agent_responses[agent_name]):
                result["agent_responses"][agent_name] = agent_responses[agent_name][i]["response"]
        
        results.append(result)
    
    return results

# Evaluate agent responses
def evaluate_responses(original_data: List[Dict[str, Any]], results: List[Dict[str, Any]], start_idx=0) -> List[Dict[str, Any]]:
    evaluations = []
    
    # Check if we have Anthropic API key for evaluation
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in .env file. Evaluation requires Claude.")
    
    # Use Claude for evaluation
    evaluator = ChatAnthropic(
        model_name="claude-3-5-sonnet-20240620",
        temperature=0,
        anthropic_api_key=anthropic_api_key
    )
    
    # Map results to original data using indices
    for i, result in enumerate(results):
        original_idx = i + start_idx
        original_item = original_data[original_idx]
        
        eval_item = {
            "id": result["id"],
            "capability": result["capability"],
            "task": result["task"],
            "instance_idx": result["instance_idx"],
            "reference_answer": original_item["reference_answer"],
            "score_rubric": original_item["score_rubric"],
            "evaluations": {}
        }
        
        print(f"Evaluating responses for: {result['id']}")
        
        for agent_name in result["agent_responses"]:
            response = result["agent_responses"][agent_name]
            
            # Skip if response is an error
            if response and response.startswith("ERROR:"):
                print(f"  Skipping evaluation for {agent_name} due to error in response")
                eval_item["evaluations"][agent_name] = {
                    "score": 0,
                    "rationale": "Response generation failed"
                }
                continue
            
            # Create evaluation prompt
            evaluation_prompt = f"""
            As an expert evaluator, you need to score a model's response according to a specific rubric.

            REFERENCE ANSWER:
            {original_item["reference_answer"]}

            SCORE RUBRIC:
            Criteria: {original_item["score_rubric"]["criteria"]}
            Score 1 - {original_item["score_rubric"]["score1_description"]}
            Score 2 - {original_item["score_rubric"]["score2_description"]}
            Score 3 - {original_item["score_rubric"]["score3_description"]}
            Score 4 - {original_item["score_rubric"]["score4_description"]}
            Score 5 - {original_item["score_rubric"]["score5_description"]}

            AGENT RESPONSE TO EVALUATE:
            {response}

            Your task:
            1. Assign a score from 1-5 based on the rubric
            2. Provide a detailed rationale for your scoring decision

            Output format:
            {{
                "score": [score as an integer],
                "rationale": "[detailed explanation for the score]"
            }}"""