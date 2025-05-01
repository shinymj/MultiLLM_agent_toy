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
from langchain_community.llms import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Transformers imports
from transformers import pipeline

# Import local model loading utilities
from local_model_helper import load_local_model

# Load environment variables for API keys
load_dotenv()

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Test LLMs on BigGen Bench dataset")
    parser.add_argument("--data-path", type=str, default="biggen_bench_instruction_idx0.json",
                        help="Path to the benchmark data file")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--llama-path", type=str, default=None,
                        help="Path to llama3.1 model")
    parser.add_argument("--deepseek-path", type=str, default=None,
                        help="Path to deepseek-r1:8b model")
    parser.add_argument("--gemma-path", type=str, default=None,
                        help="Path to gemma3:12b model")
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

# Setup LLM pipeline for local models
def setup_local_llm_pipeline(model, tokenizer, model_type):
    # Model-specific generation parameters
    if model_type == "llama":
        generation_config = {
            "max_new_tokens": 2048,
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.15,
            "do_sample": True
        }
    elif model_type == "deepseek":
        generation_config = {
            "max_new_tokens": 2048,
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
            "repetition_penalty": 1.1,
            "do_sample": True
        }
    elif model_type == "gemma":
        generation_config = {
            "max_new_tokens": 2048,
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.2,
            "do_sample": True
        }
    else:
        # Default settings
        generation_config = {
            "max_new_tokens": 2048,
            "temperature": 0.1,
            "top_p": 0.95,
            "repetition_penalty": 1.15,
            "do_sample": True
        }
    
    # Create text generation pipeline
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **generation_config
    )
    
    # Convert to LangChain LLM
    llm = HuggingFacePipeline(pipeline=text_pipeline)
    return llm

# # Setup LLM agents
# def setup_agents(args):
#     agents = {}
    
#     # Setup local models
#     if args.llama_path:
#         print("Loading LLaMA 3.1 model...")
#         model, tokenizer = load_local_model(args.llama_path, "llama")
#         agents["llama3.1"] = setup_local_llm_pipeline(model, tokenizer, "llama")
#         print("LLaMA 3.1 model loaded successfully")
    
#     if args.deepseek_path:
#         print("Loading DeepSeek R1:8B model...")
#         model, tokenizer = load_local_model(args.deepseek_path, "deepseek")
#         agents["deepseek-r1:8b"] = setup_local_llm_pipeline(model, tokenizer, "deepseek")
#         print("DeepSeek R1:8B model loaded successfully")
    
#     if args.gemma_path:
#         print("Loading Gemma 3:12B model...")
#         model, tokenizer = load_local_model(args.gemma_path, "gemma")
#         agents["gemma3:12b"] = setup_local_llm_pipeline(model, tokenizer, "gemma")
#         print("Gemma 3:12B model loaded successfully")
    
#     # Setup API models
#     if args.use_openai:
#         openai_api_key = os.getenv("OPENAI_API_KEY")
#         if not openai_api_key:
#             print("Warning: OPENAI_API_KEY not found in .env file. Skipping GPT-4o setup.")
#         else:
#             print("Setting up GPT-4o via OpenAI API...")
#             agents["gpt-4o"] = ChatOpenAI(
#                 model_name="gpt-4o", 
#                 temperature=0.1,
#                 openai_api_key=openai_api_key
#             )
#             print("GPT-4o setup complete")
    
#     if args.use_anthropic:
#         anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
#         if not anthropic_api_key:
#             print("Warning: ANTHROPIC_API_KEY not found in .env file. Skipping Claude setup.")
#         else:
#             print("Setting up Claude 3.5 Sonnet via Anthropic API...")
#             agents["claude-sonnet-3-5"] = ChatAnthropic(
#                 model_name="claude-3-5-sonnet-20240620", 
#                 temperature=0.1,
#                 anthropic_api_key=anthropic_api_key
#             )
#             print("Claude 3.5 Sonnet setup complete")
    
#     if not agents:
#         raise ValueError("No models specified. Please provide at least one model path or API option.")
    
#     return agents

def setup_agents():
    agents = {}
    
    # Change model paths as needed
    llama_path = "C:\Users\selin\.ollama\models\manifests\registry.ollama.ai\library\llama3.1"
    deepseek_path = "C:\Users\selin\.ollama\models\manifests\registry.ollama.ai\library\deepseek-r1"
    gemma_path = "C:\Users\selin\.ollama\models\manifests\registry.ollama.ai\library\gemma3"
    
    # Setup local models (modify names as appropriate)
    agents["llama3.1"] = setup_local_model("llama3.1", llama_path)
    agents["deepseek-r1:8b"] = setup_local_model("deepseek-r1:8b", deepseek_path)
    agents["gemma3:12b"] = setup_local_model("gemma3:12b", gemma_path)

        # Setup API models
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
        raise ValueError("No models specified. Please provide at least one model path or API option.")
  
    
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
            
            # Create prompt
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
            }}
            """
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert evaluator who carefully follows scoring rubrics to evaluate model outputs."),
                ("human", evaluation_prompt)
            ])
            
            # Create chain with JSON output
            chain = prompt | evaluator | StrOutputParser()
            
            # Get evaluation with retry logic
            max_retries = 3
            evaluation_json = None
            
            for attempt in range(max_retries):
                try:
                    evaluation_result = chain.invoke({})
                    evaluation_json = json.loads(evaluation_result)
                    break
                except Exception as e:
                    print(f"  Error evaluating {agent_name} (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        delay = 2 ** attempt * 5  # Exponential backoff
                        print(f"  Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        evaluation_json = {
                            "score": 0,
                            "rationale": f"ERROR: Failed to evaluate after {max_retries} attempts"
                        }
            
            eval_item["evaluations"][agent_name] = evaluation_json
            
            # Add delay to avoid rate limits
            time.sleep(2)
        
        evaluations.append(eval_item)
    
    return evaluations

# Calculate total scores by agent
def calculate_scores(evaluations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    scores = {}
    
    # Get list of all agents
    if evaluations and "evaluations" in evaluations[0]:
        agent_names = list(evaluations[0]["evaluations"].keys())
        
        # Initialize scores dictionary
        for agent_name in agent_names:
            scores[agent_name] = {
                "total_score": 0,
                "average_score": 0,
                "scores_by_task": {},
                "scores_by_capability": {}
            }
        
        # Calculate scores
        for evaluation in evaluations:
            task = evaluation["task"]
            capability = evaluation["capability"]
            
            # Initialize task scores if needed
            for agent_name in agent_names:
                # For tasks
                if task not in scores[agent_name]["scores_by_task"]:
                    scores[agent_name]["scores_by_task"][task] = {
                        "scores": [],
                        "total": 0,
                        "average": 0
                    }
                
                # For capabilities
                if capability not in scores[agent_name]["scores_by_capability"]:
                    scores[agent_name]["scores_by_capability"][capability] = {
                        "scores": [],
                        "total": 0,
                        "average": 0
                    }
            
            # Add scores
            for agent_name, agent_eval in evaluation["evaluations"].items():
                score = agent_eval["score"]
                scores[agent_name]["total_score"] += score
                
                # Add to task scores
                scores[agent_name]["scores_by_task"][task]["scores"].append(score)
                scores[agent_name]["scores_by_task"][task]["total"] += score
                
                # Add to capability scores
                scores[agent_name]["scores_by_capability"][capability]["scores"].append(score)
                scores[agent_name]["scores_by_capability"][capability]["total"] += score
        
        # Calculate averages
        for agent_name in agent_names:
            total_tasks = len(evaluations)
            scores[agent_name]["average_score"] = scores[agent_name]["total_score"] / total_tasks if total_tasks > 0 else 0
            
            # Calculate task averages
            for task in scores[agent_name]["scores_by_task"]:
                task_scores = scores[agent_name]["scores_by_task"][task]["scores"]
                task_total = len(task_scores)
                scores[agent_name]["scores_by_task"][task]["average"] = scores[agent_name]["scores_by_task"][task]["total"] / task_total if task_total > 0 else 0
            
            # Calculate capability averages
            for capability in scores[agent_name]["scores_by_capability"]:
                capability_scores = scores[agent_name]["scores_by_capability"][capability]["scores"]
                capability_total = len(capability_scores)
                scores[agent_name]["scores_by_capability"][capability]["average"] = scores[agent_name]["scores_by_capability"][capability]["total"] / capability_total if capability_total > 0 else 0
    
    return scores

# Print summary of results
def print_summary(scores):
    print("\n===== BENCHMARK RESULTS =====\n")
    
    # Print overall scores
    print("Overall Average Scores:")
    print("----------------------")
    for agent_name, score_data in scores.items():
        print(f"{agent_name}: {score_data['average_score']:.2f}/5.00")
    
    print("\nScores by Capability:")
    print("-------------------")
    for agent_name, score_data in scores.items():
        print(f"\n{agent_name}:")
        for capability, capability_data in score_data["scores_by_capability"].items():
            print(f"  {capability}: {capability_data['average']:.2f}/5.00")
    
    print("\nScores by Task:")
    print("-------------")
    for agent_name, score_data in scores.items():
        print(f"\n{agent_name}:")
        for task, task_data in score_data["scores_by_task"].items():
            print(f"  {task}: {task_data['average']:.2f}/5.00")
    
    print("\n============================\n")

# Main function
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
    
    # Combine results
    results = combine_results(test_inputs, agent_responses)
    
    # Save results
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {result_path}")
    
    # Evaluate responses
    evaluations = evaluate_responses(data, results, args.start_idx)
    
    # Save evaluations
    with open(evaluation_path, 'w') as f:
        json.dump(evaluations, f, indent=2)
    
    print(f"Evaluations saved to {evaluation_path}")
    
    # Calculate scores
    scores = calculate_scores(evaluations)
    
    # Save scores
    with open(score_path, 'w') as f:
        json.dump(scores, f, indent=2)
    
    print(f"Scores saved to {score_path}")
    
    # Print summary
    print_summary(scores)

if __name__ == "__main__":
    main()
