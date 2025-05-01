import json
import os
import time
from datetime import datetime
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_community.llms import Ollama
from langchain_anthropic import ChatAnthropic

# Load environment variables from .env file
load_dotenv()


def load_benchmark_data(file_path: str) -> List[Dict]:
    """Load benchmark data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def prepare_prompts(benchmark_data: List[Dict]) -> List[Dict]:
    """Prepare prompts by excluding reference_answer and score_rubric."""
    prompts = []
    for item in benchmark_data:
        prompt = {
            "id": item["id"],
            "capability": item["capability"],
            "task": item["task"],
            "instance_idx": item["instance_idx"],
            "system_prompt": item["system_prompt"],
            "input": item["input"],
            # Exclude reference_answer and score_rubric
        }
        prompts.append(prompt)
    return prompts


def get_agent_responses(prompts: List[Dict], agent_config: Dict) -> List[Dict]:
    """Get responses from an agent for each prompt."""
    responses = []

    # Initialize agent based on configuration
    if agent_config["type"] == "ollama":
        agent = Ollama(model=agent_config["model"])
    else:
        raise ValueError(f"Unsupported agent type: {agent_config['type']}")

    print(f"Getting responses from {agent_config['name']} ({agent_config['model']})...")
    
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}: {prompt['id']}")
        
        # Format the prompt according to the agent's expected format
        formatted_prompt = f"System: {prompt['system_prompt']}\n\nUser: {prompt['input']}"
        
        try:
            # Get response from the agent
            response = agent.invoke(formatted_prompt)
            
            # Add response to the list
            responses.append({
                "id": prompt["id"],
                "agent_name": agent_config["name"],
                "response": response
            })
            
            # Small delay to prevent rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"Error getting response for prompt {prompt['id']}: {e}")
            responses.append({
                "id": prompt["id"],
                "agent_name": agent_config["name"],
                "response": f"ERROR: {str(e)}"
            })
    
    return responses


def evaluate_responses(benchmark_data: List[Dict], agent_responses: List[Dict], evaluator_config: Dict) -> List[Dict]:
    """Evaluate agent responses using Claude."""
    evaluations = []
    
    # Initialize evaluator
    evaluator = ChatAnthropic(
        model=evaluator_config["model"],
        anthropic_api_key=evaluator_config["api_key"]
    )
    
    print(f"Evaluating responses using {evaluator_config['name']}...")
    
    # Create a mapping from prompt ID to benchmark data for easy lookup
    benchmark_map = {item["id"]: item for item in benchmark_data}
    
    # Group responses by prompt ID
    response_map = {}
    for response in agent_responses:
        if response["id"] not in response_map:
            response_map[response["id"]] = []
        response_map[response["id"]].append(response)
    
    for prompt_id, responses in response_map.items():
        benchmark_item = benchmark_map[prompt_id]
        
        # For each prompt, evaluate all agent responses
        for response in responses:
            print(f"Evaluating {response['agent_name']}'s response for {prompt_id}")
            
            evaluation_prompt = f"""
You are an evaluation expert who needs to rate an AI assistant's response to a user query.

Task ID: {prompt_id}
Capability: {benchmark_item['capability']}
Task Type: {benchmark_item['task']}

The user query was:
{benchmark_item['input']}

The reference answer is:
{benchmark_item['reference_answer']}

The AI assistant's response was:
{response['response']}

Evaluation criteria:
{benchmark_item['score_rubric']['criteria']}

Score 1 means: {benchmark_item['score_rubric']['score1_description']}
Score 2 means: {benchmark_item['score_rubric']['score2_description']}
Score 3 means: {benchmark_item['score_rubric']['score3_description']}
Score 4 means: {benchmark_item['score_rubric']['score4_description']}
Score 5 means: {benchmark_item['score_rubric']['score5_description']}

Please evaluate the AI assistant's response on a scale from 1 to 5, where 1 is the worst and 5 is the best.
Provide your score and a detailed rationale for your scoring.

Your evaluation should be in this format:
Score: [Your score between 1-5]
Rationale: [Your detailed explanation of why you gave this score]
"""
            
            try:
                # Get evaluation from Claude
                evaluation_response = evaluator.invoke(evaluation_prompt)
                
                # Extract score and rationale from evaluation response
                score_line = [line for line in evaluation_response.content[0].text.split('\n') if line.startswith('Score:')]
                rationale_lines = []
                capture_rationale = False
                
                for line in evaluation_response.content[0].text.split('\n'):
                    if line.startswith('Rationale:'):
                        capture_rationale = True
                        rationale_lines.append(line.replace('Rationale:', '').strip())
                    elif capture_rationale:
                        rationale_lines.append(line)
                
                if score_line:
                    try:
                        score = int(score_line[0].replace('Score:', '').strip())
                    except ValueError:
                        score = None
                else:
                    score = None
                
                rationale = ' '.join(rationale_lines)
                
                evaluations.append({
                    "id": prompt_id,
                    "agent_name": response["agent_name"],
                    "score": score,
                    "rationale": rationale,
                    "full_evaluation": evaluation_response.content[0].text
                })
                
                # Small delay to prevent rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"Error evaluating response for prompt {prompt_id}: {e}")
                evaluations.append({
                    "id": prompt_id,
                    "agent_name": response["agent_name"],
                    "score": None,
                    "rationale": f"ERROR: {str(e)}",
                    "full_evaluation": f"ERROR: {str(e)}"
                })
    
    return evaluations


def save_results(results: List[Dict], benchmark_id: str, result_type: str) -> str:
    """Save results to a JSON file with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{benchmark_id}_{result_type}_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {result_type} to {filename}")
    return filename


def main():
    parser = argparse.ArgumentParser(description="Benchmark AI agents on instruction following tasks")
    parser.add_argument("--benchmark", type=str, default="biggen_bench_instruction_idx0.json",
                        help="Path to benchmark JSON file")
    parser.add_argument("--anthropic_api_key", type=str, default=os.getenv("ANTHROPIC_API_KEY"),
                        help="Anthropic API key for Claude 3.7 (defaults to ANTHROPIC_API_KEY from .env)")
    parser.add_argument("--agent1", type=str, default="llama3.1",
                        help="First agent model name for Ollama")
    parser.add_argument("--agent2", type=str, default="gemma3:12b",
                        help="Second agent model name for Ollama")
    args = parser.parse_args()
    
    # Ensure we have an API key
    if not args.anthropic_api_key:
        raise ValueError("Anthropic API key is required. Either provide it via --anthropic_api_key or add ANTHROPIC_API_KEY to your .env file.")
    
    # Load benchmark data
    benchmark_data = load_benchmark_data(args.benchmark)
    benchmark_id = os.path.splitext(os.path.basename(args.benchmark))[0]
    
    # Prepare prompts
    prompts = prepare_prompts(benchmark_data)
    
    # Configure agents
    agent1_config = {
        "name": "Agent1",
        "type": "ollama",
        "model": args.agent1
    }
    
    agent2_config = {
        "name": "Agent2",
        "type": "ollama",
        "model": args.agent2
    }
    
    # Configure evaluator
    evaluator_config = {
        "name": "Claude 3.7",
        "model": "claude-3-7-sonnet-20250219",
        "api_key": args.anthropic_api_key
    }
    
    # Get responses from both agents
    agent1_responses = get_agent_responses(prompts, agent1_config)
    agent2_responses = get_agent_responses(prompts, agent2_config)
    all_responses = agent1_responses + agent2_responses
    
    # Save agent responses
    responses_file = save_results(all_responses, benchmark_id, "result")
    
    # Evaluate responses
    evaluations = evaluate_responses(benchmark_data, all_responses, evaluator_config)
    
    # Save evaluations
    evaluations_file = save_results(evaluations, benchmark_id, "evaluation")
    
    # Print summary
    print("\nEvaluation Summary:")
    agent1_scores = [eval["score"] for eval in evaluations if eval["agent_name"] == "Agent1" and eval["score"] is not None]
    agent2_scores = [eval["score"] for eval in evaluations if eval["agent_name"] == "Agent2" and eval["score"] is not None]
    
    if agent1_scores:
        print(f"Agent1 ({args.agent1}) average score: {sum(agent1_scores) / len(agent1_scores):.2f}")
    else:
        print(f"Agent1 ({args.agent1}): No valid scores")
    
    if agent2_scores:
        print(f"Agent2 ({args.agent2}) average score: {sum(agent2_scores) / len(agent2_scores):.2f}")
    else:
        print(f"Agent2 ({args.agent2}): No valid scores")


if __name__ == "__main__":
    main()