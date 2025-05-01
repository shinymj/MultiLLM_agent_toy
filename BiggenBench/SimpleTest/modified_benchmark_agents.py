import json
import os
import time
from datetime import datetime
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_ollama import OllamaLLM
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


def get_agent_responses_interleaved(prompts: List[Dict], agent_configs: List[Dict]) -> List[Dict]:
    """Get responses from multiple agents for each prompt, processing one prompt at a time."""
    responses = []

    # Initialize all agents based on configurations
    agents = {}
    for config in agent_configs:
        if config["type"] == "ollama":
            agents[config["name"]] = {
                "agent": OllamaLLM(model=config["model"]),
                "model": config["model"]
            }
        else:
            raise ValueError(f"Unsupported agent type: {config['type']}")

    print(f"Getting responses from {len(agents)} agents...")
    
    # Process one prompt at a time for all agents
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}: {prompt['id']}")
        
        # Format the prompt according to the agent's expected format
        formatted_prompt = f"System: {prompt['system_prompt']}\n\nUser: {prompt['input']}"
        
        # Get responses from each agent for this prompt
        for agent_name, agent_data in agents.items():
            print(f"  Getting response from {agent_name} ({agent_data['model']})...")
            
            try:
                # Get response from the agent
                response = agent_data["agent"].invoke(formatted_prompt)
                
                # Add response to the list
                responses.append({
                    "id": prompt["id"],
                    "agent_name": agent_name,
                    "response": response
                })
                
                # Small delay to prevent rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"  Error getting response for prompt {prompt['id']} from {agent_name}: {e}")
                responses.append({
                    "id": prompt["id"],
                    "agent_name": agent_name,
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
Reference the score descriptions above to assign the most appropriate score.

Your evaluation should be in this format:
Score: [Your score between 1-5]
Rationale: [Briefly explain which aspects of the score description the response most closely matches. Keep your explanation concise.]
"""
            
            try:
                # Get evaluation from Claude
                evaluation_response = evaluator.invoke(evaluation_prompt)
                
                # Get the evaluation text from the response
                # ChatAnthropic returns a message object with content
                evaluation_text = ""
                if hasattr(evaluation_response, 'content'):
                    # Handle new ChatAnthropic response format
                    if isinstance(evaluation_response.content, list):
                        for content_item in evaluation_response.content:
                            if hasattr(content_item, 'text'):
                                evaluation_text += content_item.text
                            elif isinstance(content_item, dict) and 'text' in content_item:
                                evaluation_text += content_item['text']
                    elif isinstance(evaluation_response.content, str):
                        evaluation_text = evaluation_response.content
                elif hasattr(evaluation_response, 'text'):
                    # Handle possible direct text property
                    evaluation_text = evaluation_response.text
                elif isinstance(evaluation_response, str):
                    # Handle if the response is directly a string
                    evaluation_text = evaluation_response
                else:
                    # Try to convert the entire response to a string as fallback
                    evaluation_text = str(evaluation_response)
                
                # Extract score and rationale from evaluation text
                score = None
                rationale = ""
                
                # Look for score in the evaluation text
                for line in evaluation_text.split('\n'):
                    if line.lower().startswith('score:'):
                        # Extract the score, handling potential unexpected formats
                        try:
                            score_str = line.replace('Score:', '').strip()
                            # Extract just the first number in case there are additional characters
                            import re
                            score_match = re.search(r'\d+', score_str)
                            if score_match:
                                score = int(score_match.group())
                        except ValueError:
                            score = None
                
                # Extract rationale - everything after "Rationale:" or similar
                rationale_start = -1
                for i, line in enumerate(evaluation_text.split('\n')):
                    if 'rationale:' in line.lower():
                        rationale_start = i
                        # Remove "Rationale:" prefix from the first line
                        rationale = line.split(':', 1)[1].strip() if ':' in line else line
                        break
                
                # If we found a rationale starting point, collect all subsequent lines
                if rationale_start >= 0:
                    rationale_lines = evaluation_text.split('\n')[rationale_start+1:]
                    rationale += ' ' + ' '.join(rationale_lines).strip()
                else:
                    # If no explicit rationale marker, use everything after the score
                    score_position = evaluation_text.lower().find('score:')
                    if score_position >= 0:
                        # Find the end of the score line
                        score_line_end = evaluation_text.find('\n', score_position)
                        if score_line_end >= 0:
                            rationale = evaluation_text[score_line_end:].strip()
                
                evaluations.append({
                    "id": prompt_id,
                    "agent_name": response["agent_name"],
                    "agent_response": response["response"],
                    "reference_answer": benchmark_item['reference_answer'],
                    "score": score,
                    "rationale": rationale
                })
                
                # Small delay to prevent rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"Error evaluating response for prompt {prompt_id}: {e}")
                evaluations.append({
                    "id": prompt_id,
                    "agent_name": response["agent_name"],
                    "agent_response": response["response"],
                    "reference_answer": benchmark_item['reference_answer'],
                    "score": None,
                    "rationale": f"ERROR: {str(e)}"
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
    
    # Get responses from both agents (interleaved)
    all_responses = get_agent_responses_interleaved(prompts, [agent1_config, agent2_config])
    
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