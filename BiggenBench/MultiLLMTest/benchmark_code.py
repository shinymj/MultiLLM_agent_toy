import json
import os
import time
import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load environment variables for API keys
load_dotenv()

# Timestamp for file naming
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# Base paths
DATA_PATH = "biggen_bench_instruction_idx0.json"
RESULT_PATH = f"biggen_bench_instruction_idx0_result_{timestamp}.json"
EVALUATION_PATH = f"biggen_bench_instruction_idx0_evaluation_{timestamp}.json"
SCORE_PATH = f"biggen_bench_instruction_idx0_score_{timestamp}.json"

# Utility function to setup local models
def setup_local_model(model_name: str, model_path: str):
    print(f"Loading {model_name} from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )
    
    # Create text generation pipeline
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=2048,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.15
    )
    
    # Convert to LangChain LLM
    llm = HuggingFacePipeline(pipeline=text_pipeline)
    return llm

# Setup LLM agents
def setup_agents():
    agents = {}
    
    # Local models
    llama_path = "path/to/llama3.1"  # Update with correct local path
    deepseek_path = "path/to/deepseek-r1-8b"  # Update with correct local path
    gemma_path = "path/to/gemma3-12b"  # Update with correct local path
    
    # Setup local models
    agents["llama3.1"] = setup_local_model("llama3.1", llama_path)
    agents["deepseek-r1:8b"] = setup_local_model("deepseek-r1:8b", deepseek_path)
    agents["gemma3:12b"] = setup_local_model("gemma3:12b", gemma_path)
    
    # Setup API models
    agents["gpt-4o"] = ChatOpenAI(
        model_name="gpt-4o", 
        temperature=0.1,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    agents["claude-sonnet-3-5"] = ChatAnthropic(
        model_name="claude-3-5-sonnet-20240620", 
        temperature=0.1,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    return agents

# Extract test inputs from the benchmark data
def extract_test_inputs(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    test_inputs = []
    for instance in data:
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
    
    for test_input in test_inputs:
        print(f"Processing test input: {test_input['id']}")
        
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
            for attempt in range(max_retries):
                try:
                    response = chain.invoke({})
                    break
                except Exception as e:
                    print(f"    Error with {agent_name}: {e}")
                    if attempt < max_retries - 1:
                        print(f"    Retrying... ({attempt + 1}/{max_retries})")
                        time.sleep(5)
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
            result["agent_responses"][agent_name] = agent_responses[agent_name][i]["response"]
        
        results.append(result)
    
    return results

# Evaluate agent responses
def evaluate_responses(original_data: List[Dict[str, Any]], results: List[Dict[str, Any]], agents: Dict[str, Any]) -> List[Dict[str, Any]]:
    evaluations = []
    
    # Use Claude for evaluation
    evaluator = ChatAnthropic(
        model_name="claude-3-5-sonnet-20240620",
        temperature=0,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    for i, result in enumerate(results):
        original_item = original_data[i]
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
            for attempt in range(max_retries):
                try:
                    evaluation_result = chain.invoke({})
                    evaluation_json = json.loads(evaluation_result)
                    break
                except Exception as e:
                    print(f"  Error evaluating {agent_name}: {e}")
                    if attempt < max_retries - 1:
                        print(f"  Retrying... ({attempt + 1}/{max_retries})")
                        time.sleep(5)
                    else:
                        evaluation_json = {
                            "score": 0,
                            "rationale": f"ERROR: Failed to evaluate after {max_retries} attempts"
                        }
            
            eval_item["evaluations"][agent_name] = evaluation_json
        
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
                "scores_by_task": {}
            }
        
        # Calculate scores
        for evaluation in evaluations:
            task = evaluation["task"]
            
            # Initialize task scores if needed
            for agent_name in agent_names:
                if task not in scores[agent_name]["scores_by_task"]:
                    scores[agent_name]["scores_by_task"][task] = {
                        "scores": [],
                        "total": 0,
                        "average": 0
                    }
            
            # Add scores
            for agent_name, agent_eval in evaluation["evaluations"].items():
                score = agent_eval["score"]
                scores[agent_name]["total_score"] += score
                scores[agent_name]["scores_by_task"][task]["scores"].append(score)
                scores[agent_name]["scores_by_task"][task]["total"] += score
        
        # Calculate averages
        for agent_name in agent_names:
            total_tasks = len(evaluations)
            scores[agent_name]["average_score"] = scores[agent_name]["total_score"] / total_tasks if total_tasks > 0 else 0
            
            for task in scores[agent_name]["scores_by_task"]:
                task_scores = scores[agent_name]["scores_by_task"][task]["scores"]
                task_total = len(task_scores)
                scores[agent_name]["scores_by_task"][task]["average"] = scores[agent_name]["scores_by_task"][task]["total"] / task_total if task_total > 0 else 0
    
    return scores

# Main function
def main():
    # Load the benchmark data
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    
    # Extract test inputs
    test_inputs = extract_test_inputs(data)
    
    # Setup agents
    agents = setup_agents()
    
    # Get responses from agents
    agent_responses = get_agent_responses(agents, test_inputs)
    
    # Combine results
    results = combine_results(test_inputs, agent_responses)
    
    # Save results
    with open(RESULT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {RESULT_PATH}")
    
    # Evaluate responses
    evaluations = evaluate_responses(data, results, agents)
    
    # Save evaluations
    with open(EVALUATION_PATH, 'w') as f:
        json.dump(evaluations, f, indent=2)
    
    print(f"Evaluations saved to {EVALUATION_PATH}")
    
    # Calculate scores
    scores = calculate_scores(evaluations)
    
    # Save scores
    with open(SCORE_PATH, 'w') as f:
        json.dump(scores, f, indent=2)
    
    print(f"Scores saved to {SCORE_PATH}")

if __name__ == "__main__":
    main()
