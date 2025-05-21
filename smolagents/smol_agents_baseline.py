import os
from datasets import load_dataset
from tqdm import tqdm

def load_gaia_datasets(levels=None, loader_path="../GAIA.py", split="validation"):
    """
    Load GAIA datasets for spcified levels as a dictionary.
    
    Args:
        levels: List of dataset levels to load (if None, loads all available levels)
        loader_path: Path to the GAIA loader script
        split: Dataset split to use
        
    Returns:
        Dictionary with level names as keys and dataset examples as values
    """
    # Default levels if none specified
    if levels is None:
        levels = ["2023_level1", "2023_level2", "2023_level3", "2023_level4", "2023_level5"]
        
    result = {}
    
    for level in levels:
        try:
            print(f"Loading {level}...")
            dataset = load_dataset(loader_path, name=level, split=split)
            
            # Convert to list of dictionaries with file content
            examples = []
            for idx, example in enumerate(dataset):
                item = dict(example)
                
                # Try to load file content if available
                file_content = ""
                file_path = item.get("file_path", "")
                file_name = item.get("file_name", "")
                
                if file_path and os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        file_content = f.read()
                elif file_name and os.path.exists(file_name):
                    with open(file_name, 'r', encoding='utf-8', errors='replace') as f:
                        file_content = f.read()
                
                item["file_content"] = file_content
                item["has_file_content"] = bool(file_content)
                examples.append(item)
                
            result[level] = examples
            print(f"Loaded {len(examples)} examples from {level}")
        except Exception as e:
            print(f"Error loading {level}: {e}")
    
    return result

from smolagents import CodeAgent, LiteLLMModel, DuckDuckGoSearchTool, VisitWebpageTool, FinalAnswerTool, Tool, tool
import json
import os
import time
from tqdm import tqdm

# Create a model using LiteLLMModel with Ollama
model = LiteLLMModel(
    model_id="ollama_chat/qwen2.5-coder:32b",  # Format: "ollama_chat/[model-name]"
    api_base="http://localhost:11434",   # Default Ollama API endpoint
    api_key="ollama",                    # This is just a placeholder, Ollama doesn't actually require an API key
    num_ctx=30000                        # Ollama default is 2048 which might be too small for complex tasks
)


# Create the CodeAgent with all tools
def get_agent():
    return CodeAgent(
        tools=[
            DuckDuckGoSearchTool(),
            VisitWebpageTool(),
            FinalAnswerTool()
        ],
        model=model,
        additional_authorized_imports=["wikipedia", "requests", "json", "re", "datetime", "os"]
    )


def answer_gaia_questions(datasets, output_dir="gaia_results"):
    """
    Process and answer all questions from the provided GAIA datasets
    
    Args:
        datasets: Dictionary with level names as keys and dataset examples as values
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a checkpoint file to track progress
    checkpoint_file = os.path.join(output_dir, "checkpoint.json")
    completed_tasks = {}
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                completed_tasks = json.load(f)
            print(f"Loaded checkpoint with {sum(len(tasks) for tasks in completed_tasks.values())} completed tasks")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting fresh.")
            completed_tasks = {}
    
    # Process each level
    for level_name, examples in datasets.items():
        print(f"\nProcessing {level_name} with {len(examples)} examples...")
        
        # Initialize level results from checkpoint or create new
        if level_name not in completed_tasks:
            completed_tasks[level_name] = {}
        
        # Load existing results if available
        level_results_file = os.path.join(output_dir, f"{level_name}_results.json")
        level_results = []
        if os.path.exists(level_results_file):
            try:
                with open(level_results_file, 'r') as f:
                    level_results = json.load(f)
                print(f"Loaded {len(level_results)} existing results for {level_name}")
            except Exception as e:
                print(f"Error loading existing results: {e}. Starting with empty results.")
                level_results = []
        
        # Create level-specific output directory
        level_dir = os.path.join(output_dir, level_name)
        os.makedirs(level_dir, exist_ok=True)
        
        # Process each example in the level
        for example in tqdm(examples):
            # Extract question information
            task_id = example.get("task_id", "unknown_id")
            
            # Skip if already completed
            if task_id in completed_tasks[level_name]:
                print(f"Skipping completed task {task_id}")
                continue
                
            question = example.get("Question", "")
            expected_answer = example.get("Final answer", "")
            
            # Handle file content if available
            file_content = ""
            file_path = example.get("file_path", "")
            file_name = example.get("file_name", "")
            
            # Try to load file content if available
            if file_path and os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        file_content = f.read()
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
            elif file_name and os.path.exists(file_name):
                try:
                    with open(file_name, 'r', encoding='utf-8', errors='replace') as f:
                        file_content = f.read()
                except Exception as e:
                    print(f"Error reading file {file_name}: {e}")
            
            # Use file_content from example if already loaded
            if not file_content and "file_content" in example and example["file_content"]:
                file_content = example["file_content"]
            
            # Skip if question is empty
            if not question:
                print(f"Skipping example {task_id}: Question is empty")
                continue
            
            # Print detailed information about the task
            print(f"\n{'='*80}")
            print(f"Processing task {task_id}:")
            print(f"Question: {question}")
            print(f"Expected answer: {expected_answer}")
            print(f"Has file content: {bool(file_content)}")
            print(f"{'='*80}")
            
            # Construct the prompt
            prompt = ""
            if file_content:
                prompt = f"Here is the file content to use for answering the question:\n\n{file_content}\n\n"
            
            prompt += f"Question: {question}\n\n"
            # Reiterate the question at the end of the prompt
            prompt += f"Please answer the question: {question}"
            
            # Initialize a new agent for each question to avoid context contamination
            agent = get_agent()
            
            try:
                # Run the agent and get the answer
                start_time = time.time()
                result = agent.run(prompt)
                end_time = time.time()
                
                # Print the agent's answer
                print(f"Question: {question}")
                print(f"Expected answer: {expected_answer}")
                print(f"Agent's answer: {result}")
                print(f"Processing time: {end_time - start_time:.2f} seconds")
                
                # Store the result
                question_result = {
                    "task_id": task_id,
                    "question": question,
                    "level": example.get("Level", ""),
                    "has_file_content": bool(file_content),
                    "model_answer": result,
                    "expected_answer": expected_answer,
                    "processing_time": end_time - start_time
                }
                
                # Save individual result
                result_file = os.path.join(level_dir, f"{task_id}.json")
                with open(result_file, "w") as f:
                    json.dump(question_result, f, indent=2)
                
                # Add to level results
                level_results.append(question_result)
                
                # Update checkpoint
                completed_tasks[level_name][task_id] = True
                with open(checkpoint_file, "w") as f:
                    json.dump(completed_tasks, f, indent=2)
                
                # Update level results file after each completion
                with open(level_results_file, "w") as f:
                    json.dump(level_results, f, indent=2)
                
                # Pause between questions to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing task {task_id}: {str(e)}")
                # Store the error
                question_result = {
                    "task_id": task_id,
                    "question": question,
                    "level": example.get("Level", ""),
                    "has_file_content": bool(file_content),
                    "error": str(e),
                    "expected_answer": expected_answer
                }
                
                # Save error result
                error_file = os.path.join(level_dir, f"{task_id}_error.json")
                with open(error_file, "w") as f:
                    json.dump(question_result, f, indent=2)
                
                level_results.append(question_result)
                
                # Update checkpoint and results file even for errors
                completed_tasks[level_name][task_id] = True
                with open(checkpoint_file, "w") as f:
                    json.dump(completed_tasks, f, indent=2)
                
                with open(level_results_file, "w") as f:
                    json.dump(level_results, f, indent=2)
                    
                # Pause between questions to avoid rate limiting
                time.sleep(1)

                
        print(f"Completed processing {level_name}. Results saved to {level_dir}")
    
    print(f"\nAll processing complete. Results saved to {output_dir}")

# Print a summary of results
def print_summary(output_dir="gaia_results"):
    """Print a summary of the processing results"""
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist. No results to summarize.")
        return
        
    total_questions = 0
    total_answered = 0
    total_errors = 0
    correct_answers = 0
    
    for level_name in ["2023_level1", "2023_level2", "2023_level3"]:
        result_file = os.path.join(output_dir, f"{level_name}_results.json")
        if not os.path.exists(result_file):
            print(f"No results file found for {level_name}")
            continue
            
        with open(result_file, "r") as f:
            results = json.load(f)
            
        questions = len(results)
        errors = sum(1 for r in results if "error" in r)
        answered = questions - errors
        
        # Count correct answers
        level_correct = 0
        for r in results:
            if "error" not in r and r.get("model_answer", "").strip() == r.get("expected_answer", "").strip():
                level_correct += 1
        
        print(f"{level_name}: {answered}/{questions} questions answered ({errors} errors), {level_correct} correct")
        
        total_questions += questions
        total_answered += answered
        total_errors += errors
        correct_answers += level_correct
    
    if total_questions > 0:
        print(f"\nOverall: {total_answered}/{total_questions} questions answered ({total_errors} errors)")
        if total_answered > 0:
            print(f"Accuracy: {correct_answers}/{total_answered} correct ({correct_answers/total_answered*100:.2f}%)")
    else:
        print("\nNo questions processed yet.")


if __name__ == "__main__":
    # Load datasets
    datasets = load_gaia_datasets(levels=["2023_level1", "2023_level2", "2023_level3"])
    
    # Answer questions and save results
    answer_gaia_questions(datasets, output_dir="gaia_results")

    # Print summary after completion
    print_summary()

