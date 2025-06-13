from smolagents import CodeAgent, LiteLLMModel, DuckDuckGoSearchTool, VisitWebpageTool, FinalAnswerTool, Tool, tool
import json
import os
import time
import sys
import datetime
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

def load_gaia_questions(dataset_folder="dataset"):
    """Load the GAIA questions from the downloaded JSON file."""
    questions_file = os.path.join(dataset_folder, "gaia_questions.json")
    
    if not os.path.exists(questions_file):
        raise FileNotFoundError(f"Questions file not found at {questions_file}. Make sure you've downloaded the dataset.")
    
    with open(questions_file, 'r') as f:
        questions = json.load(f)
    
    print(f"Loaded {len(questions)} questions from GAIA dataset")
    return questions

def process_question_with_files(agent, question_data, dataset_folder="dataset"):
    """Process a single question, handling any associated files."""
    task_id = question_data.get("task_id")
    question_text = question_data.get("question")
    
    # Check if there are any files associated with this task
    task_file_path = os.path.join(dataset_folder, f"task_{task_id}_file")
    
    # Prepare the question with file information if available
    enhanced_question = question_text
    
    if os.path.exists(task_file_path):
        # Read the file and add it to the question context
        try:
            # Try to read as text first
            with open(task_file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            enhanced_question = f"""Question: {question_text}

Associated file content:
{file_content}

Please analyze the file content and answer the question."""
        except UnicodeDecodeError:
            # If it's a binary file, just mention it exists
            enhanced_question = f"""Question: {question_text}

Note: There is an associated file for this task (task_{task_id}_file) that may be relevant to answering this question. The file appears to be binary data."""
    
    return enhanced_question

def run_agent_on_dataset(dataset_folder="dataset", save_results=True, max_questions=None, output_manager=None):
    """Run the agent on all questions in the GAIA dataset."""
    
    # Load questions
    questions = load_gaia_questions(dataset_folder)
    
    # Limit questions if specified
    if max_questions:
        questions = questions[:max_questions]
        print(f"Processing first {max_questions} questions only")
    
    # Initialize agent
    agent = get_agent()
    
    # Store results
    results = []
    
    print(f"Starting to process {len(questions)} questions...")
    
    for i, question_data in enumerate(tqdm(questions, desc="Processing questions")):
        task_id = question_data.get("task_id")
        question_text = question_data.get("question")
        
        print(f"\n--- Question {i+1}/{len(questions)} (Task ID: {task_id}) ---")
        print(f"Question: {question_text[:100]}...")
        
        try:
            # Process question with any associated files
            enhanced_question = process_question_with_files(agent, question_data, dataset_folder)
            
            # Run the agent
            start_time = time.time()
            answer = agent.run(enhanced_question)
            end_time = time.time()
            
            # Store result
            result = {
                "task_id": task_id,
                "question": question_text,
                "submitted_answer": answer,
                "processing_time": end_time - start_time,
                "status": "success"
            }
            
            print(f"Answer: {answer}")
            print(f"Time taken: {result['processing_time']:.2f} seconds")
            
        except Exception as e:
            print(f"Error processing question {task_id}: {str(e)}")
            result = {
                "task_id": task_id,
                "question": question_text,
                "submitted_answer": f"ERROR: {str(e)}",
                "processing_time": 0,
                "status": "error"
            }
        
        results.append(result)
        
        # Optional: Add delay between questions to avoid overwhelming the model
        time.sleep(1)
    
    # Save results if requested
    if save_results:
        # Use experiment folder if output_manager is provided, otherwise use dataset folder
        if output_manager and hasattr(output_manager, 'log_dir'):
            results_file = os.path.join(output_manager.log_dir, "agent_results.json")
            answers_file = os.path.join(output_manager.log_dir, "answers_for_submission.json")
        else:
            results_file = os.path.join(dataset_folder, "agent_results.json")
            answers_file = os.path.join(dataset_folder, "answers_for_submission.json")
        
        # Save detailed results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {results_file}")
        
        # Save answers in submission format (compatible with the API)
        submission_answers = [
            {"task_id": r["task_id"], "submitted_answer": r["submitted_answer"]} 
            for r in results if r["status"] == "success"
        ]
        with open(answers_file, 'w') as f:
            json.dump(submission_answers, f, indent=2)
        print(f"Submission-ready answers saved to {answers_file}")
    
    # Print summary
    successful = sum(1 for r in results if r["status"] == "success")
    print(f"\n=== SUMMARY ===")
    print(f"Total questions: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Errors: {len(results) - successful}")
    print(f"Success rate: {successful/len(results)*100:.1f}%")
    
    return results

def run_single_question(question_index=0, dataset_folder="dataset"):
    """Run the agent on a single question for testing."""
    questions = load_gaia_questions(dataset_folder)
    
    if question_index >= len(questions):
        print(f"Question index {question_index} is out of range. Dataset has {len(questions)} questions.")
        return
    
    question_data = questions[question_index]
    agent = get_agent()
    
    task_id = question_data.get("task_id")
    question_text = question_data.get("question")
    
    print(f"Running single question (Task ID: {task_id})")
    print(f"Question: {question_text}")
    
    enhanced_question = process_question_with_files(agent, question_data, dataset_folder)
    
    try:
        answer = agent.run(enhanced_question)
        print(f"Answer: {answer}")
        return answer
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

class OutputManager:
    """Manages output redirection to log files with optional console output."""
    def __init__(self, log_dir="logs", experiment_name=None, console_output=True):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create experiment-specific folder if experiment name provided
        if experiment_name:
            # Clean experiment name for folder creation
            clean_name = "".join(c for c in experiment_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            clean_name = clean_name.replace(' ', '_')
            experiment_folder = f"{timestamp}_{clean_name}"
            self.log_dir = os.path.join(log_dir, experiment_folder)
        else:
            self.log_dir = log_dir
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.log_dir, f"agent_run.log")
        self.error_file = os.path.join(self.log_dir, f"agent_errors.log")
        self.config_file = os.path.join(self.log_dir, f"run_config.json")
        
        self.log_handle = open(self.log_file, 'w', encoding='utf-8')
        self.error_handle = open(self.error_file, 'w', encoding='utf-8')
        
        self.console_output = console_output
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Save run configuration
        self._save_run_config(experiment_name, timestamp)
        
        # Redirect output
        sys.stdout = self
        sys.stderr = self.ErrorRedirect(self.error_handle, self.original_stderr if console_output else None)
        
        print(f"=== Agent Run Started at {datetime.datetime.now()} ===")
        if experiment_name:
            print(f"Experiment: {experiment_name}")
        print(f"Run folder: {self.log_dir}")
        print(f"Log file: {self.log_file}")
        print(f"Error file: {self.error_file}")
        print("=" * 60)
    
    def _save_run_config(self, experiment_name, timestamp):
        """Save configuration details for this run."""
        config = {
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "start_time": datetime.datetime.now().isoformat(),
            "log_directory": self.log_dir,
            "python_version": sys.version,
            "working_directory": os.getcwd()
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def write(self, message):
        self.log_handle.write(message)
        self.log_handle.flush()
        if self.console_output:
            self.original_stdout.write(message)
            self.original_stdout.flush()
    
    def flush(self):
        self.log_handle.flush()
        if self.console_output:
            self.original_stdout.flush()
    
    class ErrorRedirect:
        def __init__(self, file_handle, console_handle=None):
            self.file_handle = file_handle
            self.console_handle = console_handle
        
        def write(self, message):
            self.file_handle.write(message)
            self.file_handle.flush()
            if self.console_handle:
                self.console_handle.write(message)
                self.console_handle.flush()
        
        def flush(self):
            self.file_handle.flush()
            if self.console_handle:
                self.console_handle.flush()
    
    def close(self):
        print("=" * 60)
        print(f"=== Agent Run Ended at {datetime.datetime.now()} ===")
        
        # Update config with end time
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            config["end_time"] = datetime.datetime.now().isoformat()
            config["duration"] = str(datetime.datetime.now() - datetime.datetime.fromisoformat(config["start_time"]))
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except:
            pass  # Don't fail if config update fails
        
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.log_handle.close()
        self.error_handle.close()
        print(f"Run folder: {self.log_dir}")
        print(f"Logs saved to: {self.log_file}")
        print(f"Errors saved to: {self.error_file}")

# Example usage
if __name__ == "__main__":
    # Option 1: Run with experiment name (creates organized folders)
    output_manager = OutputManager(
        experiment_name="Test2Results - Naive Baseline",    
        console_output=False
    )
    
    # Option 2: Run with specific experiment name
    # output_manager = OutputManager(
    #     experiment_name="Baseline Evaluation", 
    #     console_output=True
    # )
    
    # Option 3: Run without experiment name (old behavior)
    # output_manager = OutputManager(console_output=False)
    
    try:
        # Test with a single question first
        # print("Testing with first question...")
        # run_single_question(0)
        
        # Uncomment to run on multiple questions (this will take time!)
        #print("\nRunning on first 2 questions...")
        #results = run_agent_on_dataset(max_questions=2, dataset_folder="../dataset", output_manager=output_manager)
        
        # Uncomment to run on all questions (this will take a long time!)
        print("\nRunning on all questions...")
        results = run_agent_on_dataset(dataset_folder="../dataset", output_manager=output_manager)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Always clean up
        output_manager.close()