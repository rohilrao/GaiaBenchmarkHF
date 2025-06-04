import asyncio
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.teams.magentic_one import MagenticOne
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console

# Configuration for Ollama
async def main(task=None):
    """
    Magentic-One setup with Ollama - similar to your AutoGen configuration
    
    Args:
        task (str): The task to execute. If None, uses a default task.
    
    Architecture:
    - Orchestrator: Plans and coordinates other agents
    - WebSurfer: Handles web browsing and interaction
    - FileSurfer: Manages file operations
    - Coder: Writes and analyzes code
    - ComputerTerminal: Executes code and installs libraries
    """
    
    # Default task if none provided
    if task is None:
        task = "Plot a chart of NVDA and TESLA stock price change YTD."
    
    model_client = None
    magentic_one = None
    
    try:
        # Create Ollama client - similar to your AutoGen config_list
        model_client = OllamaChatCompletionClient(
            model="llama3.2:3b",  # Use the same model as your AutoGen setup
            host="http://localhost:11434",
            # Optional configurations
            # temperature=0.7,
            # max_tokens=2048,
        )
        
        # Method 1: Use the MagenticOne helper class (simplest approach)
        print("Starting Magentic-One with Ollama...")
        
        magentic_one = MagenticOne(client=model_client)
        
        print(f"Task: {task}")
        print("Magentic-One is working...")
        
        # Execute the task with console output
        result = await Console(magentic_one.run_stream(task=task))
        
        print("\nTask completed!")
        print("Result:", result)
        
        return result
        
    except Exception as e:
        print(f"Error occurred: {e}")
        raise
    finally:
        # Proper cleanup
        if model_client:
            try:
                await model_client.close()
            except Exception as e:
                print(f"Error closing model client: {e}")
        
        # Additional cleanup for any running processes
        if magentic_one:
            try:
                # If there are any cleanup methods, call them here
                pass
            except Exception as e:
                print(f"Error during magentic_one cleanup: {e}")

# Alternative method using individual agents
async def alternative_setup():
    """
    Alternative setup using individual agents for more control
    """
    from autogen_ext.agents.web_surfer import MultimodalWebSurfer
    from autogen_ext.agents.file_surfer import FileSurfer
    from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
    from autogen_agentchat.agents import CodeExecutorAgent
    from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
    
    # Create Ollama client
    model_client = OllamaChatCompletionClient(
        model="llama3.2:3b",
        host="http://localhost:11434",
    )
    
    # Create individual agents
    web_surfer = MultimodalWebSurfer(
        "WebSurfer",
        model_client=model_client,
    )
    
    file_surfer = FileSurfer(
        "FileSurfer",
        model_client=model_client,
    )
    
    coder = MagenticOneCoderAgent(
        "Coder",
        model_client=model_client,
    )
    
    terminal = CodeExecutorAgent(
        "ComputerTerminal",
        code_executor=LocalCommandLineCodeExecutor(),
    )
    
    # Create team with all agents
    team = MagenticOneGroupChat(
        [web_surfer, file_surfer, coder, terminal], 
        model_client=model_client
    )
    
    # Execute task
    task = "Plot a chart of NVDA and TESLA stock price change YTD."
    await Console(team.run_stream(task=task))

# Enhanced configuration for better stock analysis
async def enhanced_setup():
    """
    Enhanced setup with better configuration for stock analysis
    """
    # Use a larger model for better reasoning (if available)
    model_client = OllamaChatCompletionClient(
        model="llama3.1:8b",  # Better for complex analysis
        host="http://localhost:11434",
        # Configuration for better performance
        options={
            "temperature": 0.3,      # Lower for more consistent analysis
            "top_p": 0.9,
            "num_ctx": 4096,         # Larger context window
        }
    )
    
    magentic_one = MagenticOne(
        client=model_client,
        # You can add custom system messages or configurations here
    )
    
    # More detailed task with specific requirements
    task = """
    Create a comprehensive stock analysis comparing NVDA and TESLA year-to-date performance:
    1. Fetch current stock data for both companies
    2. Calculate YTD returns and key metrics
    3. Create visualizations showing price trends
    4. Provide analysis of performance drivers
    5. Save results to a file
    """
    
    result = await Console(magentic_one.run_stream(task=task))
    return result

async def cleanup_main(task=None):
    """Main function with proper cleanup to avoid subprocess errors"""
    try:
        result = await main(task)
        return result
    finally:
        # Ensure all tasks are properly cleaned up
        pending = asyncio.all_tasks()
        for task_obj in pending:
            if not task_obj.done():
                task_obj.cancel()
        
        # Wait for all tasks to complete cancellation
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

if __name__ == "__main__":
    # Define your task here
    my_task = "Say Hello"
    
    # Example: Different tasks you can try
    # my_task = "Analyze the current weather in New York and create a summary report."
    # my_task = "Find the latest news about artificial intelligence and summarize the top 3 stories."
    # my_task = "Create a Python script to calculate compound interest and run an example."
    # my_task = "Research the top 5 programming languages in 2024 and create a comparison table."
    
    # Method 1: Use asyncio.run with proper cleanup (recommended)
    try:
        result = asyncio.run(cleanup_main(my_task))
        print(f"\nFinal result: {result}")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    
    # Method 2: Alternative - pass task directly to main
    # asyncio.run(main("Your custom task here"))
    
    # Method 3: Interactive task input
    # user_task = input("Enter your task: ")
    # asyncio.run(cleanup_main(user_task))