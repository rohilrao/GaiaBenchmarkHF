"""
Simple Query Assistant with Logging - A tool for answering queries using smolagents
with detailed progress information and intermediate outputs
"""

import os
import re
import time
from typing import Optional, Dict, List, Any, Union, Callable
from smolagents import CodeAgent, tool, HfApiModel
from smolagents.agents import ActionStep  # For accessing intermediate steps

# Initialize global variables for tracking
verbose = True
tool_usage = {
    "reformulate_question": 0,
    "browse_web": 0,
    "read_file": 0,
    "summarize": 0
}
execution_times = {
    "reformulate_question": [],
    "browse_web": [],
    "read_file": [],
    "summarize": []
}
intermediate_outputs = []

def log_message(message: str) -> None:
    """Print a log message if verbose logging is enabled."""
    if verbose:
        print(message)

def log_tool_start(tool_name: str) -> float:
    """Log when a tool starts execution and return the start time."""
    start_time = time.time()
    if verbose:
        print(f"\n🔧 STARTING TOOL: {tool_name}")
        print(f"{'-'*40}")
    
    # Increment tool usage counter
    global tool_usage
    tool_usage[tool_name] += 1
    
    return start_time

def log_tool_end(tool_name: str, start_time: float, result_preview: str = "") -> None:
    """Log when a tool finishes execution."""
    end_time = time.time()
    execution_time = end_time - start_time
    
    global execution_times
    execution_times[tool_name].append(execution_time)
    
    if verbose:
        print(f"{'-'*40}")
        print(f"🔧 FINISHED TOOL: {tool_name} (took {execution_time:.2f}s)")
        
        if result_preview:
            # Display a preview of the result
            max_display_length = 500
            if len(result_preview) > max_display_length:
                print(f"RESULT PREVIEW: {result_preview[:max_display_length]}...\n[Result truncated, total length: {len(result_preview)} chars]")
            else:
                print(f"RESULT: {result_preview}")
        
        print(f"{'-'*40}\n")

@tool
def reformulate_question(query: str) -> Dict[str, Any]:
    """
    Analyzes and reformulates a query to better understand its structure and requirements.
    
    Args:
        query: The original query to analyze
        
    Returns:
        A dictionary containing reformulated query and requirements
    """
    start_time = log_tool_start("reformulate_question")
    
    # Create a clear prompt for the model
    prompt = f"""
    Analyze this query: "{query}"
    
    1. Identify what information is being requested
    2. Determine what sources would have this information
    3. Specify what format the answer should be in
    
    Return your analysis as a JSON with these keys:
    - reformulated_query: A clearer version of the question
    - information_needed: List of specific information required
    - source_types: List of sources that might have this info (web, files, etc.)
    - response_format: Preferred format for the answer
    """
    
    # Use a specialized agent for this task
    model = HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")
    reformulator = CodeAgent(
        tools=[],
        model=model,
        verbosity_level=1  # Reduced verbosity for sub-agents
    )
    
    # Get the analysis
    result = reformulator.run(prompt)
    
    # Try to extract JSON from the result
    json_match = re.search(r'\{[\s\S]*\}', result)
    if json_match:
        import json
        try:
            extracted_result = json.loads(json_match.group())
            log_tool_end("reformulate_question", start_time, str(extracted_result))
            
            # Store as intermediate output
            global intermediate_outputs
            intermediate_outputs.append({
                "tool": "reformulate_question",
                "output": extracted_result,
                "timestamp": time.time()
            })
            
            return extracted_result
        except json.JSONDecodeError:
            pass
    
    # If JSON extraction fails, return a default structure
    default_result = {
        "reformulated_query": query,
        "information_needed": ["general information"],
        "source_types": ["web", "file"],
        "response_format": "text"
    }
    
    log_tool_end("reformulate_question", start_time, str(default_result))
    
    # Store as intermediate output
    intermediate_outputs.append({
        "tool": "reformulate_question",
        "output": default_result,
        "timestamp": time.time()
    })
    
    return default_result

@tool
def browse_web(url: str, query: Optional[str] = None) -> str:
    """
    Fetches and processes content from a web page.
    
    Args:
        url: The URL to fetch
        query: Optional specific query to focus on when processing the page
        
    Returns:
        The extracted and processed text content
    """
    start_time = log_tool_start("browse_web")
    
    import requests
    from bs4 import BeautifulSoup
    
    # Send a request to get the page content
    try:
        log_message(f"📡 Fetching URL: {url}")
            
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        log_message(f"✅ Fetched successfully: {len(response.content)} bytes")
        
        # Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        log_message(f"🔍 Extracting content...")
        
        # Try to find the main content
        main_content = soup.find('main') or soup.find('article') or soup.find(id='content') or soup.find(class_='content')
        
        if main_content:
            content = main_content.get_text(separator='\n', strip=True)
            log_message(f"✅ Found main content container")
        else:
            # Fall back to body text
            content = soup.body.get_text(separator='\n', strip=True)
            log_message(f"⚠️ No main content container found, using body text")
        
        # If a specific query is provided, use another agent to focus the extraction
        if query:
            log_message(f"🔎 Focusing extraction on query: {query}")
                
            model = HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")
            focus_agent = CodeAgent(
                tools=[],
                model=model,
                verbosity_level=1  # Reduced verbosity for sub-agents
            )
            
            prompt = f"""
            From the following web page content, extract only the information relevant to this query:
            "{query}"
            
            WEB PAGE CONTENT:
            {content[:5000]}  # Limit content to avoid token limits
            """
            
            focused_content = focus_agent.run(prompt)
            
            # Store as intermediate output
            global intermediate_outputs
            intermediate_outputs.append({
                "tool": "browse_web",
                "output": focused_content,
                "timestamp": time.time(),
                "url": url
            })
            
            log_tool_end("browse_web", start_time, focused_content)
            return focused_content
        
        # Store as intermediate output
        intermediate_outputs.append({
            "tool": "browse_web",
            "output": content,
            "timestamp": time.time(),
            "url": url
        })
        
        log_tool_end("browse_web", start_time, f"Extracted {len(content)} characters")
        return content
        
    except Exception as e:
        error_msg = f"Error browsing web page {url}: {str(e)}"
        
        # Store error as intermediate output
        intermediate_outputs.append({
            "tool": "browse_web",
            "output": error_msg,
            "timestamp": time.time(),
            "url": url,
            "error": True
        })
        
        log_tool_end("browse_web", start_time, error_msg)
        return error_msg

@tool
def read_file(file_path: str, query: Optional[str] = None) -> str:
    """
    Reads and processes content from a file.
    
    Args:
        file_path: Path to the file to read
        query: Optional specific query to focus on when processing the file
        
    Returns:
        The extracted and processed file content
    """
    start_time = log_tool_start("read_file")
    
    try:
        file_ext = file_path.split('.')[-1].lower()
        
        log_message(f"📂 Reading file: {file_path}")
        log_message(f"📄 File type: {file_ext}")
        
        # Handle different file types
        if file_ext == 'pdf':
            # Read PDF file
            from PyPDF2 import PdfReader
            
            reader = PdfReader(file_path)
            content = ""
            
            # Limit to first 10 pages to avoid token limits
            max_pages = min(10, len(reader.pages))
            
            log_message(f"📑 PDF has {len(reader.pages)} pages, reading first {max_pages}")
            
            for i in range(max_pages):
                if i % 5 == 0:
                    log_message(f"  Reading page {i+1}...")
                content += reader.pages[i].extract_text() + "\n\n"
            
        elif file_ext in ['txt', 'md', 'html']:
            # Read text file
            log_message(f"📝 Reading text file")
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
        elif file_ext == 'csv':
            # Read CSV file - simple version
            log_message(f"📊 Reading CSV file")
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
        else:
            error_msg = f"Unsupported file type: {file_ext}"
            
            # Store error as intermediate output
            global intermediate_outputs
            intermediate_outputs.append({
                "tool": "read_file",
                "output": error_msg,
                "timestamp": time.time(),
                "file_path": file_path,
                "error": True
            })
            
            log_tool_end("read_file", start_time, error_msg)
            return error_msg
        
        log_message(f"✅ File read successfully: {len(content)} characters")
        
        # If a specific query is provided, use another agent to focus the extraction
        if query:
            log_message(f"🔎 Focusing extraction on query: {query}")
                
            model = HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")
            focus_agent = CodeAgent(
                tools=[],
                model=model,
                verbosity_level=1  # Reduced verbosity for sub-agents
            )
            
            prompt = f"""
            From the following file content, extract only the information relevant to this query:
            "{query}"
            
            FILE CONTENT:
            {content[:5000]}  # Limit content to avoid token limits
            """
            
            focused_content = focus_agent.run(prompt)
            
            # Store as intermediate output
            intermediate_outputs.append({
                "tool": "read_file",
                "output": focused_content,
                "timestamp": time.time(),
                "file_path": file_path
            })
            
            log_tool_end("read_file", start_time, focused_content)
            return focused_content
        
        # Store as intermediate output
        intermediate_outputs.append({
            "tool": "read_file",
            "output": content,
            "timestamp": time.time(),
            "file_path": file_path
        })
        
        log_tool_end("read_file", start_time, f"Extracted {len(content)} characters")
        return content
        
    except Exception as e:
        error_msg = f"Error reading file {file_path}: {str(e)}"
        
        # Store error as intermediate output
        intermediate_outputs.append({
            "tool": "read_file",
            "output": error_msg,
            "timestamp": time.time(),
            "file_path": file_path,
            "error": True
        })
        
        log_tool_end("read_file", start_time, error_msg)
        return error_msg

@tool
def summarize(text: str, max_length: int = 500, focus: Optional[str] = None) -> str:
    """
    Summarizes text content.
    
    Args:
        text: The text to summarize
        max_length: Maximum length of the summary
        focus: Optional aspect to focus on
        
    Returns:
        The summary of the text
    """
    start_time = log_tool_start("summarize")
    
    log_message(f"📝 Summarizing {len(text)} characters")
    log_message(f"📏 Max length: {max_length}")
    if focus:
        log_message(f"🔍 Focus: {focus}")
    
    # Create a specialized agent for summarization
    model = HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")
    summarizer = CodeAgent(
        tools=[],
        model=model,
        verbosity_level=1  # Reduced verbosity for sub-agents
    )
    
    prompt = f"""
    Summarize the following text in no more than {max_length} characters.
    
    {f'Focus on aspects related to: {focus}' if focus else 'Provide a general summary.'}
    
    TEXT TO SUMMARIZE:
    {text[:10000]}  # Limit to avoid token limits
    """
    
    summary = summarizer.run(prompt)
    
    # Ensure the summary is within the length limit
    if len(summary) > max_length:
        summary = summary[:max_length-3] + "..."
    
    # Store as intermediate output
    global intermediate_outputs
    intermediate_outputs.append({
        "tool": "summarize",
        "output": summary,
        "timestamp": time.time(),
        "text_length": len(text),
        "summary_length": len(summary)
    })
    
    log_tool_end("summarize", start_time, summary)
    return summary

def create_log_callback() -> Callable[[ActionStep], None]:
    """Creates a callback function to log intermediate steps."""
    
    def log_step(step: ActionStep) -> None:
        """Callback function to log each step of the agent's execution."""
        if not verbose:
            return
            
        step_num = step.step_num
        action_type = step.action_type
        
        print(f"\n{'='*80}")
        print(f"STEP {step_num}: {action_type}")
        print(f"{'='*80}")
        
        if action_type == "thinking":
            print(f"💭 THINKING: {step.action}")
            
        elif action_type == "code":
            print(f"🔧 EXECUTING CODE:")
            print(f"{'-'*40}")
            print(step.action)
            print(f"{'-'*40}")
            
            # Identify tool calls in the code
            tools_found = []
            code_lines = step.action.strip().split('\n')
            for line in code_lines:
                # Look for tool function calls
                for tool_name in tool_usage.keys():
                    if f"{tool_name}(" in line:
                        tools_found.append(tool_name)
            
            if tools_found:
                tools_str = ", ".join(tools_found)
                print(f"🔍 TOOLS DETECTED: {tools_str}")
            
        elif action_type == "observation":
            print(f"👁️ OBSERVATION:")
            print(f"{'-'*40}")
            
            # The observation is often the output of the code execution
            output = step.action
            
            # Limit output length for display
            max_display_length = 1000
            if len(output) > max_display_length:
                print(f"{output[:max_display_length]}...\n[Output truncated, total length: {len(output)} chars]")
            else:
                print(output)
            print(f"{'-'*40}")
        
        print(f"{'='*80}\n")
    
    return log_step

def process_query(query: str, model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct", enable_verbose: bool = True) -> str:
    """
    Process a query using the appropriate tools with progress logging.
    
    Args:
        query: The query to process
        model_id: ID of the model to use
        enable_verbose: Whether to enable verbose logging
        
    Returns:
        The response to the query
    """
    # Set global verbose flag
    global verbose
    verbose = enable_verbose
    
    # Reset tracking for this query
    global tool_usage, execution_times, intermediate_outputs
    tool_usage = {tool: 0 for tool in tool_usage}
    execution_times = {tool: [] for tool in execution_times}
    intermediate_outputs = []
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"🔍 PROCESSING QUERY: {query}")
        print(f"{'='*80}\n")
        print(f"🤖 Using model: {model_id}")
        print(f"🧰 Available tools: {', '.join(tool_usage.keys())}")
        print(f"⏱️ Starting processing at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
    
    # Initialize the model
    model = HfApiModel(model_id=model_id)
    
    # Create the log callback
    log_callback = create_log_callback()
    
    system_prompt = """
    You are an intelligent assistant that answers questions by calling the appropriate tools.
    Follow these steps to process queries effectively:
    
    1. Use reformulate_question to understand the query structure and requirements
    2. Decide which tools to use (browse_web, read_file, or both)
    3. Gather information using the appropriate tools
    4. If needed, use summarize to create a concise response
    5. Return a clear, comprehensive answer to the query
    
    Always explain your reasoning process and cite the sources of information you used.
    
    IMPORTANT: Include detailed comments in your code to explain your decision-making process.
    """
    
    # Create the agent with tools and callback
    agent = CodeAgent(
        tools=[
            reformulate_question,
            browse_web,
            read_file,
            summarize
        ],
        model=model,
        system_prompt=system_prompt,
        additional_authorized_imports=["requests", "bs4", "PyPDF2", "re", "json", "time"],
        verbosity_level=2 if verbose else 1,
        step_callbacks=[log_callback]
    )
    
    start_time = time.time()
    
    # Process the query
    try:
        result = agent.run(query)
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"✅ QUERY PROCESSING COMPLETE")
            print(f"{'='*80}")
            print(f"⏱️ Total processing time: {total_time:.2f} seconds")
            
            # Display tool usage statistics
            print(f"\n📊 TOOL USAGE STATISTICS:")
            print(f"{'-'*40}")
            for tool, count in tool_usage.items():
                if count > 0:
                    avg_time = sum(execution_times[tool]) / count if count > 0 else 0
                    print(f"- {tool}: Used {count} times (avg. {avg_time:.2f}s per call)")
                else:
                    print(f"- {tool}: Not used")
            
            print(f"\n🔄 PROCESSING FLOW:")
            for i, output in enumerate(intermediate_outputs):
                print(f"- {output['tool']}: Generated output of {len(str(output['output']))} characters")
            
            print(f"\n{'='*80}")
        
        return result
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        if verbose:
            print(f"\n❌ ERROR: {error_msg}")
        return error_msg

def get_performance_stats() -> Dict[str, Any]:
    """
    Returns performance statistics for the last query processed.
    
    Returns:
        Dictionary with performance statistics
    """
    stats = {
        "tool_usage": tool_usage.copy(),
        "tool_execution_times": {
            tool: {
                "calls": len(times),
                "total_time": sum(times),
                "avg_time": sum(times) / len(times) if times else 0,
                "min_time": min(times) if times else 0,
                "max_time": max(times) if times else 0
            }
            for tool, times in execution_times.items() if times
        },
        "intermediate_outputs_count": len(intermediate_outputs),
        "intermediate_outputs": [
            {
                "tool": output["tool"],
                "output_length": len(str(output["output"])),
                "timestamp": output["timestamp"]
            }
            for output in intermediate_outputs
        ]
    }
    
    return stats


# Example usage
def main():
    # Process a query with verbose logging
    query = 'If Eliud Kipchoge could maintain his record-making marathon pace indefinitely, how many thousand hours would it take him to run the distance between the Earth and the Moon its closest approach? Please use the minimum perigee value on the Wikipedia page for the Moon when carrying out your calculation. Round your result to the nearest 1000 hours and do not use any comma separators if necessary.'
    
    print(f"Query: {query}")
    
    # Process the query - verbose logging will show progress
    result = process_query(query, enable_verbose=True)
    
    print("\nFINAL RESULT:")
    print("-" * 80)
    print(result)
    print("-" * 80)
    
    # Display performance statistics
    stats = get_performance_stats()
    print("\nPERFORMANCE STATISTICS:")
    import json
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()