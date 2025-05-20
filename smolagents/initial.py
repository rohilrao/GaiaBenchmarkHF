"""
Query Assistant - A tool for answering queries using smolagents
with specialized tools for reformulation, web browsing, file reading, and summarization
"""

import os
import re
from typing import Optional, Dict, List, Any, Union
from smolagents import CodeAgent, tool, HfApiModel

class QueryAssistant:
    """
    A streamlined assistant that takes a query string and uses the appropriate tools to answer it.
    Features include question reformulation, web browsing, file reading, and summarization.
    """
    
    def __init__(self, model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct"):
        """
        Initialize the QueryAssistant.
        
        Args:
            model_id: HuggingFace model ID to use
        """
        # Initialize the model
        self.model = HfApiModel(model_id=model_id)
        
        # Create the main agent with all tools
        self.agent = CodeAgent(
            tools=[
                self.reformulate_question,
                self.browse_web,
                self.read_file,
                self.summarize
            ],
            model=self.model,
            additional_authorized_imports=["requests", "bs4", "PyPDF2", "re", "json"],
            verbosity_level=1
        )
    
    @tool
    def reformulate_question(self, query: str) -> Dict[str, Any]:
        """
        Analyzes and reformulates a query to better understand its structure and requirements.
        
        Args:
            query: The original query to analyze
            
        Returns:
            A dictionary containing reformulated query and requirements
        """
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
        reformulator = CodeAgent(
            tools=[],
            model=self.model
        )
        
        # Get the analysis
        result = reformulator.run(prompt)
        
        # Try to extract JSON from the result
        json_match = re.search(r'\{[\s\S]*\}', result)
        if json_match:
            import json
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # If JSON extraction fails, return a default structure
        return {
            "reformulated_query": query,
            "information_needed": ["general information"],
            "source_types": ["web", "file"],
            "response_format": "text"
        }
    
    @tool
    def browse_web(self, url: str, query: Optional[str] = None) -> str:
        """
        Fetches and processes content from a web page.
        
        Args:
            url: The URL to fetch
            query: Optional specific query to focus on when processing the page
            
        Returns:
            The extracted and processed text content
        """
        import requests
        from bs4 import BeautifulSoup
        
        # Send a request to get the page content
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Try to find the main content
            main_content = soup.find('main') or soup.find('article') or soup.find(id='content') or soup.find(class_='content')
            
            if main_content:
                content = main_content.get_text(separator='\n', strip=True)
            else:
                # Fall back to body text
                content = soup.body.get_text(separator='\n', strip=True)
            
            # If a specific query is provided, use another agent to focus the extraction
            if query:
                focus_agent = CodeAgent(
                    tools=[],
                    model=self.model
                )
                
                prompt = f"""
                From the following web page content, extract only the information relevant to this query:
                "{query}"
                
                WEB PAGE CONTENT:
                {content[:5000]}  # Limit content to avoid token limits
                """
                
                return focus_agent.run(prompt)
            
            return content
            
        except Exception as e:
            return f"Error browsing web page {url}: {str(e)}"
    
    @tool
    def read_file(self, file_path: str, query: Optional[str] = None) -> str:
        """
        Reads and processes content from a file.
        
        Args:
            file_path: Path to the file to read
            query: Optional specific query to focus on when processing the file
            
        Returns:
            The extracted and processed file content
        """
        try:
            file_ext = file_path.split('.')[-1].lower()
            
            # Handle different file types
            if file_ext == 'pdf':
                # Read PDF file
                from PyPDF2 import PdfReader
                
                reader = PdfReader(file_path)
                content = ""
                
                # Limit to first 10 pages to avoid token limits
                max_pages = min(10, len(reader.pages))
                for i in range(max_pages):
                    content += reader.pages[i].extract_text() + "\n\n"
                
            elif file_ext in ['txt', 'md', 'html']:
                # Read text file
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
            elif file_ext == 'csv':
                # Read CSV file - simple version
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
            else:
                return f"Unsupported file type: {file_ext}"
            
            # If a specific query is provided, use another agent to focus the extraction
            if query:
                focus_agent = CodeAgent(
                    tools=[],
                    model=self.model
                )
                
                prompt = f"""
                From the following file content, extract only the information relevant to this query:
                "{query}"
                
                FILE CONTENT:
                {content[:5000]}  # Limit content to avoid token limits
                """
                
                return focus_agent.run(prompt)
            
            return content
            
        except Exception as e:
            return f"Error reading file {file_path}: {str(e)}"
    
    @tool
    def summarize(self, text: str, max_length: int = 500, focus: Optional[str] = None) -> str:
        """
        Summarizes text content.
        
        Args:
            text: The text to summarize
            max_length: Maximum length of the summary
            focus: Optional aspect to focus on
            
        Returns:
            The summary of the text
        """
        # Create a specialized agent for summarization
        summarizer = CodeAgent(
            tools=[],
            model=self.model
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
            
        return summary
    
    def process_query(self, query: str) -> str:
        """
        The main function that processes a query using the appropriate tools.
        
        Args:
            query: The user query to process
            
        Returns:
            The response to the query
        """
        system_prompt = """
        You are an intelligent assistant that answers questions by calling the appropriate tools.
        Follow these steps to process queries effectively:
        
        1. Use reformulate_question to understand the query structure and requirements
        2. Decide which tools to use (browse_web, read_file, or both)
        3. Gather information using the appropriate tools
        4. If needed, use summarize to create a concise response
        5. Return a clear, comprehensive answer to the query
        
        Always explain your reasoning process and cite the sources of information you used.
        """
        
        # Create an agent with the system prompt
        processing_agent = CodeAgent(
            tools=[
                self.reformulate_question,
                self.browse_web,
                self.read_file,
                self.summarize
            ],
            model=self.model,
            system_prompt=system_prompt,
            additional_authorized_imports=["requests", "bs4", "PyPDF2", "re", "json"],
            verbosity_level=1
        )
        
        # Process the query
        try:
            return processing_agent.run(query)
        except Exception as e:
            return f"Error processing query: {str(e)}"


# Example usage
def main():
    # Initialize the assistant
    assistant = QueryAssistant()
    
    # Example query
    query = "What are the key features of smolagents and how does it compare to LangChain?"
    
    print(f"Query: {query}")
    print("\nProcessing...")
    
    # Process the query
    result = assistant.process_query(query)
    
    print("\nResult:")
    print(result)


if __name__ == "__main__":
    main()