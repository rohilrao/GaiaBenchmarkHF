import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import time
from concurrent.futures import ThreadPoolExecutor

def search_web(query, max_results=5):
    """Search DuckDuckGo and return results."""
    print(f"Searching for: {query}")
    try:
        results = DDGS().text(query, max_results=max_results)
        print(f"Found {len(results)} results.")
        return results
    except Exception as e:
        print(f"Error during search: {str(e)}")
        return []

def extract_content(url, title=None):
    """Extract text content from a URL using requests and BeautifulSoup."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Extract all text
        text = soup.get_text(separator='\n')
        
        # Clean text
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = '\n\n'.join(lines)
        
        return f"\nFrom {title or url}:\n{text}\n"
    except Exception as e:
        return f"\nError extracting content from {url}: {str(e)}\n"

def search_and_parse(query, max_results=5):
    """Search and extract content in parallel."""
    # Get search results
    results = search_web(query, max_results)
    
    if not results:
        return {"search_results": [], "parsed_content": ""}
    
    print("Extracting content from search results...")
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks for each URL
        future_to_url = {
            executor.submit(extract_content, result['href'], result['title']): result
            for result in results
        }
        
        # Collect results as they complete
        contents = []
        for future in future_to_url:
            try:
                content = future.result()
                if content:
                    contents.append(content)
            except Exception as e:
                print(f"Error processing result: {str(e)}")
    
    return {
        "search_results": results,
        "parsed_content": ''.join(contents)
    }

def process_questions(questions):
    """Process a list of questions."""
    results = {}
    for q in questions:
        print(f"\nProcessing question: {q}")
        try:
            result = search_and_parse(q)
            results[q] = result
        except Exception as e:
            print(f"Error processing question '{q}': {str(e)}")
            results[q] = {"error": str(e), "search_results": [], "parsed_content": ""}
        # Add a small delay between questions
        time.sleep(1)
    return results

# Example usage
if __name__ == "__main__":
    query = input("Enter your search query: ")
    result = search_and_parse(query)
    
    # Display search results
    print("\nSEARCH RESULTS:")
    for r in result["search_results"]:
        print(f"Title: {r['title']}")
        print(f"URL: {r['href']}")
        print(f"Snippet: {r['body']}")
        print("-" * 20)
    
    # Display parsed content
    print("\nPARSED CONTENT FROM TOP LINKS:")
    print(result["parsed_content"])