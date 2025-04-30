import requests
import json
from typing import List, Dict, Any, Optional


def search_duckduckgo(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web using DuckDuckGo's Instant Answer API
    
    Args:
        query (str): The search query
        num_results (int): Maximum number of results to return (default: 5)
        
    Returns:
        List[Dict[str, Any]]: List of search results
    """
    try:
        # Encode the query
        encoded_query = requests.utils.quote(query)
        
        # DuckDuckGo Instant Answer API URL
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&pretty=1"
        
        # Make the request
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Parse the response
        data = response.json()
        
        # Format the results
        results = []
        
        # Add the Abstract if available
        if data.get("Abstract"):
            results.append({
                "type": "abstract",
                "text": data["Abstract"],
                "source": data.get("AbstractSource", ""),
                "url": data.get("AbstractURL", "")
            })
        
        # Add related topics
        if data.get("RelatedTopics"):
            for topic in data["RelatedTopics"][:num_results - len(results)]:
                # Skip category headers
                if not topic.get("Text"):
                    continue
                    
                results.append({
                    "type": "related",
                    "text": topic["Text"],
                    "url": topic.get("FirstURL", "")
                })
        
        return results
        
    except Exception as e:
        return [{"type": "error", "message": str(e)}]


def llm_search(query: str, num_results: int = 5) -> str:
    """
    Search function designed specifically for LLM consumption
    
    Args:
        query (str): The search query
        num_results (int): Maximum number of results to return
        
    Returns:
        str: Formatted search results as a string
    """
    if not query:
        return "Error: No search query provided"
    
    try:
        results = search_duckduckgo(query, num_results)
        
        # Format results for LLM consumption
        formatted_response = f'Search results for: "{query}"\n\n'
        
        if not results:
            formatted_response += "No results found."
        else:
            for i, result in enumerate(results):
                formatted_response += f"[{i + 1}] "
                
                if result["type"] == "abstract":
                    formatted_response += f"SUMMARY FROM {result['source']}: {result['text']}\nSource: {result['url']}\n\n"
                elif result["type"] == "related":
                    formatted_response += f"{result['text']}\nLink: {result['url']}\n\n"
                elif result["type"] == "error":
                    formatted_response += f"ERROR: {result['message']}\n\n"
        
        return formatted_response.strip()
        
    except Exception as e:
        return f"Search error: {str(e)}"


# Example usage
if __name__ == "__main__":
   # Basic search
    query = "usain bolt speed"
    results = search_duckduckgo(query, 3)
    print(json.dumps(results, indent=2))
    
    # LLM-friendly search
    llm_results = llm_search(query, 3)
    print("\n" + "-" * 50 + "\n")
    print(llm_results) 