import requests
from bs4 import BeautifulSoup
import json
from typing import List, Dict, Any, Optional
import re
import urllib.parse


def search_web(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web using multiple strategies to ensure better results
    
    Args:
        query (str): The search query
        num_results (int): Maximum number of results to return (default: 5)
        
    Returns:
        List[Dict[str, Any]]: List of search results
    """
    # First try DuckDuckGo Instant Answer API
    duckduckgo_results = search_duckduckgo_api(query)
    
    # If we got meaningful results, return them
    if duckduckgo_results and len(duckduckgo_results) > 1:
        return duckduckgo_results[:num_results]
    
    # Otherwise, fall back to scraping search results
    scraped_results = scrape_duckduckgo(query, num_results)
    
    # Combine results, prioritizing any API results we did get
    combined_results = duckduckgo_results + scraped_results
    return combined_results[:num_results]


def search_duckduckgo_api(query: str) -> List[Dict[str, Any]]:
    """
    Search using the DuckDuckGo Instant Answer API
    
    Args:
        query (str): The search query
        
    Returns:
        List[Dict[str, Any]]: List of search results
    """
    try:
        # Encode the query
        encoded_query = urllib.parse.quote(query)
        
        # DuckDuckGo Instant Answer API URL
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&pretty=1"
        
        # Make the request
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        # Parse the response
        data = response.json()
        
        # Format the results
        results = []
        
        # Check for direct answer
        if data.get("Answer") and data["Answer"] != "":
            results.append({
                "type": "answer",
                "text": data["Answer"],
                "source": "DuckDuckGo"
            })
        
        # Add the Abstract if available
        if data.get("Abstract") and data["Abstract"] != "":
            results.append({
                "type": "abstract",
                "text": data["Abstract"],
                "source": data.get("AbstractSource", "DuckDuckGo"),
                "url": data.get("AbstractURL", "")
            })
        
        # Add related topics
        if data.get("RelatedTopics"):
            for topic in data["RelatedTopics"]:
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
        print(f"API search error: {str(e)}")
        return []


def scrape_duckduckgo(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Scrape DuckDuckGo search results page
    
    Args:
        query (str): The search query
        num_results (int): Maximum number of results to return
        
    Returns:
        List[Dict[str, Any]]: List of search results
    """
    try:
        # Headers to mimic a browser visit
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        
        # Encode the query
        encoded_query = urllib.parse.quote(query)
        
        # DuckDuckGo search URL
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        # Make the request
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the result elements
        results = []
        
        # Check for the info box (often contains factual answers)
        info_box = soup.find('div', {'class': 'info-box'})
        if info_box:
            info_text = info_box.get_text(strip=True)
            if info_text:
                results.append({
                    "type": "fact",
                    "text": info_text,
                    "source": "DuckDuckGo"
                })
        
        # Get regular search results
        result_elements = soup.select('.result')
        
        for element in result_elements[:num_results]:
            # Extract the title and link
            title_element = element.select_one('.result__title')
            link_element = element.select_one('.result__url')
            snippet_element = element.select_one('.result__snippet')
            
            if title_element and snippet_element:
                title = title_element.get_text(strip=True)
                snippet = snippet_element.get_text(strip=True)
                
                # Get the URL
                url = ""
                if link_element:
                    url = link_element.get_text(strip=True)
                elif title_element.find('a'):
                    url = title_element.find('a').get('href', '')
                
                results.append({
                    "type": "result",
                    "title": title,
                    "text": snippet,
                    "url": url
                })
        
        return results
    
    except Exception as e:
        print(f"Scraping error: {str(e)}")
        return []


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
        results = search_web(query, num_results)
        
        # Format results for LLM consumption
        formatted_response = f'Search results for: "{query}"\n\n'
        
        if not results:
            formatted_response += "No results found."
        else:
            for i, result in enumerate(results):
                formatted_response += f"[{i + 1}] "
                
                if result["type"] == "answer":
                    formatted_response += f"DIRECT ANSWER: {result['text']}\n\n"
                elif result["type"] == "abstract":
                    formatted_response += f"SUMMARY FROM {result['source']}: {result['text']}\n"
                    if result.get('url'):
                        formatted_response += f"Source: {result['url']}\n\n"
                    else:
                        formatted_response += "\n"
                elif result["type"] == "fact":
                    formatted_response += f"FACT: {result['text']}\n\n"
                elif result["type"] == "related":
                    formatted_response += f"{result['text']}\n"
                    if result.get('url'):
                        formatted_response += f"Link: {result['url']}\n\n"
                    else:
                        formatted_response += "\n"
                elif result["type"] == "result":
                    formatted_response += f"TITLE: {result['title']}\n{result['text']}\n"
                    if result.get('url'):
                        formatted_response += f"URL: {result['url']}\n\n"
                    else:
                        formatted_response += "\n"
                elif result["type"] == "error":
                    formatted_response += f"ERROR: {result['message']}\n\n"
        
        return formatted_response.strip()
        
    except Exception as e:
        return f"Search error: {str(e)}"


# Example usage
if __name__ == "__main__":
    # Test with a factual query
    query = "distance moon to earth"
    print(f"Searching for: {query}")
    
    # Get LLM-friendly search results
    llm_results = llm_search(query, 3)
    print("\n" + "-" * 50 + "\n")
    print(llm_results)