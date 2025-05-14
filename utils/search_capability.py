import time
import random
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

class WebSearcher:
    """A comprehensive web search utility with multiple backends and automatic fallback."""
    # Singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36'
        ]
    
    def search_with_googlesearch(self, query, max_results=5):
        """Search using googlesearch-python library."""
        try:
            # Dynamically import to avoid errors if not installed
            from googlesearch import search
            
            print("Searching with googlesearch-python...")
            time.sleep(random.uniform(1, 3))
            
            search_results = []
            for url in search(query, num_results=max_results):
                # Get page title and snippet by fetching a bit of content
                title = url
                snippet = ""
                try:
                    headers = {'User-Agent': random.choice(self.user_agents)}
                    response = requests.get(url, headers=headers, timeout=5)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        title = soup.title.string if soup.title else url
                        # Get first paragraph or snippet of text
                        first_p = soup.find('p')
                        snippet = first_p.get_text().strip()[:150] + "..." if first_p else ""
                except Exception:
                    # If we can't get extra info, just use the URL
                    pass
                    
                search_results.append({
                    'href': url,
                    'title': title,
                    'body': snippet
                })
                time.sleep(0.5)
            
            print(f"Found {len(search_results)} results with googlesearch-python.")
            return search_results
        except Exception as e:
            print(f"googlesearch-python error: {str(e)}")
            return None
    
    def search_with_ddgs(self, query, max_results=5):
        """Search using DDGS library."""
        try:
            # Dynamically import to avoid errors if not installed
            from duckduckgo_search import DDGS
            
            print("Searching with DDGS...")
            time.sleep(random.uniform(2, 4))
            
            results = DDGS().text(query, max_results=max_results)
            if results:
                print(f"Found {len(results)} results with DDGS.")
                return list(results)  # Convert generator to list
            return None
        except Exception as e:
            print(f"DDGS error: {str(e)}")
            return None
    
    def search_with_selenium(self, query, max_results=5):
        """Search using Selenium with headless Chrome."""
        print("Searching with Selenium...")
        try:
            # Configure headless Chrome
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument(f"--user-agent={random.choice(self.user_agents)}")
            
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            
            # Go to Google and search
            driver.get(f"https://www.google.com/search?q={query}")
            time.sleep(random.uniform(2, 4))  # Wait for results to load
            
            # Extract search results
            results = []
            elements = driver.find_elements(By.CSS_SELECTOR, ".g")
            
            for i, element in enumerate(elements):
                if i >= max_results:
                    break
                    
                try:
                    link_element = element.find_element(By.CSS_SELECTOR, "a")
                    link = link_element.get_attribute("href")
                    
                    # Get title from h3 element
                    title_element = element.find_element(By.CSS_SELECTOR, "h3")
                    title = title_element.text if title_element else link
                    
                    # Get snippet
                    snippet = ""
                    try:
                        snippet_element = element.find_element(By.CSS_SELECTOR, ".VwiC3b")
                        snippet = snippet_element.text
                    except:
                        pass
                    
                    if link and not link.startswith('https://webcache.googleusercontent'):
                        results.append({
                            'href': link,
                            'title': title,
                            'body': snippet
                        })
                except Exception as e:
                    print(f"Error extracting result: {str(e)}")
                    continue
                    
            driver.quit()
            print(f"Found {len(results)} results with Selenium.")
            return results
            
        except Exception as e:
            print(f"Selenium search error: {str(e)}")
            return None
    
    def search_with_serpapi(self, query, max_results=5):
        """Search using SerpAPI as a last resort."""
        try:
            from serpapi import GoogleSearch
            
            print("Searching with SerpAPI...")
            # You should set your API key as an environment variable
            import os
            api_key = os.getenv("SERPAPI_API_KEY")

            params = {
                "engine": "google",
                "q": query,
                "api_key": api_key,
                "num": max_results
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            if "organic_results" in results:
                formatted_results = []
                for result in results["organic_results"][:max_results]:
                    formatted_results.append({
                        'href': result.get('link', ''),
                        'title': result.get('title', ''),
                        'body': result.get('snippet', '')
                    })
                print(f"Found {len(formatted_results)} results with SerpAPI.")
                return formatted_results
            return None
        except Exception as e:
            print(f"SerpAPI error: {str(e)}")
            return None

    def search(self, query, max_results=5, max_retries=3):
        """
        Search the web using multiple methods with fallback.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of search results or empty list if all methods fail
        """
        print(f"Searching for: {query}")
        
        # Try different search methods in sequence
        search_methods = [
            self.search_with_ddgs,         # Try DDGS first (fastest when it works)
            self.search_with_googlesearch,  # Then try googlesearch-python
            self.search_with_selenium,      # Then try Selenium (reliable but heavy)
            self.search_with_serpapi        # Finally try SerpAPI as last resort
        ]
        
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"Retry attempt {attempt+1}/{max_retries}")
                
            # Add increasing delay between retries
            delay = (attempt + 1) * 2 + random.uniform(0, 1)
            if attempt > 0:
                print(f"Waiting {delay:.2f}s before retry...")
                time.sleep(delay)
            
            # Try each search method until one succeeds
            for method in search_methods:
                results = method(query, max_results)
                if results:
                    return results
                
                # Small delay between trying different methods
                time.sleep(1)
                
        print("All search methods failed after retries.")
        return []


def extract_content(url, title=None):
    """Extract text content from a URL using requests and BeautifulSoup."""
    try:
        headers = {
            'User-Agent': random.choice(WebSearcher.get_instance().user_agents)
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
    searcher = WebSearcher.get_instance()
    results = searcher.search(query, max_results)
    
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


# Example usage
def main():
    # Test the search function
    results = search_and_parse("eliud kipchoge top speed", max_results=3)
    print(f"Found {len(results['search_results'])} results.")
    print(f"Content length: {len(results['parsed_content'])} characters")
    return results

if __name__ == "__main__":
    main()