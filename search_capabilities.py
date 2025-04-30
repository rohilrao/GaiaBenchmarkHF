import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS as ddg

def search_and_parse(query, max_results=5):
    # Get search results
    results = ddg().text(query, max_results=max_results)
    
    # Parse content from each link
    all_content = ""
    for r in results:
        try:
            response = requests.get(r['href'], timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Extract text from paragraph tags
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text() for p in paragraphs])
                all_content += f"\n\nFrom {r['title']}:\n{content[:500]}...\n"
        except Exception as e:
            print(f"Error fetching {r['href']}: {e}")
    
    return {
        "search_results": results,
        "parsed_content": all_content
    }

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
