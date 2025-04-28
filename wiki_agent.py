from mcp.server.fastmcp import FastMCP
import wikipedia

# Create an MCP server
mcp = FastMCP("Wikipedia Agent")

# Add a tool
@mcp.tool()
def wiki_search(query: str) -> dict:
    """
    Searches Wikipedia and returns a summary for a given query.
    
    Args:
        query (str): The search term to find on Wikipedia.

    Returns:
        dict: Contains title, summary, and URL of the Wikipedia page.
    """
    try:
        page = wikipedia.page(query)
        return {
            "title": page.title,
            "summary": page.summary,
            "url": page.url
        }
    except Exception as e:
        return {"error": str(e)}

# Run the server
if __name__ == "__main__":
    if __name__ == "__main__":
        mcp.run(host="0.0.0.0", port=3333)
