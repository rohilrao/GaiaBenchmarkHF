from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP(name = "Demo", host = '0.0.0.0', port = 6275)

# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

# 👇 ADD THIS to actually start the server
if __name__ == "__main__":
    transport = "sse":
    if transport == "stdio":
        print("Starting server with stdio transport...")
        mcp.run(transport=  "stdio")
    elif transport == "sse":
        print("Starting server with SSE transport...")
        mcp.run(transport=  "sse")
    else:
        raise ValueError("Invalid transport type. Use 'stdio' or 'sse'.")
    
