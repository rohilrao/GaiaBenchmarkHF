# Implement a simple agent that uses Ollama and Llama 3.2 Vision
# https://ominousindustries.com/blogs/ominous-industries/a-demo-of-magentic-one-agents-working-with-ollama-and-llama-3-2-vision?utm_source=chatgpt.com

# Import necessary modules
from magentic_one import Agent, Orchestrator
from ollama import OllamaClient

# Initialize the Ollama client
ollama_client = OllamaClient(server_url="http://localhost:8000")

# Define a simple agent using Magentic One
class VisionAgent(Agent):
    def __init__(self, name, ollama_client):
        super().__init__(name)
        self.ollama_client = ollama_client

    def process(self, input_data):
        # Use the Llama 3.2 Vision model for processing
        response = self.ollama_client.query(model="llama-3.2-vision", input_data=input_data)
        return response

# Initialize the orchestrator
orchestrator = Orchestrator()

# Add the VisionAgent to the orchestrator
vision_agent = VisionAgent(name="VisionAgent", ollama_client=ollama_client)
orchestrator.add_agent(vision_agent)

# Example usage
if __name__ == "__main__":
    input_data = "Describe the content of the image."
    result = orchestrator.run(input_data)
    print("Result:", result)