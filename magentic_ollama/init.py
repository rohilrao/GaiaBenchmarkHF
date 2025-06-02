# Implement a simple agent that uses Ollama and Llama 3.2 Vision
# https://ominousindustries.com/blogs/ominous-industries/a-demo-of-magentic-one-agents-working-with-ollama-and-llama-3-2-vision?utm_source=chatgpt.com

# Import necessary modules
from autogen_ext.teams.magentic_one import MagenticOne, Orchestrator
from autogen_ext.models.ollama import OllamaChatCompletionClient

# Initialize the Ollama client with the required model parameter
ollama_client = OllamaChatCompletionClient(server_url="http://localhost:11434", model="llama3.1:8b")

# Define a simple agent using Magentic One
class VisionAgent(MagenticOne):
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
    input_data = "Say hello if you are there."
    result = orchestrator.run(input_data)
    print("Result:", result)