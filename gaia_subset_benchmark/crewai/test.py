from crewai import Agent, Task, Crew
from langchain_community.llms import Ollama

# Initialize Ollama LLM
llm = Ollama(model="llama3.2:3b", base_url="http://localhost:11434")  # Change model/URL as needed

# Create agent
agent = Agent(
    role="Assistant",
    goal="Answer questions accurately",
    backstory="You are a helpful AI assistant that provides clear answers.",
    llm=llm,
    verbose=True
)

# Function to ask questions
def ask_question(question):
    task = Task(
        description=f"Answer this question: {question}",
        agent=agent,
        expected_output="A clear and helpful answer"
    )
    
    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()
    return result

# Example usage
if __name__ == "__main__":
    question = "What is the capital of France?"
    answer = ask_question(question)
    print(f"Answer: {answer}")