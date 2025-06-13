import os
from mistralai import Mistral

# Setup
api_key = "xsGohkeJydnHkFARLFjr5WdCnZGG5Brp"
client = Mistral(api_key=api_key)

# Create agent with premium web search
premium_agent = client.beta.agents.create(
    model="mistral-tiny",
    name="Premium Web Research Agent",
    description="Advanced web research agent with news agency access",
    instructions="""You are a research assistant with access to web search and premium news sources. 
    Always provide sources for your information and cite reputable news agencies when available.""",
    tools=[{"type": "web_search_premium"}],  # Premium version with AFP/AP access
    completion_args={
        "temperature": 0.1,  # Lower temperature for more factual responses
        "top_p": 0.9,
    }
)

def stream_agent_response(question, agent_id):
    """Stream the agent's response in real-time"""
    print(f"Question: {question}")
    print("Agent is thinking and searching...")
    print("-" * 50)
    
    # Create streaming conversation
    stream = client.beta.agents.conversations.create(
        agent_id=agent_id,
        inputs=question,
        stream=True  # Enable streaming
    )
    
    conversation_id = None
    
    for event in stream:
        if hasattr(event, 'type'):
            if event.type == "conversation.response.started":
                conversation_id = event.conversation_id
                print(f"Started conversation: {conversation_id}")
            
            elif event.type == "tool.execution.started":
                print(f"🔍 Starting {event.name}...")
            
            elif event.type == "tool.execution.done":
                print(f"✅ Completed {event.name}")
            
            elif event.type == "message.output.delta":
                # Stream the actual response text
                if hasattr(event, 'content'):
                    for content in event.content:
                        if hasattr(content, 'text'):
                            print(content.text, end='', flush=True)
            
            elif event.type == "conversation.response.done":
                print("\n" + "=" * 50)
                print("Response completed!")
                break
    
    return conversation_id

# Example usage
if __name__ == "__main__":
    # Research current events
    research_question = "What are the major global economic developments this week?"
    
    conv_id = stream_agent_response(research_question, premium_agent.id)
    
    # You can continue the conversation
    print("\n\nAsking follow-up question...")
    follow_up = "Can you provide more details about any market impacts?"
    
    follow_up_stream = client.beta.agents.conversations.continue_conversation(
        conversation_id=conv_id,
        inputs=follow_up,
        stream=True
    )
    
    print("Follow-up response:")
    for event in follow_up_stream:
        if hasattr(event, 'type') and event.type == "message.output.delta":
            if hasattr(event, 'content'):
                for content in event.content:
                    if hasattr(content, 'text'):
                        print(content.text, end='', flush=True)