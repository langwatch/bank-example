"""
Main Bank Customer Support Agent - Simple Agno agent with tools
"""

import os
import dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat

import langwatch
from openinference.instrumentation.agno import AgnoInstrumentor

langwatch.setup(instrumentors=[AgnoInstrumentor()])

dotenv.load_dotenv()

# Create the main support agent
support_agent = Agent(
    name="BankCustomerSupportAgent",
    model=OpenAIChat(
        id="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
    tools=[],
    description="You are a helpful assistant.",
    add_history_to_context=True,  # Let Agno handle memory
)


# Simple interface for testing
def chat_with_agent(message: str) -> str:
    """Simple interface to chat with the agent"""
    response = support_agent.run(message)
    return response.content  # type: ignore
