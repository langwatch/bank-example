"""
Main Bank Customer Support Agent - Simple Agno agent with tools

This is the production code - kept very simple. One agent with tools, Agno handles memory.
"""

import os
import json
from typing import Dict, Any
import dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.nebius import Nebius

# Import our specialized agents as tools
from agents.summary_agent import summarize_conversation
from agents.next_message_agent import suggest_next_message
from agents.customer_explorer_agent import (
    explore_customer_context,
    analyze_customer_behavior,
)

import langwatch
from openinference.instrumentation.agno import AgnoInstrumentor

langwatch.setup(instrumentors=[AgnoInstrumentor()])

dotenv.load_dotenv()

SYSTEM_PROMPT = """
You are a customer support agent for SecureBank, a modern digital banking platform.

Your role is to help customers with their banking needs professionally and efficiently. You have access to specialized tools that MUST be used in specific situations:

TOOL USAGE REQUIREMENTS:

1. **explore_customer_account** - ALWAYS use when:
   - Customer mentions fraud, unauthorized transactions, or security concerns
   - Customer asks about spending patterns, budgeting, or financial analysis
   - Customer needs account-specific insights or personalized recommendations
   - Any urgent business account issues that need immediate investigation

2. **get_message_suggestion** - ALWAYS use when:
   - Customer has complex, multi-part problems (locked accounts + fees + missing deposits)
   - You need guidance on complex banking regulations or procedures
   - Customer issue involves multiple interconnected banking services

3. **escalate_to_human** - ALWAYS use when:
   - Customer explicitly demands to speak with a manager, supervisor, or human agent
   - Customer expresses extreme frustration or dissatisfaction
   - Business customer has urgent issues affecting operations (payroll, employee payments)
   - Set urgency to "high" for business-critical issues

4. **get_conversation_summary** - Use when:
   - Customer asks you to summarize the conversation
   - You need to analyze conversation patterns or sentiment

CRITICAL: For simple questions like service hours, do NOT use unnecessary tools. Respond directly.

Guidelines:
- Be helpful, professional, and empathetic
- Use tools proactively based on the requirements above
- Provide clear, actionable solutions
- Always prioritize customer security and privacy

Remember: Tool usage is not optional when the situation matches the requirements above.
"""


def get_conversation_summary(conversation_context: str = "recent messages") -> str:
    """
    Analyze the conversation for patterns, sentiment, and key issues

    Args:
        conversation_context: Context about what to analyze

    Returns:
        JSON string with conversation analysis
    """
    langwatch.get_current_trace().update(
        metadata={"labels": ["tool_get_conversation_summary"]}
    )
    # In a real implementation, this would get the actual conversation history
    # For now, we'll simulate with a basic response
    return json.dumps(
        {
            "summary": "Conversation analysis requested",
            "sentiment": "neutral",
            "key_issues": ["general inquiry"],
            "suggested_actions": ["continue conversation"],
        }
    )


def get_message_suggestion(customer_query: str, context: str = "") -> str:
    """
    Get suggestions for responding to customer queries using knowledge base

    Args:
        customer_query: The customer's question or concern
        context: Additional context about the conversation

    Returns:
        JSON string with response suggestions
    """
    langwatch.get_current_trace().update(
        metadata={"labels": ["tool_get_message_suggestion"]}
    )
    # Simulate knowledge base lookup
    suggestion_data = {
        "suggested_response": f"I understand your concern about: {customer_query}. Let me help you with that.",
        "confidence": "medium",
        "knowledge_sources": ["general_banking_guide"],
        "alternatives": ["Ask for more details", "Escalate to specialist"],
    }
    return json.dumps(suggestion_data)


def explore_customer_account(customer_id: str, query: str) -> str:
    """
    Explore customer account data and provide rich insights

    Args:
        customer_id: Customer identifier (e.g., CUST_001)
        query: What to explore about the customer

    Returns:
        JSON string with customer insights and rich experiences
    """
    # Get customer behavior analysis
    behavior = analyze_customer_behavior(customer_id)

    # Get rich experiences based on query
    rich_experiences = explore_customer_context(customer_id, query)

    langwatch.get_current_trace().update(
        metadata={"labels": ["tool_explore_customer_account"]}
    )

    return json.dumps(
        {
            "customer_behavior": behavior,
            "rich_experiences": [
                {
                    "type": exp.component_type,
                    "title": exp.title,
                    "data": exp.data,
                    "actions": exp.actions,
                    "priority": exp.priority,
                }
                for exp in rich_experiences
            ],
        }
    )


def escalate_to_human(reason: str, urgency: str = "medium") -> str:
    """
    Escalate the conversation to a human agent

    Args:
        reason: Why the escalation is needed
        urgency: Priority level (low, medium, high)

    Returns:
        JSON string with escalation details
    """
    langwatch.get_current_trace().update(metadata={"labels": ["escalation"]})

    escalation_data = {
        "escalated": True,
        "reason": reason,
        "urgency": urgency,
        "estimated_wait": "5-10 minutes" if urgency == "high" else "10-15 minutes",
        "message": "I'm connecting you with a specialist who can provide additional assistance.",
    }
    return json.dumps(escalation_data)


# Create the main support agent
support_agent = Agent(
    name="BankCustomerSupportAgent",
    # model=OpenAIChat(
    #     id="gpt-4o-mini",
    #     api_key=os.getenv("OPENAI_API_KEY"),
    # ),
    model=Nebius(
        id="openai/gpt-oss-120b",
        api_key=os.getenv("NEBIUS_API_KEY"),
    ),
    tools=[
        get_conversation_summary,
        get_message_suggestion,
        explore_customer_account,
        escalate_to_human,
    ],
    description=SYSTEM_PROMPT,
    add_history_to_context=True,  # Let Agno handle memory
)


# Simple interface for testing
def chat_with_agent(message: str) -> str:
    """Simple interface to chat with the agent"""
    response = support_agent.run(message)
    return response.content


# Example usage
if __name__ == "__main__":
    print("=== Bank Customer Support Agent ===")
    print(
        "Agent: Hello! I'm here to help with your banking needs. How can I assist you today?"
    )

    # Simulate a conversation
    customer_message = "Hi, I'm seeing some transactions on my account that I don't recognize. I'm worried about fraud."
    print(f"\nCustomer: {customer_message}")

    response = chat_with_agent(customer_message)
    print(f"Agent: {response}")

    # Continue conversation
    customer_message2 = "Yes, there's an $85 charge from Amazon and a $45 gas station charge. Can you help me freeze my card?"
    print(f"\nCustomer: {customer_message2}")

    response2 = chat_with_agent(customer_message2)
    print(f"Agent: {response2}")

    print("\n=== Conversation Complete ===")
