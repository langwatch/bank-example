"""
Summary Agent - Takes message threads and summarizes them with sentiment analysis
"""
import os
from typing import List, Dict, Any
import dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.nebius import Nebius
from pydantic import BaseModel
import agent_config

dotenv.load_dotenv()

class MessageSummary(BaseModel):
    """Summary of a message thread with sentiment analysis"""
    summary: str
    sentiment: str  # positive, negative, neutral
    key_issues: List[str]
    customer_satisfaction_level: str  # high, medium, low
    urgency_level: str  # high, medium, low
    suggested_actions: List[str]

SUMMARY_SYSTEM_PROMPT = """
You are a specialized AI agent that analyzes customer support message threads and provides comprehensive summaries.

Your task is to:
1. Summarize the key points of the conversation
2. Analyze the customer's sentiment (positive, negative, neutral)
3. Identify key issues and concerns
4. Assess customer satisfaction level (high, medium, low)
5. Determine urgency level (high, medium, low)
6. Suggest actions for the support team

Guidelines:
- Be concise but comprehensive
- Focus on actionable insights
- Identify patterns in customer behavior
- Highlight any escalation triggers
- Consider the customer's emotional state
- Extract specific problems mentioned
- Note any successful resolutions

Always respond with a structured summary that helps support agents understand the context quickly.
"""

def create_summary_agent() -> Agent:
    """Create and return the summary agent"""
    return Agent(
        name="SummaryAgent",
        model=agent_config.get_model(),
        description=SUMMARY_SYSTEM_PROMPT,
        add_history_to_context=True,
    )

def summarize_conversation(messages: List[Dict[str, Any]]) -> MessageSummary:
    """
    Summarize a conversation thread with sentiment analysis

    Args:
        messages: List of message objects with 'role', 'content', and 'timestamp'

    Returns:
        MessageSummary object with comprehensive analysis
    """
    agent = create_summary_agent()

    # Format messages for analysis
    formatted_conversation = ""
    for i, msg in enumerate(messages, 1):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        timestamp = msg.get('timestamp', 'unknown time')
        formatted_conversation += f"Message {i} ({role} at {timestamp}):\n{content}\n\n"

    prompt = f"""
    Please analyze the following customer support conversation and provide a comprehensive summary:

    {formatted_conversation}

    Provide your analysis in the following JSON format:
    {{
        "summary": "Brief overview of the conversation",
        "sentiment": "positive/negative/neutral",
        "key_issues": ["issue1", "issue2", ...],
        "customer_satisfaction_level": "high/medium/low",
        "urgency_level": "high/medium/low",
        "suggested_actions": ["action1", "action2", ...]
    }}
    """

    response = agent.run(prompt)

    # Parse the response and create MessageSummary
    try:
        import json
        summary_data = json.loads(response.content)
        return MessageSummary(**summary_data)
    except (json.JSONDecodeError, Exception) as e:
        # Fallback if JSON parsing fails
        return MessageSummary(
            summary=response.content[:200] + "..." if len(response.content) > 200 else response.content,
            sentiment="neutral",
            key_issues=["Analysis parsing error"],
            customer_satisfaction_level="medium",
            urgency_level="medium",
            suggested_actions=["Manual review required"]
        )

# Example usage for testing
if __name__ == "__main__":
    sample_messages = [
        {
            "role": "customer",
            "content": "Hi, I'm having trouble with my online banking. I can't log in and I'm getting frustrated.",
            "timestamp": "2024-01-15 10:30:00"
        },
        {
            "role": "agent",
            "content": "I'm sorry to hear about the login issues. Let me help you resolve this. Can you tell me what error message you're seeing?",
            "timestamp": "2024-01-15 10:31:00"
        },
        {
            "role": "customer",
            "content": "It says 'Invalid credentials' but I'm sure my password is correct. This is really annoying!",
            "timestamp": "2024-01-15 10:32:00"
        }
    ]

    summary = summarize_conversation(sample_messages)
    print(f"Summary: {summary.summary}")
    print(f"Sentiment: {summary.sentiment}")
    print(f"Key Issues: {summary.key_issues}")
    print(f"Urgency: {summary.urgency_level}")
