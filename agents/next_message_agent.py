"""
Next Message Agent - Suggests next replies using knowledge base
"""
import os
from typing import List, Dict, Any, Optional
import dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.nebius import Nebius
from pydantic import BaseModel
import agent_config

dotenv.load_dotenv()

class NextMessageSuggestion(BaseModel):
    """Suggested next message with confidence and reasoning"""
    suggested_message: str
    confidence_level: str  # high, medium, low
    reasoning: str
    alternative_approaches: List[str]
    requires_escalation: bool
    knowledge_sources_used: List[str]

# Mock knowledge base - in real implementation this would be a vector database
BANKING_KNOWLEDGE_BASE = {
    "login_issues": {
        "title": "Online Banking Login Problems",
        "content": """
        Common solutions for login issues:
        1. Check if caps lock is on
        2. Try resetting password using 'Forgot Password' link
        3. Clear browser cache and cookies
        4. Try a different browser or incognito mode
        5. Account may be temporarily locked after multiple failed attempts
        6. Contact support if issue persists after 24 hours
        """,
        "escalation_triggers": ["multiple failed attempts", "account locked", "security concerns"]
    },
    "account_balance": {
        "title": "Account Balance Inquiries",
        "content": """
        For balance inquiries:
        1. Balance updates may take 1-2 business days for recent transactions
        2. Pending transactions are not included in available balance
        3. Check transaction history for detailed breakdown
        4. Contact support for discrepancies older than 5 business days
        """,
        "escalation_triggers": ["missing transactions", "incorrect balance", "fraud concerns"]
    },
    "card_issues": {
        "title": "Debit/Credit Card Problems",
        "content": """
        Card issue resolution steps:
        1. Check if card is expired or damaged
        2. Verify PIN if transaction was declined
        3. Check daily spending limits
        4. Report lost/stolen cards immediately
        5. Request replacement card if needed
        6. Temporary freeze available through mobile app
        """,
        "escalation_triggers": ["fraud", "unauthorized transactions", "lost card", "stolen card"]
    },
    "transfer_issues": {
        "title": "Money Transfer Problems",
        "content": """
        Transfer troubleshooting:
        1. Verify recipient account details
        2. Check available balance including holds
        3. Confirm transfer limits haven't been exceeded
        4. International transfers may take 3-5 business days
        5. Check for any security holds on account
        """,
        "escalation_triggers": ["failed transfer", "missing money", "international issues"]
    }
}

NEXT_MESSAGE_SYSTEM_PROMPT = """
You are a specialized AI agent that suggests appropriate next messages for bank customer support agents.

Your task is to:
1. Analyze the conversation context and customer's current issue
2. Suggest the most appropriate next response
3. Provide reasoning for your suggestion
4. Assess confidence level in your recommendation
5. Offer alternative approaches
6. Determine if escalation is needed
7. Reference relevant knowledge base articles

Guidelines:
- Be empathetic and professional
- Provide clear, actionable solutions
- Escalate when necessary (security issues, complex problems, angry customers)
- Use banking knowledge base to inform responses
- Consider customer's emotional state
- Offer step-by-step instructions when appropriate
- Always maintain customer privacy and security

Your suggestions should help agents provide consistent, high-quality support.
"""

def get_relevant_knowledge(customer_message: str, conversation_context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Get relevant knowledge base articles based on customer message and context

    Args:
        customer_message: The latest customer message
        conversation_context: Previous messages in the conversation

    Returns:
        List of relevant knowledge base articles
    """
    # Simple keyword matching - in production this would use vector similarity
    relevant_articles = []

    combined_text = customer_message.lower()
    for msg in conversation_context[-3:]:  # Look at last 3 messages for context
        combined_text += " " + msg.get('content', '').lower()

    for key, article in BANKING_KNOWLEDGE_BASE.items():
        # Check if any keywords match
        keywords = key.replace('_', ' ').split()
        if any(keyword in combined_text for keyword in keywords):
            relevant_articles.append({
                "id": key,
                "title": article["title"],
                "content": article["content"],
                "escalation_triggers": article["escalation_triggers"]
            })

    return relevant_articles

def create_next_message_agent() -> Agent:
    """Create and return the next message agent"""
    return Agent(
        name="NextMessageAgent",
        model=agent_config.get_model(),
        description=NEXT_MESSAGE_SYSTEM_PROMPT,
        add_history_to_context=True,
    )

def suggest_next_message(
    customer_message: str,
    conversation_history: List[Dict[str, Any]],
    customer_context: Optional[Dict[str, Any]] = None
) -> NextMessageSuggestion:
    """
    Suggest the next message for a customer support agent

    Args:
        customer_message: The latest message from the customer
        conversation_history: Previous messages in the conversation
        customer_context: Additional context about the customer (account info, etc.)

    Returns:
        NextMessageSuggestion with recommended response
    """
    agent = create_next_message_agent()

    # Get relevant knowledge base articles
    relevant_knowledge = get_relevant_knowledge(customer_message, conversation_history)

    # Format conversation history
    formatted_history = ""
    for i, msg in enumerate(conversation_history[-5:], 1):  # Last 5 messages for context
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        formatted_history += f"{role.title()}: {content}\n"

    # Format knowledge base articles
    knowledge_context = ""
    if relevant_knowledge:
        knowledge_context = "\n\nRelevant Knowledge Base Articles:\n"
        for article in relevant_knowledge:
            knowledge_context += f"\n{article['title']}:\n{article['content']}\n"
            knowledge_context += f"Escalation triggers: {', '.join(article['escalation_triggers'])}\n"

    # Format customer context if available
    customer_info = ""
    if customer_context:
        customer_info = f"\nCustomer Context: {customer_context}\n"

    prompt = f"""
    Based on the following conversation and knowledge base, suggest the best next response for the customer support agent.

    Conversation History:
    {formatted_history}

    Latest Customer Message: {customer_message}
    {customer_info}
    {knowledge_context}

    Please provide your suggestion in the following JSON format:
    {{
        "suggested_message": "Your suggested response to the customer",
        "confidence_level": "high/medium/low",
        "reasoning": "Why this is the best response",
        "alternative_approaches": ["alternative1", "alternative2"],
        "requires_escalation": true/false,
        "knowledge_sources_used": ["source1", "source2"]
    }}
    """

    response = agent.run(prompt)

    # Parse the response
    try:
        import json
        suggestion_data = json.loads(response.content)
        return NextMessageSuggestion(**suggestion_data)
    except (json.JSONDecodeError, Exception) as e:
        # Fallback if JSON parsing fails
        return NextMessageSuggestion(
            suggested_message=response.content[:300] + "..." if len(response.content) > 300 else response.content,
            confidence_level="medium",
            reasoning="Generated response based on conversation context",
            alternative_approaches=["Escalate to human agent", "Request additional information"],
            requires_escalation=False,
            knowledge_sources_used=["general_guidelines"]
        )

# Example usage for testing
if __name__ == "__main__":
    conversation_history = [
        {
            "role": "customer",
            "content": "Hi, I'm having trouble logging into my online banking account.",
            "timestamp": "2024-01-15 10:30:00"
        },
        {
            "role": "agent",
            "content": "I'm sorry to hear you're having login troubles. I'd be happy to help you resolve this issue.",
            "timestamp": "2024-01-15 10:31:00"
        }
    ]

    customer_message = "I keep getting an 'invalid credentials' error even though I'm sure my password is correct. This is really frustrating!"

    suggestion = suggest_next_message(customer_message, conversation_history)
    print(f"Suggested Message: {suggestion.suggested_message}")
    print(f"Confidence: {suggestion.confidence_level}")
    print(f"Reasoning: {suggestion.reasoning}")
    print(f"Requires Escalation: {suggestion.requires_escalation}")
