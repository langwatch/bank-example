"""
Bank Customer Support Agents

This module contains specialized AI agents for bank customer support:
- Summary Agent: Analyzes conversation threads and provides sentiment analysis
- Next Message Agent: Suggests appropriate responses using knowledge base
- Customer Explorer Agent: Provides rich customer data experiences
"""

from .summary_agent import summarize_conversation, MessageSummary
from .next_message_agent import suggest_next_message, NextMessageSuggestion
from .customer_explorer_agent import explore_customer_context, analyze_customer_behavior, RichExperience

__all__ = [
    "summarize_conversation",
    "MessageSummary",
    "suggest_next_message",
    "NextMessageSuggestion",
    "explore_customer_context",
    "analyze_customer_behavior",
    "RichExperience"
]
