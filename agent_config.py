"""
Shared agent model configuration.

Each main_support_agent_*.py sets the model before agents are used.
Sub-agents read from here to stay in sync.
"""
import os
from agno.models.nebius import Nebius

_current_model = None


def set_model(model):
    global _current_model
    _current_model = model


def get_model():
    if _current_model:
        return _current_model
    return Nebius(
        id="openai/gpt-oss-120b",
        api_key=os.getenv("NEBIUS_API_KEY"),
    )
