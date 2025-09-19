"""
Tests for the Next Message Agent

Business scenarios where message suggestions and knowledge base guidance are needed.
"""
import pytest
import os
import sys
import dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scenario
from agents.next_message_agent import suggest_next_message

dotenv.load_dotenv()
scenario.configure(default_model="openai/gpt-4o-mini")

class NextMessageAdapter(scenario.AgentAdapter):
    """Adapter for testing message suggestions"""

    async def call(self, input: scenario.AgentInput) -> scenario.AgentReturnTypes:
        last_message = input.messages[-1]["content"] if input.messages else ""
        conversation_history = [{"role": "customer", "content": msg["content"]} for msg in input.messages[:-1]]

        suggestion = suggest_next_message(last_message, conversation_history)

        response = f"Suggested Response: {suggestion.suggested_message}\n"
        response += f"Confidence: {suggestion.confidence_level}\n"
        response += f"Escalation Needed: {suggestion.requires_escalation}\n"
        response += f"Knowledge Sources: {', '.join(suggestion.knowledge_sources_used)}"

        return response

@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_complex_banking_issue_guidance():
    result = await scenario.run(
        name="complex banking issue resolution guidance",
        description="Customer has complex multi-step banking problem needing expert guidance",
        agents=[
            NextMessageAdapter(),
            scenario.UserSimulatorAgent(),
            scenario.JudgeAgent(criteria=[
                "Suggestion provides structured solution approach",
                "Suggestion uses relevant banking knowledge",
                "Suggestion addresses problem complexity appropriately",
                "Suggestion offers clear next steps"
            ])
        ],
        script=[
            scenario.user("My online banking is locked, I have overdraft fees, and my direct deposit failed. I don't know where to start."),
            scenario.agent(),
            scenario.judge(),
        ],
    )

    assert result.success

@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_escalation_recommendation():
    result = await scenario.run(
        name="escalation recommendation for angry customer",
        description="Customer is very frustrated and may need human assistance",
        agents=[
            NextMessageAdapter(),
            scenario.UserSimulatorAgent(),
            scenario.JudgeAgent(criteria=[
                "Suggestion recognizes need for escalation",
                "Suggestion maintains professional tone",
                "Suggestion offers appropriate escalation path",
                "Suggestion shows empathy for frustration"
            ])
        ],
        script=[
            scenario.user("I've called 5 times about this and nobody can help! This is ridiculous!"),
            scenario.agent(),
            scenario.judge(),
        ],
    )

    assert result.success

if __name__ == "__main__":
    import asyncio

    async def run_test():
        print("Running next message agent test...")
        await test_complex_banking_issue_guidance()
        print("Test completed!")

    asyncio.run(run_test())