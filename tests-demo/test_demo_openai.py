"""
Tests for the main bank customer support agent - OpenAI Model

These tests cover real business scenarios and validate tool calling behavior
using OpenAI gpt-4o-mini model for evaluation.
"""
import asyncio
import pytest
import json
import sys
import os
import dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scenario
from main_support_agent import support_agent

dotenv.load_dotenv()
scenario.configure(default_model="openai/gpt-4o-mini")


class BankSupportAgentAdapter(scenario.AgentAdapter):
    """Adapter for our main bank support agent"""

    async def call(self, input: scenario.AgentInput) -> scenario.AgentReturnTypes:
        message_content = input.last_new_user_message_str()
        response = support_agent.run(message_content)

        # Convert Agno messages to OpenAI format for Scenario
        openai_messages = []
        for message in response.messages or []:
            if message.role in ["assistant", "user", "system", "tool"]:
                msg_dict = {"role": message.role, "content": message.content}

                # Add tool calls if present (for assistant messages)
                if message.tool_calls:
                    msg_dict["tool_calls"] = message.tool_calls

                # Add tool call ID if present (for tool messages)
                if hasattr(message, "tool_call_id") and message.tool_call_id:
                    msg_dict["tool_call_id"] = message.tool_call_id

                openai_messages.append(msg_dict)

        # Return all messages except system and user (Scenario manages the conversation flow)
        # We need to include tool messages to satisfy OpenAI's requirements
        relevant_messages = [
            msg for msg in openai_messages if msg["role"] in ["assistant", "tool"]
        ]

        if relevant_messages:
            return relevant_messages

        # Fallback to content if no relevant messages found
        return response.content  # type: ignore


@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_fraud_investigation_workflow():
    # Custom assertions for tool calling validation
    def check_customer_exploration_called(state: scenario.ScenarioState):
        """Verify the agent called explore_customer_account for fraud investigation"""
        assert state.has_tool_call(
            "explore_customer_account"
        ), "Agent should call explore_customer_account for fraud concerns"

        # Check the tool was called with appropriate parameters
        tool_call = state.last_tool_call("explore_customer_account")
        if tool_call:
            args = json.loads(tool_call["function"]["arguments"])
            assert "customer_id" in args, "Tool call should include customer_id"

    def verify_no_inappropriate_tools(state: scenario.ScenarioState):
        """Ensure agent doesn't use inappropriate tools for fraud scenarios"""
        # Should not use message suggestions for clear security issues
        assert not state.has_tool_call(
            "get_message_suggestion"
        ), "Agent should not need message suggestions for clear fraud cases"

    result = await scenario.run(
        name="fraud investigation and card security - OpenAI",
        description="""
            Customer discovers unauthorized transactions on their account and is worried about fraud.
            They need immediate help to secure their account and investigate the suspicious activity.
            The agent should use customer exploration tools to analyze the account.
        """,
        agents=[
            BankSupportAgentAdapter(),
            scenario.UserSimulatorAgent(),
            scenario.JudgeAgent(
                criteria=[
                    "Agent takes fraud concerns seriously and responds with urgency",
                    "Agent offers concrete security actions like card freezing",
                    "Agent provides clear next steps for fraud investigation",
                    "Agent maintains professional and reassuring tone",
                ]
            ),
        ],
        script=[
            scenario.user(
                "Hi, I just checked my account and there are transactions I didn't make. I think my card was stolen!"
            ),
            scenario.agent(),
            check_customer_exploration_called,
            scenario.user(
                "There's an $85 charge at Amazon and a $45 charge at some gas station. I definitely didn't make these purchases."
            ),
            scenario.agent(),
            scenario.user(
                "Yes, please help me secure my account right away. I'm worried about more charges."
            ),
            scenario.agent(),
            verify_no_inappropriate_tools,
            scenario.judge(),
        ],
    )

    assert result.success, f"Fraud investigation test failed: {result.failure_reason}"  # type: ignore


@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_escalation_workflow():
    def check_escalation_called(state: scenario.ScenarioState):
        """Verify agent escalates when customer explicitly demands human help"""
        assert state.has_tool_call(
            "escalate_to_human"
        ), "Agent should escalate when customer demands manager/human help"

        tool_call = state.last_tool_call("escalate_to_human")
        if tool_call:
            args = json.loads(tool_call["function"]["arguments"])
            reason = args.get("reason", "").lower()
            assert any(
                keyword in reason
                for keyword in ["frustrated", "manager", "human", "escalation"]
            ), "Escalation reason should reflect customer's frustration and demand"

    result = await scenario.run(
        name="customer escalation to human agent - OpenAI",
        description="""
            Customer has been dealing with an ongoing issue and is frustrated.
            They explicitly demand to speak with a human agent or manager.
            The agent should handle this professionally and escalate appropriately.
        """,
        agents=[
            BankSupportAgentAdapter(),
            scenario.UserSimulatorAgent(),
            scenario.JudgeAgent(
                criteria=[
                    "Agent acknowledges customer's frustration empathetically",
                    "Agent offers to escalate when requested",
                    "Agent provides escalation timeline and process information",
                    "Agent maintains professionalism despite customer frustration",
                ]
            ),
        ],
        script=[
            scenario.user(
                "I've been calling about this same issue for two weeks and nobody can fix it. I want to speak to a real person who can actually help me!"
            ),
            scenario.agent(),
            scenario.user(
                "No more troubleshooting! I want a manager or supervisor right now. This is unacceptable service."
            ),
            scenario.agent(),
            check_escalation_called,
            scenario.judge(),
        ],
    )

    assert result.success, f"Escalation test failed: {result.failure_reason}"  # type: ignore


@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_complex_issue_triggers_knowledge_base():
    def check_message_suggestion_called(state: scenario.ScenarioState):
        """Verify agent uses knowledge base for complex multi-part issues"""
        assert state.has_tool_call(
            "get_message_suggestion"
        ), "Agent should use message suggestions for complex banking issues"

        tool_call = state.last_tool_call("get_message_suggestion")
        if tool_call:
            args = json.loads(tool_call["function"]["arguments"])
            query = args.get("customer_query", "").lower()
            assert any(
                keyword in query for keyword in ["lock", "fee", "deposit", "multiple"]
            ), "Tool call should reference the customer's specific issues"

    result = await scenario.run(
        name="complex multi-issue banking problem - OpenAI",
        description="""
            Customer has multiple interconnected banking problems: locked online banking,
            unexpected fees, and missing direct deposit. They need systematic help
            and the agent should use knowledge base guidance.
        """,
        agents=[
            BankSupportAgentAdapter(),
            scenario.UserSimulatorAgent(),
            scenario.JudgeAgent(
                criteria=[
                    "Agent addresses all parts of the multi-faceted problem",
                    "Agent provides systematic approach to resolving issues",
                    "Agent shows empathy for customer frustration",
                    "Agent offers clear next steps for each problem",
                ]
            ),
        ],
        script=[
            scenario.user(
                "I have multiple problems with my account. My online banking is locked, there's a $35 fee I don't understand, and my paycheck didn't deposit."
            ),
            scenario.agent(),
            check_message_suggestion_called,
            scenario.user(
                "I've tried resetting my password multiple times and I really need access to pay my bills. This is really stressing me out."
            ),
            scenario.agent(),
            scenario.judge(),
        ],
    )

    assert (
        result.success
    ), f"Complex issue test failed: {result.reasoning if hasattr(result, 'reasoning') else 'No failure reason available'}"


if __name__ == "__main__":
    asyncio.run(test_fraud_investigation_workflow())
