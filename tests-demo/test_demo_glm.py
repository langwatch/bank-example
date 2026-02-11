"""
Tests for the main bank customer support agent - GLM Model

These tests cover real business scenarios and validate tool calling behavior
using Nebius GLM-4.7-FP8 model for evaluation.
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
scenario.configure(default_model="nebius/zai-org/GLM-4.7-FP8")


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
        name="fraud investigation and card security - GLM",
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
        name="customer escalation to human agent - GLM",
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
async def test_urgent_business_scenario():
    def check_appropriate_urgency_response(state: scenario.ScenarioState):
        """Verify agent responds appropriately to business urgency"""
        # For urgent business issues, agent should either:
        # 1. Escalate immediately, OR
        # 2. Use customer exploration to provide immediate solutions
        has_escalation = state.has_tool_call("escalate_to_human")
        has_exploration = state.has_tool_call("explore_customer_account")

        assert (
            has_escalation or has_exploration
        ), "Agent should either escalate urgent business issues or explore customer account for immediate solutions"

        # Check that urgency is reflected in tool call parameters
        if has_escalation:
            tool_call = state.last_tool_call("escalate_to_human")
            if tool_call:
                args = json.loads(tool_call["function"]["arguments"])
                urgency = args.get("urgency", "medium")
                assert (
                    urgency == "high"
                ), "Business urgency should be marked as high priority"

    result = await scenario.run(
        name="urgent business account problem - GLM",
        description="""
            Business customer has an urgent issue affecting their operations.
            They can't access funds to pay employees. This requires immediate
            attention and appropriate priority handling.
        """,
        agents=[
            BankSupportAgentAdapter(),
            scenario.UserSimulatorAgent(),
            scenario.JudgeAgent(
                criteria=[
                    "Agent recognizes the business urgency and impact",
                    "Agent treats the issue with appropriate priority",
                    "Agent offers immediate assistance or escalation",
                    "Agent provides clear timeline for resolution",
                ]
            ),
        ],
        script=[
            scenario.user(
                "URGENT: My business account is frozen and I need to pay my employees today. This is costing me money every minute!"
            ),
            scenario.agent(),
            check_appropriate_urgency_response,
            scenario.user(
                "I can't wait. My payroll is due in 2 hours and my employees are depending on me. What can you do right now?"
            ),
            scenario.agent(),
            scenario.judge(),
        ],
    )

    assert result.success, f"Urgent business test failed: {result.failure_reason}"  # type: ignore


if __name__ == "__main__":
    asyncio.run(test_fraud_investigation_workflow())
