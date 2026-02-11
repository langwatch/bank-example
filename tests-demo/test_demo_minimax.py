"""
Tests for the main bank customer support agent - MiniMax Model

These tests cover real business scenarios and validate tool calling behavior
using Nebius MiniMax-M2.1 model for evaluation.
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
scenario.configure(default_model="nebius/MiniMaxAI/MiniMax-M2.1")


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
async def test_comprehensive_tool_coordination():
    """Test a scenario that uses multiple tools in sequence"""

    # Track which tools were called
    tools_called = []

    def track_customer_exploration(state: scenario.ScenarioState):
        if state.has_tool_call("explore_customer_account"):
            tools_called.append("explore_customer_account")

    def track_message_suggestion(state: scenario.ScenarioState):
        if state.has_tool_call("get_message_suggestion"):
            tools_called.append("get_message_suggestion")

    def track_conversation_summary(state: scenario.ScenarioState):
        if state.has_tool_call("get_conversation_summary"):
            tools_called.append("get_conversation_summary")

    def validate_tool_coordination(state: scenario.ScenarioState):
        """Ensure agent used appropriate tools throughout the conversation"""
        # Should have used customer exploration for account analysis
        assert (
            "explore_customer_account" in tools_called
        ), "Agent should explore customer account for spending analysis"

        # Verify the conversation has good depth (multiple exchanges)
        user_messages = [m for m in state.messages if m["role"] == "user"]
        agent_messages = [m for m in state.messages if m["role"] == "assistant"]
        assert len(user_messages) >= 3, "Conversation should have multiple user turns"
        assert len(agent_messages) >= 3, "Agent should respond multiple times"

    result = await scenario.run(
        name="comprehensive account analysis and advice - MiniMax",
        description="""
            Customer wants to understand their spending patterns and get financial advice.
            This requires account exploration, potentially knowledge base guidance,
            and possibly conversation analysis. The agent should coordinate multiple tools effectively.
        """,
        agents=[
            BankSupportAgentAdapter(),
            scenario.UserSimulatorAgent(),
            scenario.JudgeAgent(
                criteria=[
                    "Agent provides personalized insights based on account data",
                    "Agent offers actionable financial recommendations",
                    "Agent asks relevant follow-up questions",
                    "Agent coordinates multiple information sources effectively",
                ]
            ),
        ],
        script=[
            scenario.user(
                "I want to get better at managing my money. Can you analyze my spending and help me understand where I can improve?"
            ),
            scenario.agent(),
            track_customer_exploration,
            scenario.user(
                "That's helpful! Can you also suggest a realistic budget based on my spending patterns and give me specific advice?"
            ),
            scenario.agent(),
            track_message_suggestion,
            scenario.user(
                "This conversation has been really valuable. Can you summarize the key insights and recommendations we discussed?"
            ),
            scenario.agent(),
            track_conversation_summary,
            validate_tool_coordination,
            scenario.judge(),
        ],
    )

    assert result.success, f"Tool coordination test failed: {result.failure_reason}"  # type: ignore


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
        name="customer escalation to human agent - MiniMax",
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
async def test_tool_precision_simple_query():
    """Test that agent doesn't over-use tools for simple queries"""

    def verify_minimal_tool_usage(state: scenario.ScenarioState):
        """Ensure agent doesn't call unnecessary tools for simple questions"""
        # Count total tool calls
        tool_calls = 0
        for message in state.messages:
            if message["role"] == "assistant" and "tool_calls" in message:
                tool_calls += len(message["tool_calls"])  # type: ignore

        # For simple service hours question, should use minimal or no tools
        assert (
            tool_calls <= 1
        ), f"Agent should use minimal tools for simple queries, but used {tool_calls} tool calls"

    result = await scenario.run(
        name="simple service hours inquiry - MiniMax",
        description="""
            Customer asks a simple question about service hours.
            This should not require complex tool usage or analysis.
            Agent should respond directly and efficiently.
        """,
        agents=[
            BankSupportAgentAdapter(),
            scenario.UserSimulatorAgent(),
            scenario.JudgeAgent(
                criteria=[
                    "Agent responds directly to simple questions",
                    "Agent provides clear and helpful information",
                    "Agent doesn't over-complicate simple interactions",
                    "Agent maintains friendly and professional tone",
                ]
            ),
        ],
        script=[
            scenario.user("What are your customer service hours?"),
            scenario.agent(),
            verify_minimal_tool_usage,
            scenario.user("Thank you, that's helpful."),
            scenario.agent(),
            scenario.judge(),
        ],
    )

    assert result.success, f"Tool precision test failed: {result.failure_reason}"  # type: ignore


if __name__ == "__main__":
    asyncio.run(test_comprehensive_tool_coordination())
