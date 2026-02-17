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
from main_support_agent_glm import support_agent

dotenv.load_dotenv()
scenario.configure(default_model="openai/gpt-4o")


def _parse_tool_arguments(tool_call: dict) -> dict:
    raw_args = tool_call["function"].get("arguments", {})
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            return {}
    return {}


def _assert_success(result: scenario.ScenarioResult, test_name: str) -> None:
    assert result.success, f"{test_name} failed: {result.reasoning or 'No failure reasoning returned'}"


def _build_tool_trace_messages(response) -> list[dict]:
    messages: list[dict] = []
    for i, tool in enumerate(response.tools or []):
        tool_call_id = tool.tool_call_id or f"tool_call_{i}"
        tool_name = tool.tool_name or "unknown_tool"
        tool_args = tool.tool_args if isinstance(tool.tool_args, dict) else {}
        tool_result = tool.result if isinstance(tool.result, str) else json.dumps(tool.result or {})

        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": tool_result,
            }
        )

    if isinstance(response.content, str) and response.content:
        messages.append({"role": "assistant", "content": response.content})

    return messages


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
            has_tool_calls = any(
                msg["role"] == "assistant" and "tool_calls" in msg for msg in relevant_messages
            )
            if has_tool_calls:
                return relevant_messages

        synthetic_messages = _build_tool_trace_messages(response)
        if synthetic_messages:
            return synthetic_messages

        # Fallback to content if no relevant messages found
        return response.content  # type: ignore


@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_fraud_investigation_workflow():
    result = await scenario.run(
        name="fraud investigation and card security - GLM",
        description="""
            Customer discovers unauthorized transactions on their account and is worried about fraud.
            They need immediate help to secure their account and investigate the suspicious activity.
            This tests whether the agent responds with appropriate urgency and offers concrete security actions.
        """,
        agents=[
            BankSupportAgentAdapter(),
            scenario.UserSimulatorAgent(model="openai/gpt-4o"),
            scenario.JudgeAgent(
                model="openai/gpt-4o",
                criteria=[
                    "Agent takes fraud concerns seriously and responds with urgency",
                    "Agent gathers necessary information (account details) to investigate",
                    "Agent offers concrete security actions like card freezing or blocking",
                    "Agent provides clear next steps for fraud investigation and dispute process",
                    "Agent maintains professional and reassuring tone throughout",
                    "Agent does not re-ask for customer ID that was already provided",
                ]
            ),
        ],
        script=[
            scenario.user(
                "Hi, I just checked my account and there are transactions I didn't make. I think my card was stolen!"
            ),
            scenario.agent(),
            scenario.user(
                "My customer ID is CUST_001. There's an $85 charge at Amazon and a $45 charge at some gas station yesterday. I definitely didn't make these purchases."
            ),
            scenario.agent(),
            scenario.user(
                "Yes, please help me secure my account right away. I'm really worried about more charges appearing."
            ),
            scenario.agent(),
            scenario.judge(),
        ],
    )

    _assert_success(result, "Fraud investigation test")


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
            args = _parse_tool_arguments(tool_call)
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
            scenario.UserSimulatorAgent(model="openai/gpt-4o"),
            scenario.JudgeAgent(
                model="openai/gpt-4o",
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

    _assert_success(result, "Escalation test")


@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_complex_issue_triggers_knowledge_base():
    result = await scenario.run(
        name="complex multi-issue banking problem - GLM",
        description="""
            Customer has multiple interconnected banking problems: locked online banking,
            unexpected fees, and missing direct deposit. They need systematic help.
            This tests whether the agent can handle multiple issues comprehensively.
        """,
        agents=[
            BankSupportAgentAdapter(),
            scenario.UserSimulatorAgent(model="openai/gpt-4o"),
            scenario.JudgeAgent(
                model="openai/gpt-4o",
                criteria=[
                    "Agent acknowledges ALL three issues (locked banking, fee, missing deposit)",
                    "Agent provides systematic approach with clear steps for each issue",
                    "Agent shows empathy for customer's stress and urgency",
                    "Agent prioritizes the most urgent issue (locked account for bill payments)",
                    "Agent offers concrete next steps that the customer can act on",
                ]
            ),
        ],
        script=[
            scenario.user(
                "I have multiple problems with my account. My online banking is locked, there's a $35 fee I don't understand, and my paycheck didn't deposit. My customer ID is CUST_001."
            ),
            scenario.agent(),
            scenario.user(
                "I've tried resetting my password multiple times and I really need access to pay my bills today. This is really stressing me out."
            ),
            scenario.agent(),
            scenario.judge(),
        ],
    )

    _assert_success(result, "Complex issue test")


@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_urgent_business_scenario():
    result = await scenario.run(
        name="urgent business account problem - GLM",
        description="""
            Business customer has an urgent issue affecting their operations.
            They can't access funds to pay employees. This tests whether the agent
            recognizes urgency and takes appropriate high-priority action.
        """,
        agents=[
            BankSupportAgentAdapter(),
            scenario.UserSimulatorAgent(model="openai/gpt-4o"),
            scenario.JudgeAgent(
                model="openai/gpt-4o",
                criteria=[
                    "Agent immediately recognizes the business urgency and employee impact",
                    "Agent responds with high priority and urgency in tone",
                    "Agent takes concrete action (investigating the freeze or escalating to specialists)",
                    "Agent provides realistic timeline or sets expectations appropriately",
                    "Agent offers interim solutions or workarounds if available",
                ]
            ),
        ],
        script=[
            scenario.user(
                "URGENT: My business account is frozen and I need to pay my employees today. This is costing me money every minute! My business account number is CUST_001."
            ),
            scenario.agent(),
            scenario.user(
                "I can't wait. My payroll is due in 2 hours and my employees are depending on me. What can you do right now?"
            ),
            scenario.agent(),
            scenario.judge(),
        ],
    )

    _assert_success(result, "Urgent business test")


@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_simple_inquiry_no_tools():
    result = await scenario.run(
        name="simple inquiry without tool usage - GLM",
        description="""
            Customer asks a simple question about branch hours or general banking info.
            The agent should answer directly without invoking any tools.
        """,
        agents=[
            BankSupportAgentAdapter(),
            scenario.UserSimulatorAgent(model="openai/gpt-4o"),
            scenario.JudgeAgent(
                model="openai/gpt-4o",
                criteria=[
                    "Agent answers the simple question directly and helpfully",
                    "Agent does not over-complicate the response",
                    "Agent maintains a friendly and professional tone",
                ]
            ),
        ],
        script=[
            scenario.user(
                "What are your customer support hours? I just want to know when I can call if I have an issue."
            ),
            scenario.agent(),
            scenario.judge(),
        ],
    )

    _assert_success(result, "Simple inquiry test")


@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_spending_analysis_request():
    result = await scenario.run(
        name="spending analysis and budgeting help - GLM",
        description="""
            Customer wants to understand their spending patterns and get budgeting advice.
            The agent should use explore_customer_account to analyze their transactions.
        """,
        agents=[
            BankSupportAgentAdapter(),
            scenario.UserSimulatorAgent(model="openai/gpt-4o"),
            scenario.JudgeAgent(
                model="openai/gpt-4o",
                criteria=[
                    "Agent uses account exploration tools to analyze spending",
                    "Agent provides specific insights about spending categories",
                    "Agent offers actionable budgeting advice or recommendations",
                    "Agent is helpful and non-judgmental about spending habits",
                ]
            ),
        ],
        script=[
            scenario.user(
                "My customer ID is CUST_001. I feel like I'm spending too much lately. Can you help me understand where my money is going?"
            ),
            scenario.agent(),
            scenario.user(
                "That's really helpful. Are there any areas where you think I could cut back?"
            ),
            scenario.agent(),
            scenario.judge(),
        ],
    )

    _assert_success(result, "Spending analysis test")


@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_lost_card_replacement():
    result = await scenario.run(
        name="lost card replacement workflow - GLM",
        description="""
            Customer has lost their debit card and needs a replacement.
            Tests whether the agent handles the card replacement process properly
            including immediate security measures.
        """,
        agents=[
            BankSupportAgentAdapter(),
            scenario.UserSimulatorAgent(model="openai/gpt-4o"),
            scenario.JudgeAgent(
                model="openai/gpt-4o",
                criteria=[
                    "Agent treats lost card with appropriate urgency",
                    "Agent suggests freezing or blocking the lost card immediately",
                    "Agent explains the replacement card process and timeline",
                    "Agent asks about any unauthorized transactions since the card was lost",
                    "Agent reassures the customer about account security",
                ]
            ),
        ],
        script=[
            scenario.user(
                "I lost my debit card somewhere yesterday. I've looked everywhere and can't find it. My customer ID is CUST_001."
            ),
            scenario.agent(),
            scenario.user(
                "I don't think anyone has used it, but I'm not sure. Can you check and get me a new card?"
            ),
            scenario.agent(),
            scenario.judge(),
        ],
    )

    _assert_success(result, "Lost card replacement test")


@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_overdraft_fee_dispute():
    result = await scenario.run(
        name="overdraft fee dispute and resolution - GLM",
        description="""
            Customer with a basic checking account notices an overdraft fee and wants
            it reversed. Tests empathy, account investigation, and fee resolution.
        """,
        agents=[
            BankSupportAgentAdapter(),
            scenario.UserSimulatorAgent(model="openai/gpt-4o"),
            scenario.JudgeAgent(
                model="openai/gpt-4o",
                criteria=[
                    "Agent shows empathy for the customer's frustration about the fee",
                    "Agent investigates the account to understand the overdraft situation",
                    "Agent explains how the overdraft fee occurred",
                    "Agent offers a resolution path (fee waiver, escalation, or explanation)",
                    "Agent suggests ways to avoid future overdraft fees",
                ]
            ),
        ],
        script=[
            scenario.user(
                "I just saw a $35 overdraft fee on my account and I'm really upset. I had money in there! My customer ID is CUST_002."
            ),
            scenario.agent(),
            scenario.user(
                "This isn't fair. I've been a customer for 2 years and this is the first time this has happened. Can you waive the fee?"
            ),
            scenario.agent(),
            scenario.judge(),
        ],
    )

    _assert_success(result, "Overdraft fee dispute test")


if __name__ == "__main__":
    asyncio.run(test_fraud_investigation_workflow())
