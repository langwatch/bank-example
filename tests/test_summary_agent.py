"""
Tests for the Summary Agent

These tests cover the business scenarios where conversation summarization is needed:
- Analyzing customer sentiment after difficult conversations
- Identifying key issues for quality assurance
- Providing conversation insights for agent training
- Detecting escalation patterns
"""
import pytest
import os
import sys
import dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scenario
from agents.summary_agent import summarize_conversation

dotenv.load_dotenv()
scenario.configure(default_model="openai/gpt-4o-mini")

class SummaryAgentAdapter(scenario.AgentAdapter):
    """Adapter for testing conversation summarization"""

    async def call(self, input: scenario.AgentInput) -> scenario.AgentReturnTypes:
        # Convert scenario messages to our format
        messages = []
        for i, msg in enumerate(input.messages):
            role = "customer" if i % 2 == 0 else "agent"
            messages.append({
                "role": role,
                "content": msg["content"],
                "timestamp": f"2024-01-15 10:{30+i}:00"
            })

        summary = summarize_conversation(messages)
        return f"Summary: {summary.summary}\nSentiment: {summary.sentiment}\nKey Issues: {', '.join(summary.key_issues)}\nUrgency: {summary.urgency_level}"

@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_fraud_conversation_analysis():
    result = await scenario.run(
        name="fraud conversation summary and analysis",
        description="""
            After a customer reports fraud and the conversation resolves the issue,
            the summary agent should accurately capture the fraud concern, the
            resolution steps taken, and the customer's emotional journey.
        """,
        agents=[
            SummaryAgentAdapter(),
            scenario.UserSimulatorAgent(),
            scenario.JudgeAgent(criteria=[
                "Summary captures the fraud concern clearly",
                "Summary identifies security actions taken",
                "Summary reflects customer's initial worry and final relief",
                "Summary includes key transaction details mentioned",
                "Summary provides actionable insights for future fraud cases"
            ])
        ],
        script=[
            scenario.user("I'm really worried - there are charges on my card I didn't make. I think someone stole my information!"),
            scenario.user("Thank you for helping me freeze the card and starting the investigation. I feel much better now."),
            scenario.agent(),
            scenario.judge(),
        ],
    )

    assert result.success, f"Fraud conversation analysis test failed: {result.failure_reason}"

@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_escalated_conversation_analysis():
    result = await scenario.run(
        name="escalated conversation analysis",
        description="""
            A customer became increasingly frustrated and was escalated to human support.
            The summary should capture the escalation triggers, customer sentiment
            progression, and lessons for preventing similar escalations.
        """,
        agents=[
            SummaryAgentAdapter(),
            scenario.UserSimulatorAgent(),
            scenario.JudgeAgent(criteria=[
                "Summary identifies escalation triggers and reasons",
                "Summary tracks customer frustration progression",
                "Summary suggests improvements to prevent future escalations",
                "Summary captures what resolution attempts were tried",
                "Summary provides clear urgency assessment"
            ])
        ],
        script=[
            scenario.user("I've been trying to resolve this issue for weeks and nobody can help me!"),
            scenario.user("This is ridiculous! I want to speak to a manager RIGHT NOW!"),
            scenario.agent(),
            scenario.judge(),
        ],
    )

    assert result.success, f"Escalated conversation analysis test failed: {result.failure_reason}"

@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_complex_problem_resolution_summary():
    result = await scenario.run(
        name="complex problem resolution summary",
        description="""
            Customer had multiple interconnected banking issues that were
            systematically resolved. Summary should capture all issues,
            resolution steps, and customer satisfaction progression.
        """,
        agents=[
            SummaryAgentAdapter(),
            scenario.UserSimulatorAgent(),
            scenario.JudgeAgent(criteria=[
                "Summary captures all distinct issues mentioned",
                "Summary shows the systematic resolution approach",
                "Summary tracks customer satisfaction improvement",
                "Summary identifies which solutions were most effective",
                "Summary provides insights for handling similar complex cases"
            ])
        ],
        script=[
            scenario.user("I have multiple problems: my online banking is locked, there's a fee I don't understand, and my direct deposit is missing."),
            scenario.user("Thank you for walking me through each issue step by step. Everything is working now and I understand the fee."),
            scenario.agent(),
            scenario.judge(),
        ],
    )

    assert result.success, f"Complex problem summary test failed: {result.failure_reason}"

@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_positive_customer_experience_analysis():
    result = await scenario.run(
        name="positive customer experience analysis",
        description="""
            Customer had an excellent experience with quick problem resolution
            and great service. Summary should capture what made the experience
            positive for training and quality assurance purposes.
        """,
        agents=[
            SummaryAgentAdapter(),
            scenario.UserSimulatorAgent(),
            scenario.JudgeAgent(criteria=[
                "Summary captures positive sentiment and satisfaction",
                "Summary identifies specific actions that pleased the customer",
                "Summary highlights best practices demonstrated",
                "Summary shows efficient problem resolution",
                "Summary provides insights for replicating positive experiences"
            ])
        ],
        script=[
            scenario.user("I need help with my account balance - it looks wrong to me."),
            scenario.user("Wow, that was so helpful! You explained everything clearly and fixed the issue immediately. Thank you!"),
            scenario.agent(),
            scenario.judge(),
        ],
    )

    assert result.success, f"Positive experience analysis test failed: {result.failure_reason}"

@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_sentiment_progression_tracking():
    result = await scenario.run(
        name="customer sentiment progression analysis",
        description="""
            Customer starts frustrated, becomes more upset, then gradually
            becomes satisfied as issues are resolved. Summary should capture
            this emotional journey accurately.
        """,
        agents=[
            SummaryAgentAdapter(),
            scenario.UserSimulatorAgent(),
            scenario.JudgeAgent(criteria=[
                "Summary captures initial customer frustration",
                "Summary notes sentiment progression through conversation",
                "Summary identifies turning points in customer mood",
                "Summary shows final positive resolution",
                "Summary provides insights on managing customer emotions"
            ])
        ],
        script=[
            scenario.user("This is so frustrating! My card keeps getting declined and I don't know why."),
            scenario.user("Okay, I'm starting to understand the issue better now."),
            scenario.user("Perfect! Thank you for being so patient and helpful. My card is working now."),
            scenario.agent(),
            scenario.judge(),
        ],
    )

    assert result.success, f"Sentiment progression test failed: {result.failure_reason}"

if __name__ == "__main__":
    import asyncio

    async def run_test():
        print("Running fraud conversation analysis test...")
        await test_fraud_conversation_analysis()
        print("Summary agent test completed!")

    asyncio.run(run_test())