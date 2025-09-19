"""
Tests for the Customer Explorer Agent

Business scenarios where rich customer data exploration is needed.
"""
import pytest
import os
import sys
import dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scenario
from agents.customer_explorer_agent import explore_customer_context, analyze_customer_behavior

dotenv.load_dotenv()
scenario.configure(default_model="openai/gpt-4o-mini")

class CustomerExplorerAdapter(scenario.AgentAdapter):
    """Adapter for testing customer data exploration"""

    async def call(self, input: scenario.AgentInput) -> scenario.AgentReturnTypes:
        last_message = input.messages[-1]["content"] if input.messages else ""
        customer_id = "CUST_001"

        behavior = analyze_customer_behavior(customer_id)
        rich_experiences = explore_customer_context(customer_id, last_message, input.messages)

        response = f"Customer Analysis for {customer_id}:\n"
        response += f"Spending: ${behavior.get('total_spending_5_days', 0):.2f}\n"
        response += f"Risk Indicators: {len([r for r in behavior.get('risk_indicators', {}).values() if r])}\n"
        response += f"Rich Experiences: {len(rich_experiences)}\n"

        for exp in rich_experiences[:2]:
            response += f"- {exp.title} (Priority: {exp.priority})\n"

        return response

@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_fraud_investigation_analysis():
    result = await scenario.run(
        name="fraud investigation customer analysis",
        description="Customer reports fraud and needs account analysis for security",
        agents=[
            CustomerExplorerAdapter(),
            scenario.UserSimulatorAgent(),
            scenario.JudgeAgent(criteria=[
                "Analysis identifies transaction patterns",
                "Analysis provides security recommendations",
                "Analysis shows appropriate risk assessment",
                "Analysis offers account protection measures"
            ])
        ],
        script=[
            scenario.user("I see charges I didn't make. Can you analyze my account for fraud?"),
            scenario.agent(),
            scenario.judge(),
        ],
    )

    assert result.success

@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_spending_analysis_for_budgeting():
    result = await scenario.run(
        name="spending pattern analysis",
        description="Customer wants spending insights for better budgeting",
        agents=[
            CustomerExplorerAdapter(),
            scenario.UserSimulatorAgent(),
            scenario.JudgeAgent(criteria=[
                "Analysis provides spending breakdown",
                "Analysis identifies saving opportunities",
                "Analysis offers budgeting recommendations",
                "Analysis shows spending trends"
            ])
        ],
        script=[
            scenario.user("I want to save money. Can you analyze my spending patterns?"),
            scenario.agent(),
            scenario.judge(),
        ],
    )

    assert result.success

if __name__ == "__main__":
    import asyncio

    async def run_test():
        print("Running customer explorer test...")
        await test_fraud_investigation_analysis()
        print("Test completed!")

    asyncio.run(run_test())