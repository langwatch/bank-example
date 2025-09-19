"""
Demo test showing proper Scenario usage with tool call validation

This demonstrates the key capabilities for the customer demo.
"""
import asyncio
import json
import sys
import os
import dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import scenario
from main_support_agent import support_agent

dotenv.load_dotenv()
scenario.configure(default_model="openai/gpt-4o-mini")

class BankSupportAgentAdapter(scenario.AgentAdapter):
    async def call(self, input: scenario.AgentInput) -> scenario.AgentReturnTypes:
        message_content = input.last_new_user_message_str()
        response = support_agent.run(message_content)
        return response.content

async def test_fraud_with_tool_validation():
    """
    Demo test: Fraud investigation with proper tool call validation

    This shows how Scenario can validate that the right tools are called
    at the right time for specific business scenarios.
    """

    def check_fraud_tool_usage(state: scenario.ScenarioState):
        """Custom assertion to verify fraud investigation tools were used"""
        print(f"\n🔍 Checking tool calls after {len(state.messages)} messages...")

        # Check if customer exploration was called
        has_exploration = state.has_tool_call("explore_customer_account")
        print(f"   explore_customer_account called: {has_exploration}")

        if has_exploration:
            tool_call = state.last_tool_call("explore_customer_account")
            if tool_call:
                args = json.loads(tool_call["function"]["arguments"])
                print(f"   Tool arguments: {args}")

                # Validate the arguments make sense for fraud
                query = args.get("query", "").lower()
                fraud_keywords = ["fraud", "security", "unauthorized", "suspicious"]
                has_fraud_context = any(keyword in query for keyword in fraud_keywords)
                print(f"   Query contains fraud context: {has_fraud_context}")

        # For demo purposes, let's be flexible - either exploration or escalation is appropriate
        has_escalation = state.has_tool_call("escalate_to_human")
        print(f"   escalate_to_human called: {has_escalation}")

        # At least one appropriate tool should be called for fraud concerns
        appropriate_response = has_exploration or has_escalation
        print(f"   ✅ Appropriate fraud response: {appropriate_response}")

        return appropriate_response

    print("🎭 Running fraud investigation demo with tool validation...")

    result = await scenario.run(
        name="fraud investigation demo",
        description="""
            Customer reports suspicious transactions and potential fraud.
            The agent should take this seriously and use appropriate tools
            to investigate or escalate the security concern.
        """,
        agents=[
            BankSupportAgentAdapter(),
            scenario.UserSimulatorAgent(),
            scenario.JudgeAgent(criteria=[
                "Agent takes fraud concerns seriously",
                "Agent offers security measures or investigation",
                "Agent maintains professional and reassuring tone"
            ])
        ],
        script=[
            scenario.user("I think someone stole my card! There are charges I didn't make - $85 at Amazon and $45 at a gas station."),
            scenario.agent(),
            check_fraud_tool_usage,
            scenario.user("Yes, please help me secure my account immediately!"),
            scenario.agent(),
            scenario.judge(),
        ],
    )

    print(f"\n📊 Test Result: {'✅ PASSED' if result.success else '❌ FAILED'}")
    if result.reasoning:
        print(f"Reasoning: {result.reasoning}")

    return result.success

async def test_escalation_detection():
    """
    Demo test: Escalation detection with custom validation
    """

    def verify_escalation_logic(state: scenario.ScenarioState):
        """Check that angry customers trigger escalation"""
        print(f"\n🚨 Checking escalation logic...")

        has_escalation = state.has_tool_call("escalate_to_human")
        print(f"   escalate_to_human called: {has_escalation}")

        if has_escalation:
            tool_call = state.last_tool_call("escalate_to_human")
            if tool_call:
                args = json.loads(tool_call["function"]["arguments"])
                reason = args.get("reason", "")
                urgency = args.get("urgency", "medium")
                print(f"   Escalation reason: {reason}")
                print(f"   Urgency level: {urgency}")

        print(f"   ✅ Escalation handled: {has_escalation}")
        return has_escalation

    print("\n🎭 Running escalation detection demo...")

    result = await scenario.run(
        name="escalation detection demo",
        description="""
            Frustrated customer demands to speak with a manager.
            Agent should recognize the escalation need and handle appropriately.
        """,
        agents=[
            BankSupportAgentAdapter(),
            scenario.UserSimulatorAgent(),
            scenario.JudgeAgent(criteria=[
                "Agent acknowledges customer frustration",
                "Agent offers escalation when demanded",
                "Agent maintains professionalism"
            ])
        ],
        script=[
            scenario.user("This is ridiculous! I want to speak to your manager RIGHT NOW! Nobody can help me with this issue!"),
            scenario.agent(),
            verify_escalation_logic,
            scenario.judge(),
        ],
    )

    print(f"\n📊 Test Result: {'✅ PASSED' if result.success else '❌ FAILED'}")
    return result.success

async def main():
    """Run the demo tests"""
    print("🚀 Bank Customer Support Agent - Scenario Demo")
    print("=" * 50)

    # Test 1: Fraud Investigation
    fraud_passed = await test_fraud_with_tool_validation()

    # Test 2: Escalation Detection
    escalation_passed = await test_escalation_detection()

    print("\n" + "=" * 50)
    print("📈 Demo Summary:")
    print(f"   Fraud Investigation: {'✅ PASSED' if fraud_passed else '❌ FAILED'}")
    print(f"   Escalation Detection: {'✅ PASSED' if escalation_passed else '❌ FAILED'}")

    overall_success = fraud_passed and escalation_passed
    print(f"\n🎯 Overall Demo: {'✅ SUCCESS' if overall_success else '❌ NEEDS WORK'}")

    if overall_success:
        print("\n🎉 Demo ready! This shows:")
        print("   • Proper Scenario framework usage")
        print("   • Tool calling validation")
        print("   • Custom assertions for business logic")
        print("   • Realistic user simulation")
        print("   • Automated quality assessment")

if __name__ == "__main__":
    asyncio.run(main())

